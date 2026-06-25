"""Geometry-aware PatchCore implementation for wafer maps."""
import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import patchcore.backbones
import patchcore.common
import patchcore.sampler
from patchcore.geometry import GeometryConfig, patch_geometry
from patchcore.networks.embedding_adapter import ResidualEmbeddingAdapter
from patchcore.patchcore import PatchMaker


class PatchCore(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def load(self, backbone, layers_to_extract_from, device, input_shape,
             pretrain_embed_dimension, target_embed_dimension, patchsize=3,
             patchstride=1, anomaly_score_num_nn=1, featuresampler=None,
             nn_method=None, cbam_checkpoint=None, embedding_adapter_path=None,
             radial_bins=4, angular_bins=8, min_wafer_coverage=0.5,
             geometry_radial_neighbors=1, geometry_angular_neighbors=1,
             cbam_state=None, adapter_state=None, **kwargs):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = list(layers_to_extract_from)
        self.input_shape = tuple(input_shape)
        self.device = device
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.featuresampler = featuresampler or patchcore.sampler.IdentitySampler()
        self.geometry_config = GeometryConfig(
            radial_bins, angular_bins, min_wafer_coverage, 0.2,
            geometry_radial_neighbors, geometry_angular_neighbors)
        self.forward_modules = torch.nn.ModuleDict()
        extractor = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, device)
        self.feature_dimensions = extractor.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = extractor
        self.forward_modules["preprocessing"] = patchcore.common.Preprocessing(
            self.feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preadapt_aggregator"] = patchcore.common.Aggregator(
            target_dim=target_embed_dimension).to(device)
        self.target_embed_dimension = target_embed_dimension
        self._load_cbam(cbam_checkpoint, cbam_state)
        self._load_adapter(embedding_adapter_path, adapter_state)
        for parameter in self.forward_modules.parameters():
            parameter.requires_grad_(False)
        self.forward_modules.eval()
        self.anomaly_scorer = patchcore.common.GeometryNearestNeighbourScorer(
            anomaly_score_num_nn, self.geometry_config,
            bool(getattr(nn_method, "on_gpu", False)), 4)
        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=device, target_size=input_shape[-2:])

    def _load_cbam(self, path, state):
        if path is None and state is None:
            return
        from patchcore.networks.cbam import CBAM
        payload = state or torch.load(path, map_location=self.device)
        if payload.get("layers") != self.layers_to_extract_from:
            raise ValueError("CBAM checkpoint layers do not match.")
        if payload.get("dimensions") != self.feature_dimensions:
            raise ValueError("CBAM checkpoint dimensions do not match.")
        modules = torch.nn.ModuleList([
            CBAM(dim, payload.get("reduction", 16), payload.get("spatial_kernel", 7))
            for dim in self.feature_dimensions]).to(self.device)
        modules.load_state_dict(payload["state_dict"])
        modules.eval()
        self.forward_modules["cbam"] = modules
        self.cbam_metadata = {key: payload.get(key) for key in (
            "backbone", "layers", "dimensions", "resize", "imagesize",
            "reduction", "spatial_kernel")}

    def _load_adapter(self, path, state):
        if path is None and state is None:
            return
        payload = state or torch.load(path, map_location=self.device)
        if payload.get("dimension") != self.target_embed_dimension:
            raise ValueError("Embedding adapter dimension does not match.")
        module = ResidualEmbeddingAdapter(
            self.target_embed_dimension, payload.get("dropout", 0.1)).to(self.device)
        module.load_state_dict(payload["state_dict"])
        module.eval()
        self.forward_modules["embedding_adapter"] = module
        self.adapter_metadata = {
            "dimension": self.target_embed_dimension,
            "dropout": payload.get("dropout", 0.1)}

    def extract_feature_maps(self, images):
        features = self.forward_modules["feature_aggregator"](images)
        if "cbam" in self.forward_modules:
            for index, layer in enumerate(self.layers_to_extract_from):
                features[layer] = self.forward_modules["cbam"][index](features[layer])
        return features

    def _embed(self, images, raw_images=None, detach=True, provide_patch_shapes=False):
        raw_images = images if raw_images is None else raw_images
        with torch.no_grad():
            maps = self.extract_feature_maps(images)
            patched = [self.patch_maker.patchify(
                maps[layer], return_spatial_info=True)
                for layer in self.layers_to_extract_from]
            shapes = [item[1] for item in patched]
            features = [item[0] for item in patched]
            reference = shapes[0]
            for index in range(1, len(features)):
                value = features[index]
                source = shapes[index]
                value = value.reshape(
                    value.shape[0], source[0], source[1], *value.shape[2:]
                ).permute(0, -3, -2, -1, 1, 2)
                base = value.shape
                value = value.reshape(-1, *value.shape[-2:])
                value = F.interpolate(
                    value.unsqueeze(1), size=reference,
                    mode="bilinear", align_corners=False).squeeze(1)
                value = value.reshape(*base[:-2], *reference).permute(
                    0, -2, -1, 1, 2, 3)
                features[index] = value.reshape(
                    len(value), -1, *value.shape[-3:])
            features = [value.reshape(-1, *value.shape[-3:]) for value in features]
            embeddings = self.forward_modules["preadapt_aggregator"](
                self.forward_modules["preprocessing"](features))
            if "embedding_adapter" in self.forward_modules:
                embeddings = self.forward_modules["embedding_adapter"](embeddings)
            geometry = patch_geometry(
                raw_images, tuple(reference), self.geometry_config)
        if detach:
            embeddings = embeddings.detach().cpu().numpy()
            geometry = {
                key: value.detach().cpu().numpy()
                if torch.is_tensor(value) else value
                for key, value in geometry.items()}
        return (embeddings, shapes, geometry) if provide_patch_shapes else embeddings

    def fit(self, training_data):
        feature_sets, bin_sets = [], []
        for batch in tqdm.tqdm(
                training_data, desc="Computing support features...", leave=False):
            images = batch["image"] if isinstance(batch, dict) else batch
            raw = batch.get("raw_image", images) if isinstance(batch, dict) else images
            features, _, geometry = self._embed(
                images.float().to(self.device), raw.float().to(self.device),
                provide_patch_shapes=True)
            valid = geometry["valid"].reshape(-1)
            feature_sets.append(features[valid])
            bin_sets.append(geometry["bin_ids"].reshape(-1)[valid])
        if not isinstance(
                self.featuresampler, patchcore.sampler.GeometryAwareCoresetSampler):
            raise ValueError("Geometry PatchCore requires geometry_coreset.")
        features, bins, _ = self.featuresampler.run_with_metadata(
            np.concatenate(feature_sets).astype(np.float32),
            np.concatenate(bin_sets).astype(np.int64))
        self.anomaly_scorer.fit(features, bins)

    def predict(self, data, return_segmentations=True):
        if not isinstance(data, torch.utils.data.DataLoader):
            return self._predict(data)
        scores, masks, labels, ground_truth = [], [], [], []
        for batch in tqdm.tqdm(data, desc="Inferring...", leave=False):
            labels.extend(batch["is_anomaly"].numpy().tolist())
            if return_segmentations:
                ground_truth.extend(batch["mask"].numpy().tolist())
            batch_scores, batch_masks = self._predict(
                batch["image"], batch.get("raw_image", batch["image"]), return_segmentations)
            scores.extend(batch_scores)
            if return_segmentations:
                masks.extend(batch_masks)
        return scores, masks, labels, ground_truth

    def _predict(self, images, raw_images=None, return_segmentations=True):
        images = images.float().to(self.device)
        raw = images if raw_images is None else raw_images.float().to(self.device)
        features, shapes, geometry = self._embed(
            images, raw, provide_patch_shapes=True)
        scores = self.anomaly_scorer.predict(
            features, geometry["bin_ids"].reshape(-1),
            geometry["valid"].reshape(-1))
        scores = scores.reshape(len(images), -1)
        image_scores = scores.max(axis=1)
        if not return_segmentations:
            return list(image_scores), []
        grid = shapes[0]
        masks = self.anomaly_segmentor.convert_to_segmentation(
            scores.reshape(len(images), grid[0], grid[1]))
        masks = [
            mask * wafer for mask, wafer
            in zip(masks, geometry["wafer_mask"][:, 0])]
        return list(image_scores), masks

    def save_to_path(self, path, prepend=""):
        os.makedirs(path, exist_ok=True)
        self.anomaly_scorer.save(path, prepend)
        params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension":
                self.forward_modules["preprocessing"].output_dim,
            "target_embed_dimension": self.target_embed_dimension,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_score_num_nn":
                self.anomaly_scorer.n_nearest_neighbours,
            "radial_bins": self.geometry_config.radial_bins,
            "angular_bins": self.geometry_config.angular_bins,
            "min_wafer_coverage": self.geometry_config.min_wafer_coverage,
            "geometry_radial_neighbors":
                self.geometry_config.radial_neighbors,
            "geometry_angular_neighbors":
                self.geometry_config.angular_neighbors}
        with open(os.path.join(
                path, prepend + "patchcore_params.pkl"), "wb") as handle:
            pickle.dump(params, handle, pickle.HIGHEST_PROTOCOL)
        modules = {}
        if "cbam" in self.forward_modules:
            modules["cbam_state"] = {
                **self.cbam_metadata,
                "state_dict": self.forward_modules["cbam"].state_dict()}
        if "embedding_adapter" in self.forward_modules:
            modules["adapter_state"] = {
                **self.adapter_metadata,
                "state_dict":
                    self.forward_modules["embedding_adapter"].state_dict()}
        torch.save(
            modules, os.path.join(path, prepend + "patchcore_modules.pth"))

    def load_from_path(self, path, device, nn_method=None, prepend=""):
        with open(os.path.join(
                path, prepend + "patchcore_params.pkl"), "rb") as handle:
            params = pickle.load(handle)
        name = params.pop("backbone.name")
        params["backbone"] = patchcore.backbones.load(name)
        params["backbone"].name = name
        params.update(torch.load(
            os.path.join(path, prepend + "patchcore_modules.pth"),
            map_location=device))
        self.load(**params, device=device, nn_method=nn_method)
        self.anomaly_scorer.load(path, prepend)
