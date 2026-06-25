import abc
from typing import Union

import numpy as np
import torch
import tqdm


class IdentitySampler:
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        return features


class BaseSampler(abc.ABC):
    def __init__(self, percentage: float):
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)

    def run_with_metadata(self, features, bin_ids):
        sampled = self.run(features)
        raise NotImplementedError("This sampler does not expose selected metadata indices.")


class GreedyCoresetSampler(BaseSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """Greedy Coreset sampling base class."""
        super().__init__(percentage)

        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Subsamples features using Greedy Coreset.

        Args:
            features: [N x D]
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """Computes batchwise Euclidean distances using PyTorch."""
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs iterative greedy coreset selection.

        Args:
            features: [NxD] input feature bank to sample.
        """
        distance_matrix = self._compute_batchwise_differences(features, features)
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """Approximate Greedy Coreset sampling base class."""
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """Runs approximate iterative greedy coreset selection.

        This greedy coreset implementation does not require computation of the
        full N x N distance matrix and thus requires a lot less memory, however
        at the cost of increased sampling times.

        Args:
            features: [NxD] input feature bank to sample.
        """
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


class GeometryAwareCoresetSampler(ApproximateGreedyCoresetSampler):
    """Balanced approximate greedy coreset selection inside geometry bins."""

    def __init__(self, percentage, device, seed=0, number_of_starting_points=10):
        super().__init__(percentage, device, number_of_starting_points)
        self.seed = seed

    def run_with_metadata(self, features, bin_ids):
        is_numpy = isinstance(features, np.ndarray)
        tensor = torch.from_numpy(features) if is_numpy else features
        bins = np.asarray(bin_ids, dtype=np.int64)
        total = max(1, int(len(tensor) * self.percentage))
        populations = {int(b): np.flatnonzero(bins == b) for b in np.unique(bins)}
        quotas = {b: 0 for b in populations}
        active = set(populations)
        remaining = total
        while remaining and active:
            progress = False
            for b in sorted(active):
                if quotas[b] < len(populations[b]) and remaining:
                    quotas[b] += 1
                    remaining -= 1
                    progress = True
            active = {b for b in active if quotas[b] < len(populations[b])}
            if not progress:
                break
        selected = []
        state = np.random.get_state()
        np.random.seed(self.seed)
        try:
            for b in sorted(populations):
                indices = populations[b]
                quota = quotas[b]
                if quota >= len(indices):
                    selected.extend(indices.tolist())
                    continue
                reduced = self._reduce_features(tensor[indices])
                local = self._compute_greedy_coreset_indices_for_count(reduced, quota)
                selected.extend(indices[local].tolist())
        finally:
            np.random.set_state(state)
        selected = np.asarray(selected, dtype=np.int64)
        sampled = tensor[selected]
        if is_numpy:
            sampled = sampled.cpu().numpy()
        return sampled, bins[selected], selected

    def _compute_greedy_coreset_indices_for_count(self, features, count):
        starts = min(self.number_of_starting_points, len(features))
        anchors = np.random.choice(len(features), starts, replace=False).tolist()
        distances = self._compute_batchwise_differences(features, features[anchors]).mean(dim=1, keepdim=True)
        selected = []
        with torch.no_grad():
            for _ in range(count):
                index = torch.argmax(distances).item()
                selected.append(index)
                new_distance = self._compute_batchwise_differences(features, features[index:index + 1])
                distances = torch.minimum(distances, new_distance)
        return np.asarray(selected)


class RandomSampler(BaseSampler):
    def __init__(self, percentage: float):
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Randomly samples input feature collection.

        Args:
            features: [N x D]
        """
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]


class SeededRandomSampler(BaseSampler):
    """Random sampler with an optional fixed seed for reproducibility."""

    def __init__(self, percentage: float, seed: int = None):
        super().__init__(percentage)
        self.seed = seed

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        num_random_samples = int(len(features) * self.percentage)
        # Preserve global numpy RNG state when using a local seed
        if self.seed is not None:
            rng_state = np.random.get_state()
            np.random.seed(self.seed)

        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )

        if self.seed is not None:
            np.random.set_state(rng_state)

        subset_indices = np.array(subset_indices)
        return features[subset_indices]
