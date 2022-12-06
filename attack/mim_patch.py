from .projected_gradient_descent import ProjectedGradientDescent
import numpy as np
from typing import Optional, Union, TYPE_CHECKING
from art.config import ART_NUMPY_DTYPE
from tqdm import trange
from art.utils import (
    compute_success,
    get_labels_np_array,
    random_sphere,
    projection,
    check_and_transform_label_format,
)


class MIMPatch(ProjectedGradientDescent):
    """
    Apply Masked PGD to image and video inputs,
    where images are assumed to have shape (NHWC)
    and video are assumed to have shape (NFHWC)
    """

    def __init__(self, estimator, **kwargs):
        self._project = True
        super().__init__(estimator=estimator, **kwargs)

    def _check_compatibility_input_and_eps(self, x: np.ndarray):
        """
        Check the compatibility of the input with `eps` and `eps_step` which are of the same shape.

        :param x: An array with the original inputs.
        """
        if isinstance(self.eps, np.ndarray):
            # Ensure the eps array is broadcastable
            if self.eps.ndim > x.ndim:
                raise ValueError("The `eps` shape must be broadcastable to input shape.")

    def _set_targets(self, x: np.ndarray, y: np.ndarray, classifier_mixin: bool = True) -> np.ndarray:
        """
        Check and set up targets.

        :param x: An array with the original inputs.
        :param y: Target values (class labels) one-hot-encoded of shape `(nb_samples, nb_classes)` or indices of shape
                  (nb_samples,). Only provide this parameter if you'd like to use true labels when crafting adversarial
                  samples. Otherwise, model predictions are used as labels to avoid the "label leaking" effect
                  (explained in this paper: https://arxiv.org/abs/1611.01236). Default is `None`.
        :param classifier_mixin: Whether the estimator is of type `ClassifierMixin`.
        :return: The targets.
        """
        if classifier_mixin:
            y = check_and_transform_label_format(y, self.estimator.nb_classes)

        if y is None:
            # Throw error if attack is targeted, but no targets are provided
            if self.targeted:
                raise ValueError("Target labels `y` need to be provided for a targeted attack.")

            # Use model predictions as correct outputs
            if classifier_mixin:
                targets = get_labels_np_array(self.estimator.predict(x, batch_size=self.batch_size))
            else:
                targets = self.estimator.predict(x, batch_size=self.batch_size)

        else:
            targets = y

        return targets

    def _random_eps(self):
        """
        Check whether random eps is enabled, then scale eps and eps_step appropriately.
        """
        if self.random_eps:
            ratio = self.eps_step / self.eps

            if isinstance(self.eps, (int, float)):
                self.eps = np.round(self.norm_dist.rvs(1)[0], 10)
            else:
                self.eps = np.round(self.norm_dist.rvs(size=self.eps.shape), 10)
            self.eps_step = ratio * self.eps

    def generate(self, x, y=None, mask=None, **generate_kwargs):
        video_input = generate_kwargs.get("video_input", False)

        if "ymin" in generate_kwargs:
            ymin = generate_kwargs["ymin"]

        if "xmin" in generate_kwargs:
            xmin = generate_kwargs["xmin"]

        assert x.ndim in [
            4,
            5,
        ], "This attack is designed for images (4-dim) and videos (5-dim)"

        channels_mask = generate_kwargs.get(
            "mask", np.ones(x.shape[-1], dtype=np.float32)
        )
        channels = np.where(channels_mask)[0]
        if mask is None:
            mask = np.zeros(shape=x.shape[1:], dtype=np.float32)
            if "patch_ratio" in generate_kwargs:
                patch_ratio = generate_kwargs["patch_ratio"]
                ymax = ymin + int(x.shape[-3] * patch_ratio ** 0.5)
                xmax = xmin + int(x.shape[-2] * patch_ratio ** 0.5)
                if video_input:
                    mask[:, ymin:ymax, xmin:xmax, channels] = 1.0
                else:
                    mask[ymin:ymax, xmin:xmax, channels] = 1.0
            elif "patch_height" in generate_kwargs and "patch_width" in generate_kwargs:
                patch_height = generate_kwargs["patch_height"]
                patch_width = generate_kwargs["patch_width"]
                if video_input:
                    mask[
                    :, ymin: ymin + patch_height, xmin: xmin + patch_width, channels
                    ] = 1.0
                else:
                    mask[
                    ymin: ymin + patch_height, xmin: xmin + patch_width, channels
                    ] = 1.0
            elif "d" in generate_kwargs:
                d = generate_kwargs["d"]
                radius = d // 2
                xcenter = int(xmin + radius)
                ycenter = int(ymin + radius)
                if video_input:
                    raise ValueError(
                        "not implemented"
                    )
                else:
                    h = mask.shape[0]
                    w = mask.shape[1]
                    Y, X = np.ogrid[:h, :w]
                    dist_from_center = np.sqrt((X - xcenter) ** 2 + (Y - ycenter) ** 2)
                    circle_mask = (dist_from_center <= radius).astype(float)
                    mask = np.stack([circle_mask, circle_mask, circle_mask], axis=-1)

            else:
                raise ValueError(
                    "generate_kwargs did not define 'patch_ratio', or it did not define 'patch_height' and 'patch_width'"
                )
        self._check_compatibility_input_and_eps(x=x)
        self._random_eps()

        if self.num_random_init > 0:
            raise ValueError("Random initialisation is only supported for classification.")

        # Set up targets
        targets = self._set_targets(x, y, classifier_mixin=False)

        # Start to compute adversarial examples
        if x.dtype == np.object:
            adv_x = x.copy()
        else:
            adv_x = x.astype(ART_NUMPY_DTYPE)

        grad = np.zeros_like(x)
        for i_max_iter in trange(self.max_iter, desc="MIM - Iterations", disable=not self.verbose):
            adv_x, grad = self._compute(
                adv_x,
                x,
                targets,
                grad,
                mask,
                self.eps,
                self.eps_step,
                self._project,
                self.num_random_init > 0 and i_max_iter == 0,
            )

        return adv_x

    def _compute(
            self,
            x: np.ndarray,
            x_init: np.ndarray,
            y: np.ndarray,
            grad: np.ndarray,
            mask: Optional[np.ndarray],
            eps: Union[int, float, np.ndarray],
            eps_step: Union[int, float, np.ndarray],
            project: bool,
            random_init: bool,
    ):
        if random_init:
            n = x.shape[0]
            m = np.prod(x.shape[1:]).item()
            random_perturbation = random_sphere(n, m, eps, self.norm).reshape(x.shape).astype(ART_NUMPY_DTYPE)
            if mask is not None:
                random_perturbation = random_perturbation * (mask.astype(ART_NUMPY_DTYPE))
            x_adv = x.astype(ART_NUMPY_DTYPE) + random_perturbation

            if self.estimator.clip_values is not None:
                clip_min, clip_max = self.estimator.clip_values
                x_adv = np.clip(x_adv, clip_min, clip_max)
        else:
            if x.dtype == np.object:
                x_adv = x.copy()
            else:
                x_adv = x.astype(ART_NUMPY_DTYPE)

        if grad is None:
            grad = x.copy()
        # Compute perturbation with implicit batching
        for batch_id in range(int(np.ceil(x.shape[0] / float(self.batch_size)))):
            batch_index_1, batch_index_2 = batch_id * self.batch_size, (batch_id + 1) * self.batch_size
            batch_index_2 = min(batch_index_2, x.shape[0])
            batch = x_adv[batch_index_1:batch_index_2]

            # pdb.set_trace()

            # batch_labels = y[batch_index_1:batch_index_2][0]   ##for drawing teaser img
            batch_labels = [y.item()]

            mask_batch = mask
            if mask is not None:
                # Here we need to make a distinction: if the masks are different for each input, we need to index
                # those for the current batch. Otherwise (i.e. mask is meant to be broadcasted), keep it as it is.
                if len(mask.shape) == len(x.shape):
                    mask_batch = mask[batch_index_1:batch_index_2]

            # Get perturbation
            perturbation = self._compute_perturbation(batch, batch_labels, mask_batch)

            # momentum gradient
            grad = grad + perturbation
            perturbation = np.sign(grad)

            # Compute batch_eps and batch_eps_step
            if isinstance(eps, np.ndarray) and isinstance(eps_step, np.ndarray):
                if len(eps.shape) == len(x.shape) and eps.shape[0] == x.shape[0]:
                    batch_eps = eps[batch_index_1:batch_index_2]
                    batch_eps_step = eps_step[batch_index_1:batch_index_2]

                else:
                    batch_eps = eps
                    batch_eps_step = eps_step

            else:
                batch_eps = eps
                batch_eps_step = eps_step

            # Apply perturbation and clip
            x_adv[batch_index_1:batch_index_2] = self._apply_perturbation(batch, perturbation, batch_eps_step)

            if project:
                if x_adv.dtype == np.object:
                    for i_sample in range(batch_index_1, batch_index_2):
                        if isinstance(batch_eps, np.ndarray) and batch_eps.shape[0] == x_adv.shape[0]:
                            perturbation = projection(
                                x_adv[i_sample] - x_init[i_sample], batch_eps[i_sample], self.norm
                            )

                        else:
                            perturbation = projection(x_adv[i_sample] - x_init[i_sample], batch_eps, self.norm)

                        x_adv[i_sample] = x_init[i_sample] + perturbation

                else:
                    perturbation = projection(
                        x_adv[batch_index_1:batch_index_2] - x_init[batch_index_1:batch_index_2], batch_eps, self.norm
                    )
                    x_adv[batch_index_1:batch_index_2] = x_init[batch_index_1:batch_index_2] + perturbation

        return x_adv, grad.copy()

    def _compute_perturbation(
            self, batch: np.ndarray, batch_labels: np.ndarray, mask: Optional[np.ndarray]
    ) -> np.ndarray:
        # Pick a small scalar to avoid division by 0
        tol = 10e-8

        # Get gradient wrt loss; invert it if attack is targeted
        grad = self.estimator.loss_gradient(batch, batch_labels) * (1 - 2 * int(self.targeted))

        # Check for NaN before normalisation an replace with 0
        if np.isnan(grad).any():
            logger.warning("Elements of the loss gradient are NaN and have been replaced with 0.0.")
            grad = np.where(np.isnan(grad), 0.0, grad)

        # Apply mask
        if mask is not None:
            grad = np.where(mask == 0.0, 0.0, grad)

        # Apply l1 norm bound
        def _apply_norm(grad, object_type=False):
            if np.isinf(grad).any():
                logger.info("The loss gradient array contains at least one positive or negative infinity.")

            if not object_type:
                ind = tuple(range(1, len(batch.shape)))
            else:
                ind = None
            grad = grad / (np.sum(np.abs(grad), axis=ind, keepdims=True) + tol)
            return grad

        if batch.dtype == np.object:
            for i_sample in range(batch.shape[0]):
                grad[i_sample] = _apply_norm(grad[i_sample], object_type=True)
                assert batch[i_sample].shape == grad[i_sample].shape
        else:
            grad = _apply_norm(grad)

        assert batch.shape == grad.shape

        return grad

    def _apply_perturbation(
            self, batch: np.ndarray, perturbation: np.ndarray, eps_step: Union[int, float, np.ndarray]
    ) -> np.ndarray:

        perturbation_step = eps_step * perturbation
        perturbation_step[np.isnan(perturbation_step)] = 0
        batch = batch + perturbation_step
        if self.estimator.clip_values is not None:
            clip_min, clip_max = self.estimator.clip_values
            batch = np.clip(batch, clip_min, clip_max)

        return batch
