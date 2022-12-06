from .projected_gradient_descent import ProjectedGradientDescent
import numpy as np


class PGDPatch(ProjectedGradientDescent):
    """
    Apply Masked PGD to image and video inputs,
    where images are assumed to have shape (NHWC)
    and video are assumed to have shape (NFHWC)
    """

    def __init__(self, estimator, **kwargs):
        super().__init__(estimator=estimator, **kwargs)

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
            # rectangle and square attack
            elif "patch_height" in generate_kwargs and "patch_width" in generate_kwargs:
                patch_height = generate_kwargs["patch_height"]
                patch_width = generate_kwargs["patch_width"]
                if video_input:
                    mask[
                        :, ymin : ymin + patch_height, xmin : xmin + patch_width, channels
                    ] = 1.0
                else:
                    mask[
                        ymin : ymin + patch_height, xmin : xmin + patch_width, channels
                    ] = 1.0
            # circle attack
            elif "d" in generate_kwargs:
                d = generate_kwargs["d"]
                radius = d/2
                xcenter = xmin + radius
                ycenter = ymin + radius
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
            # ellips attack
            elif "a" in generate_kwargs:
                a = generate_kwargs["a"]
                b = generate_kwargs["b"]
                xcenter = xmin + a
                ycenter = ymin + b
                if video_input:
                    raise ValueError(
                        "not implemented"
                    )
                else:
                    h = mask.shape[0]
                    w = mask.shape[1]
                    Y, X = np.ogrid[:h, :w]
                    dist_from_center = (X - xcenter) ** 2/a**2 + (Y - ycenter) ** 2/b**2
                    e_mask = (dist_from_center <= 1).astype(float)
                    mask = np.stack([e_mask, e_mask, e_mask], axis=-1)
            else:
                raise ValueError(
                    "generate_kwargs did not define 'patch_ratio', or it did not define 'patch_height' and 'patch_width'"
                )
        return super().generate(x, y=y, mask=mask)
