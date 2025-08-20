import torch
import torch.nn as nn
import math

PI = math.pi
sqrt = math.sqrt


class RealSphericalHarmonics(nn.Module):
    def __init__(self, degrees: list[int]):
        super().__init__()
        max_l = max(degrees)
        assert 0 <= max_l <= 4, "This implementation supports l_max in [0, 4]"
        self.degrees = degrees

    def forward(self, vecs: torch.Tensor) -> torch.Tensor:
        assert vecs.shape[-1] == 3, "Input must have shape [batch, 3]"
        x, y, z = torch.unbind(
            torch.nn.functional.normalize(vecs, dim=-1), dim=-1
        )
        out = []
        for degree in self.degrees:
            # l = 0
            if degree == 0:
                out.append(0.5 * sqrt(1 / PI) * torch.ones_like(x))  # Y_00

            # l = 1
            if degree == 1:
                out += [
                    sqrt(3 / (4 * PI)) * y,  # Y_1^-1
                    sqrt(3 / (4 * PI)) * z,  # Y_10
                    sqrt(3 / (4 * PI)) * x,  # Y_11
                ]

            # l = 2
            if degree == 2:
                out += [
                    0.5 * sqrt(15 / PI) * x * y,  # Y_2^-2
                    0.5 * sqrt(15 / PI) * y * z,  # Y_2^-1
                    0.25 * sqrt(5 / PI) * (3 * z**2 - 1),  # Y_20
                    0.5 * sqrt(15 / PI) * x * z,  # Y_21
                    0.25 * sqrt(15 / PI) * (x**2 - y**2),  # Y_22
                ]

            # l = 3
            if degree == 3:
                out += [
                    0.25
                    * sqrt(35 / (2 * PI))
                    * y
                    * (3 * x**2 - y**2),  # Y_3^-3
                    0.5 * sqrt(105 / PI) * x * y * z,  # Y_3^-2
                    0.25 * sqrt(21 / (2 * PI)) * y * (5 * z**2 - 1),  # Y_3^-1
                    0.25 * sqrt(7 / PI) * (5 * z**3 - 3 * z),  # Y_30
                    0.25 * sqrt(21 / (2 * PI)) * x * (5 * z**2 - 1),  # Y_31
                    0.25 * sqrt(105 / PI) * (x**2 - y**2) * z,  # Y_32
                    0.25 * sqrt(35 / (2 * PI)) * x * (x**2 - 3 * y**2),  # Y_33
                ]

            # l = 4
            if degree == 4:
                out += [
                    0.75 * sqrt(35 / PI) * x * y * (x**2 - y**2),  # Y_4^-4
                    0.75
                    * sqrt(35 / (2 * PI))
                    * y
                    * (3 * x**2 - y**2)
                    * z,  # Y_4^-3
                    0.75 * sqrt(5 / PI) * x * y * (7 * z**2 - 1),  # Y_4^-2
                    0.75
                    * sqrt(5 / (2 * PI))
                    * y
                    * (7 * z**3 - 3 * z),  # Y_4^-1
                    0.1875
                    * sqrt(1 / PI)
                    * (35 * z**4 - 30 * z**2 + 3),  # Y_40
                    0.75 * sqrt(5 / (2 * PI)) * x * (7 * z**3 - 3 * z),  # Y_41
                    0.375
                    * sqrt(5 / PI)
                    * (x**2 - y**2)
                    * (7 * z**2 - 1),  # Y_42
                    0.75
                    * sqrt(35 / (2 * PI))
                    * x
                    * (x**2 - 3 * y**2)
                    * z,  # Y_43
                    0.1875
                    * sqrt(35 / PI)
                    * (
                        x**2 * (x**2 - 3 * y**2) - y**2 * (3 * x**2 - y**2)
                    ),  # Y_44
                ]

        return torch.stack(out, dim=-1)  # shape: [batch, (l_max+1)**2]
