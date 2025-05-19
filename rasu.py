from typing import Iterator, Optional, Sequence

import gemmi
import numpy as np
import torch
from reciprocalspaceship.decorators import cellify, spacegroupify
from reciprocalspaceship.utils import (
    apply_to_hkl,
    generate_reciprocal_asu,
    generate_reciprocal_cell,
)

from abismal_torch.symmetry.op import Op


class ReciprocalASU(torch.nn.Module):
    @cellify
    @spacegroupify
    def __init__(
        self,
        cell: gemmi.UnitCell,
        spacegroup: gemmi.SpaceGroup,
        dmin: float,
        anomalous: Optional[bool] = True,
        **kwargs
    ) -> None:
        """
        Base Layer that maps observed reflections to the reciprocal asymmetric unit (rasu).

        Args:
            cell (gemmi.UnitCell): Unit cell parameters.
            spacegroup (gemmi.SpaceGroup): Space group.
            dmin (float): Highest resolution in Ångstroms.
            anomalous (bool, optional): If true, treat Friedel mates as non-redudant.

        Attributes:
            rasu_size (int): Number of unique reflections in the rasu. This variable is previously
                named as asu_size.
            Hmax (np.array(np.int32)): Maximum h, k, and l in the rasu under the given dmin.
            H_rasu (torch.Tensor): Unique Miller indices in the rasu. Registered as a buffer.
            centric (torch.Tensor): A boolean tensor indicating if each unique reflection
                is centric. Registered as a buffer.
            multiplicity (torch.Tensor): A float tensor with the multiplicity of each unique
                reflection. This variable is previously named as epsilon. Registered as a buffer.
            reflection_id_grid (torch.Tensor): A 3D tensor that contains the 1D integer
                ID of each unique reflection. ID value ranges from 0 to rasu_size, and is -1 if the
                reflection is not in the rasu due to dmin cutoff, systematic absences, etc. This
                variable is previously named as miller_id. Registered as a buffer.

        """
        super().__init__(**kwargs)
        self.cell = cell
        self.spacegroup = spacegroup
        self.anomalous = anomalous
        self.dmin = dmin

        H_rasu = generate_reciprocal_asu(cell, spacegroup, dmin, anomalous)
        self.rasu_size = len(H_rasu)
        self.register_buffer("H_rasu", torch.tensor(H_rasu, dtype=torch.int32))

        go = spacegroup.operations()
        self.Hmax = np.array(cell.get_hkl_limits(dmin))
        hmax, kmax, lmax = self.Hmax
        reflection_id_grid = -np.ones(
            (2 * hmax + 1, 2 * kmax + 1, 2 * lmax + 1), dtype=np.int32
        )
        for op in go:
            Hop = apply_to_hkl(H_rasu, op)
            h, k, l = Hop.T
            reflection_id_grid[h, k, l] = np.arange(self.rasu_size)
            if not anomalous:
                h, k, l = -Hop.T
                reflection_id_grid[h, k, l] = np.arange(self.rasu_size)
        self.register_buffer(
            "reflection_id_grid",
            torch.tensor(reflection_id_grid, dtype=torch.int32),
        )

        self.register_buffer(
            "centric",
            torch.tensor(go.centric_flag_array(H_rasu), dtype=torch.bool),
        )
        self.register_buffer(
            "multiplicity",
            torch.tensor(go.epsilon_factor_array(H_rasu), dtype=torch.float32),
        )

    def gather(self, source: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
            source (torch.Tensor): A tensor of shape (self.rasu_size, ...)
            H (torch.Tensor): A tensor of shape (n_refln, 3)

        Returns:
            gathered (torch.Tensor): A tensor of shape (n_refln, ...)
        """
        idx = self.reflection_id_grid[tuple(H.T)]
        gathered = source[idx]
        return gathered


class ReciprocalASUCollection(torch.nn.Module):
    def __init__(self, *rasus: ReciprocalASU, **kwargs) -> None:
        """
        A collection of rasu objects.

        Args:
            *rasus (ReciprocalASU): arbitrary number of rasu objects.

        Attributes:
            reciprocal_asus (tuple[ReciprocalASU]): A tuple of rasu objects.
            Hmax (np.array(np.int32)): Maximum h, k, and l of all asus.
            rac_size (int): Total number of unique reflections in all asus.
            H_rasu (torch.Tensor): A tensor of shape (rac_size, 3) that contains the
                Miller indices of all unique reflections in all asus. Registered as a buffer.
            rasu_ids (torch.Tensor): A tensor of shape (rac_size,) that contains the
                rasu ID of each reflection. Registered as a buffer.
            centric (torch.Tensor): A tensor of shape (rac_size,) that contains the
                centric flag of each reflection. Registered as a buffer.
            multiplicity (torch.Tensor): A tensor of shape (rac_size,) that contains
                the multiplicity of each reflection. Registered as a buffer.
            reflection_id_grid (torch.Tensor): A tensor of shape (n_asus, 2*h+1, 2*k+1, 2*l+1)
                that contains the reflection ID of each reflection. Registered as a buffer.
        """
        super().__init__(**kwargs)
        self.reciprocal_asus = tuple(rasus)
        rasu_ids, centric, multiplicity = [], [], []
        H_rasu = []
        self.rac_size = 0
        self.Hmax = np.zeros(3, dtype=np.int32)

        for rasu_id, rasu in enumerate(self.reciprocal_asus):
            centric.append(rasu.centric)
            multiplicity.append(rasu.multiplicity)
            self.rac_size += rasu.rasu_size
            self.Hmax = np.maximum(self.Hmax, rasu.Hmax)
            rasu_ids.append(torch.full((rasu.rasu_size,), rasu_id, dtype=torch.int32))
            H_rasu.append(rasu.H_rasu)
        self.register_buffer("centric", torch.cat(centric))
        self.register_buffer("multiplicity", torch.cat(multiplicity))
        self.register_buffer("rasu_ids", torch.cat(rasu_ids))
        self.register_buffer("H_rasu", torch.cat(H_rasu))

        h, k, l = self.Hmax
        self.register_buffer(
            "reflection_id_grid",
            torch.full(
                (len(self), 2 * h + 1, 2 * k + 1, 2 * l + 1), -1, dtype=torch.int32
            ),
        )
        offset = 0
        for rasu_id, rasu in enumerate(self.reciprocal_asus):
            Hcell = generate_reciprocal_cell(rasu.cell, dmin=rasu.dmin)
            h, k, l = Hcell.T
            self.reflection_id_grid[torch.tensor(rasu_id, dtype=torch.int),torch.tensor(h, dtype=torch.int), torch.tensor(k, dtype=torch.int), torch.tensor(l, dtype=torch.int)] = (
                rasu.reflection_id_grid[torch.tensor(h, dtype=torch.int), torch.tensor(k, dtype=torch.int), torch.tensor(l, dtype=torch.int)] + offset
            )
            # self.reflection_id_grid[rasu_id, h, k, l] = (
            #     rasu.reflection_id_grid[h,k,l] + offset
            # )

            offset += rasu.rasu_size

    def gather(
        self, source: torch.Tensor, rasu_id: torch.Tensor, H: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters:
            source (torch.Tensor): A tensor of shape (rac_size, ...).
            rasu_id (torch.Tensor): A tensor of shape (n_refln,) that contains the
                rasu ID of each reflection.
            H (torch.Tensor): A tensor of shape (n_refln, 3)

        Returns:
            gathered (torch.Tensor): A tensor of shape (n_refln, ...).
        """
        h, k, l = H.T
        idx = self.reflection_id_grid[rasu_id, h, k, l]
        gathered = source[idx]
        return gathered

    def __iter__(self) -> Iterator[ReciprocalASU]:
        return iter(self.reciprocal_asus)

    def __next__(self) -> ReciprocalASU:
        return next(self.reciprocal_asus)

    def __len__(self) -> int:
        return len(self.reciprocal_asus)


class ReciprocalASUGraph(ReciprocalASUCollection):
    def __init__(
        self,
        *rasus: ReciprocalASU,
        parents: Optional[torch.Tensor] = None,
        reindexing_ops: Optional[Sequence[str]] = None,
        **kwargs
    ) -> None:
        """
        A graph of rasu objects.

        Args:
            *rasus (ReciprocalASU): arbitrary number of rasu objects.
            parents (torch.Tensor, optional): A tensor of shape (n_rasus,) that contains the
                parent ID of each rasu.
            reindexing_ops (Sequence[str], optional): A sequence of strings that contain the
                reindexing operations for each rasu.

        Attributes:
            is_root (torch.Tensor): A boolean tensor of shape (rac_size,) that is True for
                reflections whose rasu has no parent, i.e. parent rasu is itself. Registered as a buffer.
            parent_reflection_ids (torch.Tensor): A tensor of shape (rac_size,) that contains
                the reflection ID of each reflection after reindexing in the parent rasu. Registered as a buffer.
            H_parent (torch.Tensor): A tensor of shape (rac_size, 3) that contains the
                Miller indices of reflections in all rasus after reindexing.
        """
        super().__init__(*rasus, **kwargs)
        if parents is None:
            parents = torch.arange(len(rasus))
        self.parents = parents
        self.reindexing_ops = reindexing_ops
        self.parent_rasu_ids = self.parents[self.rasu_ids]

        H_parent = []
        for i, rasu in enumerate(self.reciprocal_asus):
            op = "x,y,z"
            if reindexing_ops is not None:
                op = reindexing_ops[i]
            op = Op(op)
            H_parent.append(op(rasu.H_rasu))

        self.H_parent = torch.cat(H_parent, dim=0)
        self.register_buffer("is_root", self.parent_rasu_ids == self.rasu_ids)
        h, k, l = self.H_parent.T
        self.register_buffer(
            "parent_reflection_ids",
            self.reflection_id_grid[self.parent_rasu_ids, h, k, l],
        )
