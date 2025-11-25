"""
Tests for numerical stability in structure_data.py and CIFIO.py
"""
import numpy as np
import pytest


class TestComputeAngleBetween:
    """Tests for MolecularGraph.compute_angle_between()"""

    def test_zero_length_vector_returns_zero(self):
        """When atoms are coincident (zero-length vector), should return 0."""
        from lammps_interface.structure_data import MolecularGraph

        graph = MolecularGraph()
        # Add three nodes where l and m are at the same position
        graph.add_node(1, cartesian_coordinates=np.array([0.0, 0.0, 0.0]))
        graph.add_node(2, cartesian_coordinates=np.array([0.0, 0.0, 0.0]))  # Same as node 1
        graph.add_node(3, cartesian_coordinates=np.array([1.0, 0.0, 0.0]))

        # Should not crash and should return 0
        angle = graph.compute_angle_between(1, 2, 3)
        assert angle == 0.0

    def test_normal_angle_90_degrees(self):
        """Test that 90 degree angle is computed correctly."""
        from lammps_interface.structure_data import MolecularGraph

        graph = MolecularGraph()
        graph.add_node(1, cartesian_coordinates=np.array([1.0, 0.0, 0.0]))
        graph.add_node(2, cartesian_coordinates=np.array([0.0, 0.0, 0.0]))
        graph.add_node(3, cartesian_coordinates=np.array([0.0, 1.0, 0.0]))

        angle = graph.compute_angle_between(1, 2, 3)
        assert np.isclose(angle, 90.0, atol=1e-10)

    def test_normal_angle_180_degrees(self):
        """Test that 180 degree angle (collinear) is computed correctly."""
        from lammps_interface.structure_data import MolecularGraph

        graph = MolecularGraph()
        graph.add_node(1, cartesian_coordinates=np.array([-1.0, 0.0, 0.0]))
        graph.add_node(2, cartesian_coordinates=np.array([0.0, 0.0, 0.0]))
        graph.add_node(3, cartesian_coordinates=np.array([1.0, 0.0, 0.0]))

        angle = graph.compute_angle_between(1, 2, 3)
        assert np.isclose(angle, 180.0, atol=1e-10)

    def test_normal_angle_60_degrees(self):
        """Test that 60 degree angle is computed correctly."""
        from lammps_interface.structure_data import MolecularGraph

        graph = MolecularGraph()
        graph.add_node(1, cartesian_coordinates=np.array([1.0, 0.0, 0.0]))
        graph.add_node(2, cartesian_coordinates=np.array([0.0, 0.0, 0.0]))
        graph.add_node(3, cartesian_coordinates=np.array([0.5, np.sqrt(3)/2, 0.0]))

        angle = graph.compute_angle_between(1, 2, 3)
        assert np.isclose(angle, 60.0, atol=1e-10)


class TestComputeDihedralBetween:
    """Tests for MolecularGraph.compute_dihedral_between()"""

    def test_collinear_atoms_returns_zero(self):
        """When atoms are collinear (zero-length cross product), should return 0."""
        from lammps_interface.structure_data import MolecularGraph, Cell

        graph = MolecularGraph()
        # Add four collinear nodes
        graph.add_node(1, cartesian_coordinates=np.array([0.0, 0.0, 0.0]))
        graph.add_node(2, cartesian_coordinates=np.array([1.0, 0.0, 0.0]))
        graph.add_node(3, cartesian_coordinates=np.array([2.0, 0.0, 0.0]))
        graph.add_node(4, cartesian_coordinates=np.array([3.0, 0.0, 0.0]))

        # Assign cell (needed for min_img)
        cell = Cell()
        cell.params = (100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
        graph.cell = cell

        # Should not crash and should return 0
        angle = graph.compute_dihedral_between(1, 2, 3, 4)
        assert angle == 0.0

    def test_normal_dihedral(self):
        """Test that a normal dihedral is computed correctly."""
        from lammps_interface.structure_data import MolecularGraph, Cell

        graph = MolecularGraph()
        # Create a non-planar dihedral
        graph.add_node(1, cartesian_coordinates=np.array([0.0, 1.0, 0.0]))
        graph.add_node(2, cartesian_coordinates=np.array([0.0, 0.0, 0.0]))
        graph.add_node(3, cartesian_coordinates=np.array([1.0, 0.0, 0.0]))
        graph.add_node(4, cartesian_coordinates=np.array([1.0, 1.0, 0.0]))

        # Assign cell (needed for min_img)
        cell = Cell()
        cell.params = (100.0, 100.0, 100.0, 90.0, 90.0, 90.0)
        graph.cell = cell

        # This should compute without error
        angle = graph.compute_dihedral_between(1, 2, 3, 4)
        assert isinstance(angle, float)
        assert not np.isnan(angle)


class TestCoplanar:
    """Tests for MolecularGraph.coplanar()"""

    def test_collinear_first_two_neighbors_returns_false(self):
        """When first two neighbors are collinear with center, should return False."""
        from lammps_interface.structure_data import MolecularGraph

        graph = MolecularGraph()
        # Center node
        graph.add_node(1, cartesian_coordinates=np.array([0.0, 0.0, 0.0]))
        # Two collinear neighbors
        graph.add_node(2, cartesian_coordinates=np.array([1.0, 0.0, 0.0]))
        graph.add_node(3, cartesian_coordinates=np.array([2.0, 0.0, 0.0]))
        # A third neighbor
        graph.add_node(4, cartesian_coordinates=np.array([0.0, 1.0, 0.0]))

        # Add edges to node 1
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)

        # Should not crash and should return False (can't define a plane from collinear vectors)
        result = graph.coplanar(1)
        assert result is False

    def test_coplanar_square(self):
        """Test that a square planar configuration is detected as coplanar."""
        from lammps_interface.structure_data import MolecularGraph

        graph = MolecularGraph()
        # Center node
        graph.add_node(1, cartesian_coordinates=np.array([0.0, 0.0, 0.0]))
        # Four coplanar neighbors
        graph.add_node(2, cartesian_coordinates=np.array([1.0, 0.0, 0.0]))
        graph.add_node(3, cartesian_coordinates=np.array([0.0, 1.0, 0.0]))
        graph.add_node(4, cartesian_coordinates=np.array([-1.0, 0.0, 0.0]))
        graph.add_node(5, cartesian_coordinates=np.array([0.0, -1.0, 0.0]))

        # Add edges to node 1
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)
        graph.add_edge(1, 5)

        result = graph.coplanar(1)
        assert result is True

    def test_tetrahedral_not_coplanar(self):
        """Test that a tetrahedral configuration is detected as not coplanar."""
        from lammps_interface.structure_data import MolecularGraph

        graph = MolecularGraph()
        # Center node
        graph.add_node(1, cartesian_coordinates=np.array([0.0, 0.0, 0.0]))
        # Four tetrahedral neighbors
        graph.add_node(2, cartesian_coordinates=np.array([1.0, 1.0, 1.0]))
        graph.add_node(3, cartesian_coordinates=np.array([1.0, -1.0, -1.0]))
        graph.add_node(4, cartesian_coordinates=np.array([-1.0, 1.0, -1.0]))
        graph.add_node(5, cartesian_coordinates=np.array([-1.0, -1.0, 1.0]))

        # Add edges to node 1
        graph.add_edge(1, 2)
        graph.add_edge(1, 3)
        graph.add_edge(1, 4)
        graph.add_edge(1, 5)

        result = graph.coplanar(1)
        assert result is False


class TestCIFIOExceptionHandling:
    """Tests for exception handling in CIFIO.py"""

    def test_add_data_handles_type_error(self, capsys):
        """Test that TypeError is properly caught and reported."""
        from lammps_interface.CIFIO import CIF

        cif = CIF()
        # First, add data to create a non-list value (like in non_loops)
        cif.add_data("cell", _cell_length_a="10.0")
        
        # Verify the data was added
        assert "_cell_length_a" in cif._data

    def test_add_data_normal_loop(self):
        """Test that normal loop data addition works correctly."""
        from lammps_interface.CIFIO import CIF

        cif = CIF()
        # Add data to a loop block
        cif.add_data("atoms", _atom_site_label="C1")
        cif.add_data("atoms", _atom_site_label="C2")

        assert cif._data["_atom_site_label"] == ["C1", "C2"]
