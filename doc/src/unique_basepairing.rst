Unique base pairing describes the restriction on the specific complementary
nucleotide with which a particular base can pair. This can be used to prevent
asymmetric base pairs or to simplify the free energy landscape. With unique
base pairing enabled base pairs can only form between complementary nucleotides
with specific atom IDs. This functionality draws on :doc:`fix property/atom <fix_property_atom>`
and a modified :doc:`read_data <read_data>` command.

To use unique base pairing, the data file of a system with N nucleotides must contain a section like

.. code-block:: LAMMPS

   Basepairs # i_idc

   1 idc1
   2 idc2
   3 idc3
   4 idc4
   ...
   N idcN

where idc is the non-negative atom ID of a complementary nucleotide that binds uniquely
to the preceding atom ID.

Unique base pairing can be combined with normal base pairing by setting a zero or negative value for idc.
For instance, in a 4-mer with 8 nucleotides consisting of a ssDNA strand 3'-A-A-A-A-5' with atom IDs 3'-1-2-3-4-5'
and a complementary strand 5'-T-T-T-T-3' with atom IDs 5'-8-7-6-5-3' set up as

.. code-block:: LAMMPS

   Basepairs # i_idc

   1 8
   2 -1
   3 -1
   4 5
   5 4
   6 -1
   7 -1
   8 1

the A nucleotide with ID 1 can only hybridize with the T nucleotide with ID 8 and
the A nucleotide with ID 4 can only hybridize with the T nucleotide with ID 5,
whereas the A nucleotides with ID 2 and 3 can hybridize with either T nucleotide with ID 6 and 7.

The input file requires an instance of the :doc:`fix property/atom <fix_property_atom>` and a
:doc:`read_data <read_data>` command as follows:

.. code-block:: LAMMPS

   fix Basepairs all property/atom i_idc ghost yes
   read_data file fix Basepairs NULL Basepairs

where *file* is the name of the data file and the only modifiable argument.
An example input and data file for a dsDNA ring can be found in
``examples/PACKAGES/cgdna/examples/lj_units/oxDNA3/unique_bp``
or in the corresponding folder for real units.
