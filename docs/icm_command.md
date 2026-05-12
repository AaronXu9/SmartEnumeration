# ICM Terminal Command
COMMAND: a_Dock_5ZTY_rec_lig.m// 
ICM evaluates it and prints the full atom table — every atom with all its properties (coordinates, charge, MMFF code, etc.). Think of it as "show me everything about all atoms in this molecule."
Atom — the atom's name within the molecule (e.g. c1, n4, o3, h11). This is what you use in selections like a_.../1/c1.
Res — residue number. For small molecules there's only one residue, so this is always 1.
Mol — molecule name. Here it's always m.
Obj — object name. Here it's Dock_5ZTY_rec_lig.
X Y Z — 3D coordinates of the atom in Ångströms. These are the docked pose coordinates — do not touch these.
Occ — occupancy. Always 1.00 for a single conformation. Crystallography concept, not relevant here.
B — B-factor (temperature factor). The vt1/vt2 virtual atoms show 20.00, everything else is 0.00. Not relevant for our pipeline.
MMFF — MMFF94 atom type number. Defines how ICM treats this atom energetically (charges, van der Waals radius, etc.).
Code — ICM internal atom code. Similar role to MMFF type.
Xi — this is the isotope field. Currently showing 0 for all atoms — meaning the isotope labels are not present in this object. This is the problem.
Chrg — partial charge on the atom.
formal — formal charge (integer). Most atoms 0, some nitrogens show +1/3.
Grad — energy gradient. 0.0 means not minimized yet.
Area — solvent accessible surface area. 0.0 means not calculated yet.
Grp — group flag. _ means unassigned.
as_ — the full ICM address of the atom. This is the exact string you use to select it.

COMMAND: print Nof(a_Dock_5ZTY_rec_lig.m//)
Nof() = "Number of" — counts items in a selection
a_Dock_5ZTY_rec.m// = all atoms in molecule m
Result: 26 — this ligand has 26 atoms total

COMMAND: print Name(a_Dock_5ZTY_rec_lig.m//)
Name() = returns the name property of each atom in the selection
Result: prints all 26 atom names in order: vt1 vt2 h1 c1 c2...


as_graph is the special ICM variable that holds the graphical selection — exactly what gets set when you click an atom with the hand tool in the GUI. Assigning to it directly should replicate the click.

Selection syntax: a_Dock_5ZTY_rec_lig.m//1/r6: 
a_Dock_5ZTY_rec_lig.m// 
as_graph = a_Dock_5ZTY_rec_lig.m/1/r6