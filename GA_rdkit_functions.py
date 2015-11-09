from rdkit.Chem import AllChem as ac
from rdkit import Chem as chem
import numpy as np
import networkx as nx
import polygon as pg
import math as math
import rdkit as rdkit
""" replace one group in a molecule with another group and save
result in .mol file """ 



""" CODE """

substituted_heavy_atom_numbers = [39, 40, 41, 42, 43, 44, 45]

def ConvertMolToGraph(mol):
    """
    converts a molecule into a mathematical graph via a networkx object
    """
    G = nx.Graph()
    
    atom_numbers = [atom.GetIdx() for atom in mol.GetAtoms()]
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]

    G.add_nodes_from(atom_numbers)
    G.add_edges_from(bonds)
    
    return G

def OptimizeSmile(smile):
    """"
    takes the SMILES of a molecule and returns a 3D embedding of that molecule
    """
    #create a molecule object from SMILE and add Hs to it

    m = chem.MolFromSmiles(smile)
    m = chem.AddHs(m)
    
    #make molecule 3D and optimize under UFF force field
    
    ac.EmbedMolecule(m,useRandomCoords=True)
    #ac.UFFOptimizeMolecule(m)
    
    return m



def FragsToSubstitutionReactionSmarts(frag1_smiles, frag2_smiles):
    """
    creates a reaction SMARTS for two molecules. the reaction substitutes the
    core of fragment 1 with the core of fragment 2, keeping the functional
    group fragment 1 in place.
    """
    SubAt = 'K'

    frag1 = chem.MolFromSmiles(frag1_smiles)
    frag1 = chem.RemoveHs(frag1)
     
    
    for atom in frag1.GetAtoms():      
        if (atom.GetAtomicNum() in substituted_heavy_atom_numbers):          
            atom.SetAtomicNum(19)
    
    frag1 = chem.MolToSmiles(frag1)
      
    frag2 = chem.MolFromSmiles(frag2_smiles)
    frag2 = chem.RemoveHs(frag2)

    for atom in frag2.GetAtoms():        
        if (atom.GetAtomicNum() in substituted_heavy_atom_numbers):
            atom.SetAtomicNum(19)
    
    frag2 = chem.MolToSmiles(frag2)
    
    print(frag1)
    print(frag2)
    
    smart = ""        
    reaction_smart = ""     
    index_counter1 = 1    
    
    for x in frag1:        
        if x != SubAt:
            smart += x
        
        if x == SubAt:          
            smart += '*:' + str(index_counter1) + ''
            index_counter1 += 1
    
    reaction_smart += smart + "."   
    
    for x in frag2:        
        if x != SubAt:
            reaction_smart += x
            
        if x == SubAt:
            reaction_smart += '*'
            
    reaction_smart += ">>"
    smart = ""
    index_counter2 = 1
    
    for x in frag2:        
        if x != SubAt:
            smart += x
        
        if x == SubAt:          
            smart += '*:' + str(index_counter2) + ''
            index_counter2 += 1              
            
    if index_counter1 != index_counter2:        
        raise Exception('Fragments have different numbers of attachement points!')
        
    else:    
        reaction_smart += smart
    
    print('\n' + reaction_smart + '\n')
   
    return reaction_smart

def FragsToAdditionReactionSmarts(frag1_smiles, frag2_smiles):
    """
    takes two fragments and joins them to form one molecule - unused function
    """
    frag1 = chem.MolFromSmiles(frag1_smiles)
    frag1 = chem.RemoveHs(frag1)
    frag2 = chem.MolFromSmiles(frag2_smiles)
    frag2 = chem.RemoveHs(frag2)
    
    reaction_smarts = "("    
    reaction_smarts_inter = "("    
    
    for atom in frag1.GetAtoms():
        if (atom.GetAtomicNum() in substituted_heavy_atom_numbers):
            at1_smarts = atom.GetSmarts()            
            reaction_smarts += at1_smarts
            reaction_smarts_inter += at1_smarts            
            break

    reaction_smarts += "."        
    reaction_smarts_inter += ").("    
    
    for atom in frag2.GetAtoms():
        if (atom.GetAtomicNum() in substituted_heavy_atom_numbers):
            at2_smarts = atom.GetSmarts()            
            reaction_smarts += at2_smarts
            reaction_smarts_inter += at2_smarts            
            break
    
    reaction_smarts += ")>>" + at1_smarts + at2_smarts
    reaction_smarts_inter += ")>>" + at1_smarts + at2_smarts    
    
    at1_smarts_edit = at1_smarts.replace("]", ":1]")
    at2_smarts_edit = at2_smarts.replace("]", ":2]")
    reaction_smarts = reaction_smarts.replace(at1_smarts, at1_smarts_edit)
    reaction_smarts = reaction_smarts.replace(at2_smarts, at2_smarts_edit)
    reaction_smarts_inter = reaction_smarts_inter.replace(at1_smarts, at1_smarts_edit)
    reaction_smarts_inter = reaction_smarts_inter.replace(at2_smarts, at2_smarts_edit)
    
    return reaction_smarts, reaction_smarts_inter

def FragsSame(frag1, r2):
    """
    checks if two fragments are the same
    """
    
    if frag1.GetNumAtoms() == r2.GetNumAtoms() and frag1.HasSubstructMatch(r2) == True:
        return True
    else:
        return False

def Substitute(heavy_file_location, cur_frag_smiles, new_frag_smiles, file_name_with_dir):
    """    
    takes a cage and replaces a linker or building block with new new linker or building 
    block
    """
    ps_counter = 0

    reaction = FragsToSubstitutionReactionSmarts(cur_frag_smiles, new_frag_smiles)      
    r1 = chem.MolFromMolFile(heavy_file_location)
    r1 = chem.RemoveHs(r1)
    
    frag1 = chem.MolFromSmiles(cur_frag_smiles)     
    frag1 = chem.RemoveHs(frag1) 
    
    r2 = chem.MolFromSmiles(new_frag_smiles)
    r2 = chem.RemoveHs(r2)    
    
    for atom in frag1.GetAtoms():
        if (atom.GetAtomicNum() in substituted_heavy_atom_numbers):
            cur_sub_num = atom.GetAtomicNum()
    
    for atom in r2.GetAtoms():
        if (atom.GetAtomicNum() in substituted_heavy_atom_numbers):
            new_sub_num = atom.GetAtomicNum()
            
    for atom in r1.GetAtoms():
        if (atom.GetAtomicNum() == cur_sub_num):
            atom.SetAtomicNum(new_sub_num)
    
    for atom in frag1.GetAtoms():
        if (atom.GetAtomicNum() == cur_sub_num):
            atom.SetAtomicNum(new_sub_num)
            
    if frag1.GetNumAtoms() > r2.GetNumAtoms() and frag1.HasSubstructMatch(r2) == True:
        is_substruct = True
    
    else:
        is_substruct = False
             
    fragments_same = FragsSame(frag1, r2)        
    if fragments_same == True:        
        result_file = open(file_name_with_dir,'w')
        result_file.write(chem.MolToMolBlock(r1))
        result_file.close()
                        
        return file_name_with_dir
    
    rxn = ac.ReactionFromSmarts(reaction)           
    ps = rxn.RunReactants((r1, r2))
    ps_checker = len(ps)    
    
    if len(ps) == 0:
        print(len(ps))
        raise Exception("Substitution of fragments failed.")
        
    else:
         
        print('LEN PS   ' + str(len(ps)))         
         
        product = ps[0][0]
        
        while True:                        
            
            ps_counter += 1
                
            a = chem.MolToMolBlock(product)           
            r1 = chem.MolFromMolBlock(a)                   
              
            for match in r1.GetSubstructMatches(r2):               
                for atom in match:
                    r1.GetAtomWithIdx(atom).SetProp('_protected','1')                                      
                    
            if is_substruct == True:
                
                for match in r1.GetSubstructMatches(frag1):
                    
                    for atom in match:                       
                        r1.GetAtomWithIdx(atom).ClearProp('_protected')
                                                     
            ps = rxn.RunReactants((r1,r2))    
            len_ps = len(ps)
            
            if len_ps >= ps_checker:
                raise Exception("Substitution of fragments failed.")
            
            print(ps_counter)     
            print('len_ps ' + str(len_ps))           
            ps_checker = len_ps      
            if len_ps != 0:            
                product = ps[0][0]                

            else:
                result_file = open(file_name_with_dir,'w')
                ac.EmbedMolecule(product)                
                result_file.write(chem.MolToMolBlock(product))
                result_file.close()
                        
                return file_name_with_dir
                
def ChangeFunctionalGroupAtom(molecule_smiles, number_of_group_to_sub):
    """    
    takes the functional group of a molecule and substitutes it for a heavy atom
    """
    molecule = chem.MolFromSmiles(molecule_smiles)   
    molecule = chem.AddHs(molecule)

    functional_group_list = [
                             ("[N]([H])[H]","[Rh]", 1),  ("C(=O)[H]","[Y]", 2),
                             ("C(=O)O[H]","[Zr]", 3), ("C(=O)N([H])[H]","[Nb]", 4), 
                             ("C(=O)S[H]","[Mo]", 5), ("O[H]","[Tc]", 6), 
                             ("[S][H]","[Ru]", 7),("", None, 0)
                             ]

    for func_grp_data in functional_group_list:
        if func_grp_data[2] == number_of_group_to_sub:        
            functional_group = chem.MolFromSmarts(func_grp_data[0])
            replacement = chem.MolFromSmarts(func_grp_data[1])
            rms = ac.ReplaceSubstructs(molecule, functional_group, replacement, replaceAll = True)
            molecule = rms[0]
        
    new_mol_smiles = chem.MolToSmiles(molecule)    
    new_mol = chem.MolFromSmiles(new_mol_smiles)
    ac.EmbedMolecule(new_mol)
    ac.MMFFOptimizeMolecule(new_mol)
    new_mol = chem.MolToMolBlock(new_mol)
    return new_mol

def ReverseChangeFunctionalGroupAtom(molecule_smiles):
    molecule = chem.MolFromSmarts(molecule_smiles)
    #molecule = chem.AddHs(molecule)

    functional_group_list = [
                             ("[N][H][H]","[Rh]", 1),  ("C(=O)[H]","[Y]", 2),
                             ("C(=O)O","[Zr]", 3), ("C(=O)N","[Nb]", 4), 
                             ("C(=O)S","[Mo]", 5), ("O","[Tc]", 6), 
                             ("[S]","[Ru]", 7),("", None, 0)
                             ]
                             
    for func_grp_data in functional_group_list:
        functional_group = chem.MolFromSmarts(func_grp_data[1])       
        replacement = chem.MolFromSmarts(func_grp_data[0])
        if molecule.HasSubstructMatch(functional_group):
            rms = ac.ReplaceSubstructs(molecule, functional_group, replacement, replaceAll=True)          
            molecule = rms[0]
            break
        else:
            continue
        
    
    new_mol_smiles = chem.MolToSmiles(molecule)
    print(new_mol_smiles)
    new_mol = chem.MolFromSmarts(new_mol_smiles)

    for atom in new_mol.GetAtoms():
        atom.SetIsAromatic(False)        
        print(atom.GetIdx(), atom.GetSymbol(), len(atom.GetNeighbors()), atom.GetIsAromatic())     

        
    chem.RemoveHs(new_mol)
    chem.Kekulize(new_mol)
    chem.AddHs(new_mol)
    ac.EmbedMolecule(new_mol)
    ac.MMFFOptimizeMolecule(new_mol)
    new_mol = chem.MolToMolBlock(new_mol)
    return new_mol
    


class Atom(object):
    """
    the Atom class is used to find the distances between metal atoms in geometry
    file where the atoms are disconnected. this allows the connection of correct atoms
    """
    bb = None
    bb_heavy_atoms_per_molecule = None
    link = None    
    link_heavy_atoms_per_molecule = None
    linked_mols = []    
    
    def __init__(self, element, number, heavy_atom_num, x, y, z):
        self.element = element
        self.number = number
        self.heavy_atom_num = heavy_atom_num
        self.x = x
        self.y = y
        self.z = z
        self.min_partner = 10**9
        self.min_distance = 10**9
        self.distances = {}        
        self.paired = False
        
    def assign_molecule_number(self):       
        if self.element == Atom.bb:
            self.mol_number = math.ceil(self.heavy_atom_num / Atom.bb_heavy_atoms_per_molecule)  
        if self.element == Atom.link:
            self.mol_number = math.ceil(self.heavy_atom_num / Atom.link_heavy_atoms_per_molecule)
        
        return 1
    
    def distance(self, atom2):
        x_diff_sq = (self.x - atom2.x) ** 2
        y_diff_sq = (self.y - atom2.y) ** 2
        z_diff_sq = (self.z - atom2.z) ** 2
        r = np.sqrt(x_diff_sq + y_diff_sq + z_diff_sq)
        self.distances[atom2] = r
        atom2.distances[self] = r
        
        return r
        
    def pair_up(self, atom2):       
        if self.paired == True:
            return None
        
        if self.paired == False: 
            while True:               
                min_partner = min(self.distances, key=self.distances.get)
                
                if (min_partner.paired == False and (self.mol_number, min_partner.mol_number) not in Atom.linked_mols):                   
                    self.min_partner = min_partner
                    self.paired = True
                    min_partner.min_partner = self
                    min_partner.paired = True
                    Atom.linked_mols.append((self.mol_number, min_partner.mol_number))                    
                    return None
                
                else:                    
                    del self.distances[min_partner]
                    continue
  
def MolFileV2000ToV3000(mol_file_name):
    """
    converts a mol v2000 file to mol v3000
   
    """     
    mol = chem.MolFromMolFile(mol_file_name, removeHs = False)
    chem.MolToMolFile(mol, mol_file_name, forceV3000 = True)
    return mol_file_name

def AdditionSame(mol_file_name, bb, bb_heavy_atom_num_per_mol, output_file_name):
    """
    takens an input file with disconnected moleucles and connects them
    """    
    
    MolFileV2000ToV3000(mol_file_name)        
    
    mol_file = open(mol_file_name, "r")
    new_mol_file_content = ""
    
    atom_element = None    
    atom_list = []

    take_atom = False
    take_bond = False
    write_line = True

    type1_heavy_atom_number = 1

    for raw_line in mol_file:
       
        line = raw_line.split()
        
        if "M  V30 END BOND" in raw_line:
            write_line = False            
            
        if "M  V30 COUNTS" in raw_line:
            count_line = raw_line
            at_num = line[3]            
            
        if write_line == True:            
            new_mol_file_content += raw_line
                
        if "M  V30 BEGIN ATOM" in raw_line:
            take_atom = True
            continue
        
        if "M  V30 END ATOM" in raw_line:
            take_atom = False
            continue
        
        if "M  V30 BEGIN BOND" in raw_line:
            take_bond = True
            continue
        
        if take_atom == True:
    
            if (line[3] == "Y" or line[3] == "Zr" or line[3] == "Nb" or line[3] == "Mo" or line[3] == "Tc" or
                line[3] == "Ru" or line[3] == "Rh"):
                    
                    atom_element = line[3]
                    atom_list.append(Atom(line[3], line[2], type1_heavy_atom_number, float(line[4]), float(line[5]), float(line[6])))
                    type1_heavy_atom_number += 1                    
                    continue
                
        if take_bond == True and len(line) == 6:            
            bond_number = int(line[2])    
    
    Atom.bb_heavy_atoms_per_molecule = bb_heavy_atom_num_per_mol            
    Atom.bb = bb    
    Atom.linked_mols = [] 
    
    for atom in atom_list:
        atom.assign_molecule_number()

    for atom1 in atom_list:
        for atom2 in atom_list:
            atom1.distance(atom2)   
    
    for atom1 in atom_list:
        for atom2 in atom_list:
            atom1.pair_up(atom2)
    
    double_bond_combs = (("Rh","Y"), ("Nb","Y"), ("Mb","Rh"))    
    
    for atom1 in atom_list:
        bond_number += 1
        double_bond_present = [atom1.element in tup and atom1.min_partner.element in tup for tup in double_bond_combs]
        
        if True in double_bond_present:
            bond_order = "2"
        else:
            bond_order = "1"
            
        new_mol_file_content += "M  V30 {2} {3} {0} {1}\n".format(atom1.number, atom1.min_partner.number, 
                                                                  bond_number, bond_order)
    
    
    new_mol_file_content += "M  V30 END BOND\nM  V30 END CTAB\nM  END"
    mol_file.close()

    new_mol_file_content = new_mol_file_content.replace(" VAL=1", "")
    new_mol_file_content = new_mol_file_content.replace(count_line, "M  V30 COUNTS {0} {1} 0 0 0\n".format(at_num,bond_number))
    new_mol_file_name = output_file_name
    new_mol_file = open(new_mol_file_name, "w")
    new_mol_file.write(new_mol_file_content)
    new_mol_file.close()
    return 1



def AdditionDifferent(mol_file_name, bb, bb_heavy_atom_num_per_mol, link, link_heavy_atom_num_per_mol, output_file_name):
    """
    takens an input file with disconnected moleucles and connects them
    """    
    
    MolFileV2000ToV3000(mol_file_name)        
    
    mol_file = open(mol_file_name, "r")
    new_mol_file_content = ""
    
    atom1_element = None    
    atom2_element = None
    atom1_list = []
    atom2_list = []

    take_atom = False
    take_bond = False
    write_line = True

    type1_heavy_atom_number = 1
    type2_heavy_atom_number = 1

    for raw_line in mol_file:
       
        line = raw_line.split()
        
        if "M  V30 END BOND" in raw_line:
            write_line = False            
            
        if "M  V30 COUNTS" in raw_line:
            count_line = raw_line
            at_num = line[3]            
            
        if write_line == True:            
            new_mol_file_content += raw_line
                
        if "M  V30 BEGIN ATOM" in raw_line:
            take_atom = True
            continue
        
        if "M  V30 END ATOM" in raw_line:
            take_atom = False
            continue
        
        if "M  V30 BEGIN BOND" in raw_line:
            take_bond = True
            continue
        
        if take_atom == True:
    
            if (line[3] == "Y" or line[3] == "Zr" or line[3] == "Nb" or line[3] == "Mo" or line[3] == "Tc" or
                line[3] == "Ru" or line[3] == "Rh") and (atom1_element == None) and (atom2_element == None):
                    
                    atom1_element = line[3]
                    atom1_list.append(Atom(line[3], line[2], type1_heavy_atom_number, float(line[4]), float(line[5]), float(line[6])))
                    type1_heavy_atom_number += 1                    
                    continue
                
            if ((line[3] == "Y" or line[3] == "Zr" or line[3] == "Nb" or line[3] == "Mo" or line[3] == "Tc" or
                line[3] == "Ru" or line[3] == "Rh") and (atom1_element != None) and (atom2_element == None) and 
                (line[3] != atom1_element)):
                    
                    atom2_element = line[3]
                    atom2_list.append(Atom(line[3], line[2], type2_heavy_atom_number, float(line[4]), float(line[5]), float(line[6])))
                    type2_heavy_atom_number += 1                        
                    continue
                
            if (line[3] == "Y" or line[3] == "Zr" or line[3] == "Nb" or line[3] == "Mo" or line[3] == "Tc" or
                line[3] == "Ru" or line[3] == "Rh") and (atom1_element != None) and (atom1_element == line[3]):
                    
                    atom1_list.append(Atom(line[3], line[2], type1_heavy_atom_number, float(line[4]), float(line[5]), float(line[6])))
                    type1_heavy_atom_number += 1
                    continue
                    
            if (line[3] == "Y" or line[3] == "Zr" or line[3] == "Nb" or line[3] == "Mo" or line[3] == "Tc" or
                line[3] == "Ru" or line[3] == "Rh") and (atom2_element != None) and (atom2_element == line[3]):
                    
                    atom2_list.append(Atom(line[3], line[2], type2_heavy_atom_number, float(line[4]), float(line[5]), float(line[6])))
                    type2_heavy_atom_number += 1
                    continue  

        if take_bond == True and len(line) == 6:            
            bond_number = int(line[2])    
    
    Atom.bb_heavy_atoms_per_molecule = bb_heavy_atom_num_per_mol    
    Atom.link_heavy_atoms_per_molecule = link_heavy_atom_num_per_mol    
    
    Atom.bb = bb
    Atom.link = link
    
    Atom.linked_mols = [] 
    
    for atom in atom1_list:
        atom.assign_molecule_number()
    
    for atom in atom2_list:
        atom.assign_molecule_number()
    
    for atom1 in atom1_list:
        for atom2 in atom2_list:
            atom1.distance(atom2)   
    
    for atom1 in atom1_list:
        for atom2 in atom2_list:
            atom1.pair_up(atom2)
    
    double_bond_combs = (("Rh","Y"), ("Nb","Y"), ("Mb","Rh"))    
    
    for atom1 in atom1_list:
        bond_number += 1
        double_bond_present = [atom1.element in tup and atom1.min_partner.element in tup for tup in double_bond_combs]
        
        if True in double_bond_present:
            bond_order = "2"
        else:
            bond_order = "1"
            
        new_mol_file_content += "M  V30 {2} {3} {0} {1}\n".format(atom1.number, atom1.min_partner.number, 
                                                                  bond_number, bond_order)
    
    
    new_mol_file_content += "M  V30 END BOND\nM  V30 END CTAB\nM  END"
    mol_file.close()

    new_mol_file_content = new_mol_file_content.replace(" VAL=1", "")
    new_mol_file_content = new_mol_file_content.replace(count_line, "M  V30 COUNTS {0} {1} 0 0 0\n".format(at_num,bond_number))
    new_mol_file_name = output_file_name
    new_mol_file = open(new_mol_file_name, "w")
    new_mol_file.write(new_mol_file_content)
    new_mol_file.close()
    return 1

def GetAtomGroupCoM(molecule, conformer, atom_list):
    """
    gets the centre of mass of a group of atoms
    """
    
    mass_xposition_product_total = 0.0
    mass_yposition_product_total = 0.0
    mass_zposition_product_total = 0.0    
    mass_total = 0.0    
    
    for atom_id in atom_list:    
    
        atom_position = (conformer.GetAtomPosition(atom_id).x, conformer.GetAtomPosition(atom_id).y, 
                         conformer.GetAtomPosition(atom_id).z)
    
        atom_mass =  molecule.GetAtomWithIdx(atom_id).GetMass()
        
        mass_xposition_product_total += atom_mass * atom_position[0]        
        mass_yposition_product_total += atom_mass * atom_position[1]
        mass_zposition_product_total += atom_mass * atom_position[2]

        mass_total += atom_mass        
        
    x_coord = mass_xposition_product_total / mass_total
    y_coord = mass_yposition_product_total / mass_total
    z_coord = mass_zposition_product_total / mass_total
    
    return x_coord, y_coord,z_coord


def GetWindowSize(mol_file_name):
    """
    gets the window size of a cage
    """        
    
    mol = chem.MolFromMolFile(mol_file_name, removeHs = False)
    mol = chem.RemoveHs(mol)
    
    #mol_graph = ConvertMolToGraph(mol)    
    #mol_ring_object = nx.cycle_basis(mol_graph)
    
    #chem.GetSSSR(mol)
    mol_ring_object = mol.GetRingInfo().AtomRings()       
    
    #mol_ring_object = chem.GetSymmSSSR(mol)    
    
    ring_lengths = [len(x) for x in mol_ring_object]   
    max_ring_length = max(ring_lengths)
    
    conformer = mol.GetConformer(0)

    max_window_lengths = []
    min_window_lengths = []
    short_coords = []
    short_atom_numbers = []
    short_atom_types = []
    long_coords = []
    long_atom_numbers = []
    long_atom_types = []        
    
    for ring in mol_ring_object:
        if len(ring) == max_ring_length:            
            
            centre_of_mass = GetAtomGroupCoM(mol, conformer, ring)           
            
            shortest_distance = 999.9
            longest_distance = 0.0
            
            for atom_id in ring:
                
                cur_atom_x = conformer.GetAtomPosition(atom_id).x
                cur_atom_y = conformer.GetAtomPosition(atom_id).y
                cur_atom_z = conformer.GetAtomPosition(atom_id).z
                
                x_dist_sq = (cur_atom_x - centre_of_mass[0]) ** 2
                y_dist_sq = (cur_atom_y - centre_of_mass[1]) ** 2
                z_dist_sq = (cur_atom_z - centre_of_mass[2]) ** 2
                
                dist = np.sqrt(x_dist_sq + y_dist_sq + z_dist_sq)

                if dist < shortest_distance:                  
                    shortest_distance = dist
                    short_x = cur_atom_x
                    short_y = cur_atom_y
                    short_z = cur_atom_z
                    short_at_type = mol.GetAtomWithIdx(atom_id).GetAtomicNum()
                    
                if dist > longest_distance:
                    longest_distance = dist
                    long_x = cur_atom_x
                    long_y = cur_atom_y
                    long_z = cur_atom_z
                    long_at_type = mol.GetAtomWithIdx(atom_id).GetAtomicNum()
            
            max_window_lengths.append((longest_distance - 1.7)*2)            
            min_window_lengths.append((shortest_distance - 1.7)*2)
            
            mol_file = open(mol_file_name, "r")
            
            if short_x == -0.0 or short_x == 0.0:
                short_x = 0
                
            if short_y == -0.0 or short_y == 0.0:
                short_y = 0

            if short_z == -0.0 or short_z == 0.0:
                short_z = 0                
            
            for line in mol_file:
                if str(short_x) in line and str(short_y) in line and str(short_z) in line:
                    line = line.split()
                    short_at = line[2]
                if str(long_x) in line and str(long_y) in line and str(long_z) in line:
                    line = line.split()
                    long_at = line[2]
                    
            mol_file.close()
            
            short_coords.append((short_x, short_y, short_z))
            short_atom_numbers.append(short_at)
            short_atom_types.append(short_at_type)
            long_coords.append((long_x, long_y, long_z))
            long_atom_numbers.append(long_at)
            long_atom_types.append(long_at_type)

    return (max_window_lengths, min_window_lengths, short_coords, 
           short_atom_numbers, short_atom_types, long_coords, long_atom_numbers,
           long_atom_types)

def GetCavitySize(mol_file_name):
    """
    gets the cavity size of a cage
    """
    
    mol = chem.MolFromMolFile(mol_file_name)
    try:
        conformer = mol.GetConformer(0)
    except:
        return 0
    
    atom_list = [x.GetIdx() for x in mol.GetAtoms()]
    centre_of_mass = GetAtomGroupCoM(mol, conformer, atom_list)
    shortest_distance = 999.9
    
    for atom_id in atom_list:
        
        cur_atom_x = conformer.GetAtomPosition(atom_id).x
        cur_atom_y = conformer.GetAtomPosition(atom_id).y
        cur_atom_z = conformer.GetAtomPosition(atom_id).z

        x_dist_sq = (cur_atom_x - centre_of_mass[0]) ** 2
        y_dist_sq = (cur_atom_y - centre_of_mass[1]) ** 2
        z_dist_sq = (cur_atom_z - centre_of_mass[2]) ** 2
        
#        print("\n\n\n****************************************************\n\n\n")
#        print("\n\n\n****************************************************\n\n\n")
#        print("\n\n\n****************************************************\n\n\n")
#        print("\n\n  THE ELEMENT IS: ", mol.GetAtomWithIdx(atom_id).GetSymbol(), "\n\n")
#        print("\n\n\n****************************************************\n\n\n")
#        print("\n\n\n****************************************************\n\n\n")
#        print("\n\n\n****************************************************\n\n\n")
        
        element = str(mol.GetAtomWithIdx(atom_id).GetSymbol())
        
#        print("############################## ", element, "   ", pg.atom_vdw_radii[element], " #############################")
        #Calculates the cage's radius        
        dist = float(np.sqrt(x_dist_sq + y_dist_sq + z_dist_sq)) - pg.atom_vdw_radii[element]
#        print("############################## ", dist, " #############################")
        
        if dist < shortest_distance:                  
            shortest_distance = dist
    
    ##Returns the shortest diameter (radius * 2)
    return (shortest_distance * 2)
        
def EnforceChirality(mol_file_location):
    
    mol = chem.MolFromMolFile(mol_file_location, removeHs=False)
    conformer = mol.GetConformer(0)
    
    chem.AssignAtomChiralTagsFromStructure(mol)
    atoms_to_not_change = []
    for atom in mol.GetAtoms():  
        if atom.GetIdx() in atoms_to_not_change:
            #atom.SetAtomicNum(39)            
            continue
        for neighbor in atom.GetNeighbors():
            if (atom.GetChiralTag() == neighbor.GetChiralTag() and 
                atom.GetChiralTag() in [rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW] and 
                neighbor.GetChiralTag() in [rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW]):
                                
                for neighbor2 in atom.GetNeighbors():
                    if neighbor2.GetAtomicNum() == 1:
                        atom_id = atom.GetIdx()                        
                        neighbor2_id = neighbor2.GetIdx()
                        atom_coords = conformer.GetAtomPosition(atom_id)                        
                        neighbor2_coords = conformer.GetAtomPosition(neighbor2_id)
                        x_vec = neighbor2_coords.x - atom_coords.x                            
                        y_vec = neighbor2_coords.y - atom_coords.y
                        z_vec = neighbor2_coords.z - atom_coords.z
                        neighbor2_coords.x = 0.5*x_vec
                        neighbor2_coords.y = 0.5*y_vec
                        neighbor2_coords.z = 0.5*z_vec                         
                        conformer.SetAtomPosition(neighbor2_id, neighbor2_coords)
                        #neighbor2.SetAtomicNum(39)                        
                        atoms_to_not_change.extend([x.GetIdx() for x in atom.GetNeighbors()])                                               
                        break
                break
            break
    
    chem.MolToMolFile(mol, "testout.mol", confId=0)
                
    
    return 1
"""
import os as os
os.chdir("/home/lukas")
EnforceChirality("test1out.mol")
""" 
    
    
    
    
    

     