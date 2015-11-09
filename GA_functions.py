import GA_rdkit_functions as grf
import polygon as pg
import operator as op
import random as random
from rdkit import Chem as chem
import os as os
import time as time
import subprocess as sp
from rdkit.Chem import AllChem as ac
import numpy as np
import copy as cp
import itertools
from sklearn.cluster import DBSCAN
from sklearn import metrics
import multiprocessing as mp

class Cage(object):
    
    """ Class that defines a cage and all its attributes. It initiates with a series of 
    different values, which are then defined as attributes. """
    
    def __init__(self, bb_smiles_prist, bb_smiles, link_smiles_prist, link_smiles, 
                 mol_file_location, heavy_mol_file_location, parent1, parent2, chosen_shape): #
        self.bb_smiles_prist = bb_smiles_prist        
        self.bb_smiles = bb_smiles
        self.link_smiles_prist = link_smiles_prist        
        self.link_smiles = link_smiles   
        self.mol_file_location = mol_file_location
        self.heavy_mol_file_location = heavy_mol_file_location
        self.parent1 = parent1
        self.parent2 = parent2
        self.shape = chosen_shape
        
        ######*********** To be expanded as more topologies are implemented ************####
        ## Defines an attribute for the number of windows depending on the topology.
        if chosen_shape == "Trigonal Bi":
            self.window_num = 3
        elif chosen_shape == "Tetrahedron":
            self.window_num = 4
        elif chosen_shape == "Cube":
            self.window_num = 6
        elif chosen_shape == "Octahedron":
            self.window_num = 8
        elif chosen_shape == "Dodecahedron":
            self.window_num = 12
        ######*********** To be expanded as more topologies are implemented ************####


class MOL:
    """
    TAKEN FROM MARCIN's code 20/10/2015
    This class loads and opens a MOL(V3000) file.
    self.path: Path of the file
    self.name: Name of the file
    self.body: list of string as in loaded file
    self.no_of_atoms: total number of atoms
    self.comment: Second (comment) line in XYZ file
    self.atom_list: Creates list of kind [[id1, no1, x1, y1, z1], ...,[idn, non, xn, yn, zn]]
    self.array: It also creates a numpy array containing only atom coordinates [[x1,y1,z1], ..., [xn,yn,zn]]
    """
    def __init__(self, mol):
        self.path = mol
        self.name = mol.split('/')[-1][:-4]
        with open(mol, 'r') as mol_source:
            self.body = [i.split() for i in mol_source.readlines()]
            #self.no_of_atoms = int(self.body[0][0])
            #self.comment = self.body[1][0]
        iteration = 1
        atom_list = []
        array = []
        flag = False
        for i in self.body:
            if len(i) >= 4:
                if i[2] == 'END' and i[3] == 'ATOM':
                    flag = False
                if flag == True:
                    atom_list.append([i[3], iteration, float(i[4]), float(i[5]), float(i[6])])
                    array.append([float(i[4]), float(i[5]), float(i[6])])
                    iteration += 1
                if i[2] == 'BEGIN' and i[3] == 'ATOM':
                    flag = True
        self.atom_list = atom_list
        self.array = np.array(array)
        
    def com2zero(self):
        
        """
        This function first calculates the center of mass of the molecule and than translates all the 
        coordinates, so that new center of mass ends up in the origin.
        """
        
        com = center_of_mass(self.atom_list)
        self.atom_list = [[i[0],i[1],i[2]-com[0],i[3]-com[1],i[4]-com[2]] for i in self.atom_list]
        self.array = np.array([[i[0]-com[0],i[1]-com[1],i[2]-com[2]] for i in self.array])
        
        
######################################################################################################
######################################################################################################
#**************** Functions for the GENETIC OPERATIONS (Mating and Mutation)  ***********************#
######################################################################################################
######################################################################################################


def CageMating(individual1, individual2, bb_grp_sub_num, link_grp_sub_num, output_file_name_with_dir):
    
    """ This function applies the mating operation of two different parental cages (cage1 and cage2).
    It requires as parameters (1) cage1, (2) cage2, (3) the number of functional groups to be substituted
    for the building block, (4) for the linker and (5) the absolute path of the output file for the 
    offspring. It returns the two offspring structures."""
       
    cage1_bb = individual1.bb_smiles #Info about parent number 1
    cage1_link = individual1.link_smiles
    cage1_shape = individual1.shape
    
    cage2_bb = individual2.bb_smiles #Info about parent number 2
    cage2_link = individual2.link_smiles
    cage2_shape = individual2.shape

    offspring1_file_name_with_dir = output_file_name_with_dir.replace(".mol", "_offspring1.mol")
    offspring1_heavy_file_name_with_dir = offspring1_file_name_with_dir.replace(".mol", "_HEAVY_ATOMS.mol")
    offspring2_file_name_with_dir = output_file_name_with_dir.replace(".mol", "_offspring2.mol")
    offspring2_heavy_file_name_with_dir = offspring2_file_name_with_dir.replace(".mol", "_HEAVY_ATOMS.mol")  

    ## This line is the main body of the function, where offspring 1 is assembled by crossing the bb and linkers of 
    ## cage1 and cage2. It calls the BuildCage function which is defined in the polygon module.

    offspring1_heavy_file_name_with_dir, offspring1_bb_n, bb_sub_atoms, offspring1_lk_n, lk_sub_atoms = pg.BuildCage(cage1_shape,
                                                                                                                     offspring1_file_name_with_dir, cage2_bb, bb_grp_sub_num, cage1_link, link_grp_sub_num)
     
    ## Heavy atoms are substituted with the final atoms in the offspring1 structure.
    pg.final_sub(offspring1_heavy_file_name_with_dir, offspring1_file_name_with_dir)
    
    offspring1_bb_prist = individual2.bb_smiles_prist    #Definition of attributes regarding offsrping1
    offspring1_bb = individual2.bb_smiles
    offspring1_lk_prist = individual1.link_smiles_prist    
    offspring1_link = individual1.link_smiles
    
    ## Cage class is called, so that all the attributes about offspring 1 are initialised.
    
    offspring1 = Cage(offspring1_bb_prist, offspring1_bb, offspring1_lk_prist, offspring1_link, offspring1_file_name_with_dir,
                  offspring1_file_name_with_dir, individual1.index, individual2.index, cage1_shape)

    offspring1.origin = "mating"
    
    ## This line is the main body of the function, where offspring 1 is assembled by crossing the bb and linkers of 
    ## cage1 and cage2. It calls the BuildCage function which is defined in the polygon module.
    
    offspring2_heavy_file_name_with_dir, offspring2_bb_n, bb_sub_atoms, offspring2_lk_n, lk_sub_atoms = pg.BuildCage(cage2_shape, 
                                                                                                                     offspring2_file_name_with_dir, cage1_bb, bb_grp_sub_num, cage2_link, link_grp_sub_num)
    
    ## Heavy atoms are substituted with the final atoms in the offspring2 structure.
    pg.final_sub(offspring2_heavy_file_name_with_dir, offspring2_file_name_with_dir)
    
    offspring2_bb_prist = individual1.bb_smiles_prist   #Definition of attributes regarding offsrping1
    offspring2_bb = individual1.bb_smiles
    offspring2_lk_prist = individual2.link_smiles_prist    
    offspring2_link = individual2.link_smiles
    
    ## Cage class is called, so that all the attributes about offspring 2 are initialised.
    
    offspring2 = Cage(offspring2_bb_prist, offspring2_bb, offspring2_lk_prist, offspring2_link, offspring2_file_name_with_dir, 
                          offspring2_heavy_file_name_with_dir, individual1.index, individual2.index, cage2_shape)
                          
    offspring2.origin = "mating"
        
    return offspring1, offspring2


def FragmentMutation(individual, output_file_location, source_folder, grp_sub_num, bb_folder):
    
#    print("\n\n\n TRYING FRAGMENT MUTATION \n\n\n")
    
    mutation_file = random.choice(os.listdir(source_folder))
    mutation_location = source_folder + mutation_file
    new_frag_mol = chem.MolFromMolFile(mutation_location)   
    new_frag_smiles = chem.MolToSmiles(new_frag_mol)    
    heavy_mutant_location = output_file_location.replace(".mol", "_HEAVY_ATOMS.mol")    
    
    if ("+" in new_frag_smiles or "-" in new_frag_smiles or "." in new_frag_smiles):
           raise Exception("Invalid SMILES selected from Database during Mutation")
    
    new_frag_smiles_prist = new_frag_smiles                                  
    new_frag_mol_block = grf.ChangeFunctionalGroupAtom(new_frag_smiles, 
                                                       grp_sub_num)
    new_frag_mol = chem.MolFromMolBlock(new_frag_mol_block)
    new_frag_smiles = chem.MolToSmiles(new_frag_mol)
    
    if source_folder == bb_folder:          
        grf.Substitute(individual.heavy_mol_file_location, individual.bb_smiles, 
                       new_frag_smiles, heavy_mutant_location)
        
        mutant = Cage(new_frag_smiles_prist, new_frag_smiles, individual.link_smiles_prist, 
                      individual.link_smiles, output_file_location, heavy_mutant_location, individual.index, "mutant", individual.shape)
    
    else:       
        grf.Substitute(individual.heavy_mol_file_location, individual.link_smiles, 
                       new_frag_smiles, heavy_mutant_location)  
        
        mutant = Cage(individual.bb_smiles_prist, individual.bb_smiles, new_frag_smiles_prist, 
                      new_frag_smiles, output_file_location, heavy_mutant_location, individual.index, "mutant", individual.shape)                       
               
    pg.final_sub(heavy_mutant_location, output_file_location)

    mutant.origin = "mutation"
    
    return mutant,
    

def TopologyMutation(individual, output_file_location, bb_grp_sub_num, link_grp_sub_num):
    
    """
    Mutation that randomly changes the topology of a cage, keeping the same bb and lk.
    """
    
#    print("\n\n\n TRYING TOPOLOGY MUTATION \n\n\n")
    
    heavy_mutant_location = output_file_location.replace(".mol", "_HEAVY_ATOMS.mol")
    
    list_of_shapes = ["Tetrahedron", "Trigonal Bi", "Cube", "Dodecahedron"]
    chosen_shape = random.choice(list_of_shapes)
    while individual.shape == chosen_shape:
        chosen_shape = random.choice(list_of_shapes)
    
    heavy_mutant_location, bb_n, bb_sub_atoms, lk_n, lk_sub_atoms = pg.BuildCage(chosen_shape, output_file_location, 
                                                                                 individual.bb_smiles, bb_grp_sub_num, individual.link_smiles, link_grp_sub_num)                                                                                        
                                                                                        
   
    mutant = Cage(individual.bb_smiles_prist, individual.bb_smiles, individual.link_smiles_prist, individual.link_smiles, 
                  output_file_location, heavy_mutant_location, individual.index, "mutant", chosen_shape)
    
    pg.final_sub(heavy_mutant_location, output_file_location)
                  
    mutant.origin = "mutation"
    
    return mutant,

    
def TieMutation(individual, output_file_location):
    
#    print("\n\n\n ************** TRYING TIE MUTATION ****************\n\n\n")
    
    
    link_mol = chem.MolFromSmiles(individual.link_smiles)    
    for atom in link_mol.GetAtoms():
        if atom.GetAtomicNum() >= 39:
            atom_smiles = atom.GetSymbol()
            break

    reaction_smarts = "([" + atom_smiles + ":1].[" + atom_smiles + ":2])>>[" + atom_smiles + ":1]C[" + atom_smiles + ":2]"
    rxn = ac.ReactionFromSmarts(reaction_smarts)
    ps = rxn.RunReactants((link_mol,))    
    new_link_smiles = chem.MolToSmiles(ps[0][0])
    a = chem.MolFromSmiles(individual.link_smiles)
    b = chem.MolFromSmiles(new_link_smiles)
    ac.EmbedMolecule(a)
    ac.EmbedMolecule(b)    
    chem.MolToMolFile(a , "link.mol")
    chem.MolToMolFile(b, "new.mol")
    heavy_mutant_location = output_file_location.replace(".mol", "_HEAVY_ATOMS.mol")    
    grf.Substitute(individual.heavy_mol_file_location, individual.link_smiles, 
                   new_link_smiles, heavy_mutant_location)
    
    pg.final_sub(heavy_mutant_location, output_file_location)    
    mutant = Cage("", individual.bb_smiles, "", new_link_smiles, output_file_location, 
                  heavy_mutant_location, individual.index, "mutant", individual.shape)
    mutant.origin = "mutation"
    
    return mutant,


def BoronMutation(individual, output_file_location):

    def RingSubAllowed(atom_list):
        if len(atom_list) != 6:
            return False
            
        atom_degree_list = np.array([atom.GetDegree() for atom in atom_list])
        allowed_list1 = np.array([2,3,2,3,2,3])
        shift_list1 = allowed_list1
        allowed_list2 = np.array([3,2,3,2,2,2])
        shift_list2 = allowed_list2
        
        while True:
            shift_list1 = np.roll(shift_list1, 1)
            shift_list2 = np.roll(shift_list2, 1)
            
            if np.array_equal(shift_list1, atom_degree_list) or np.array_equal(shift_list2, atom_degree_list):
                return True
            
            if np.array_equal(allowed_list2, shift_list2):
                return False
                          
    def ConvertToBoronRing(atom_list):
        if RingSubAllowed(atom_list) == False:
            return 0
        
        for atom in atom_list:
            if atom.GetDegree() == 3:
                atom.SetAtomicNum(5)
                for neighbor in atom.GetNeighbors():
                    if neighbor in atom_list:                    
                        neighbor.SetAtomicNum(8)
        
        for atom in atom_list:
            neighbor_atoms = [neighbor.GetAtomicNum() for neighbor in atom.GetNeighbors()]            
            if 8 in neighbor_atoms:
                atom.SetAtomicNum(5)
            else:
                atom.SetAtomicNum(8)
        
        for atom in atom_list:
            atom.SetIsAromatic(False)            
            for bond in atom.GetBonds():
                bond.SetBondType(chem.BondType.SINGLE)
            
        return 1    

    def ConvertToBoronAtoms(mol):
        converted = False
        for atom in mol.GetAtoms():           
            if (atom.GetAtomicNum() == 7 and (atom.GetDegree() + atom.GetNumImplicitHs()) == 3 and
                atom.IsInRingSize(6) == False):
                atom.SetAtomicNum(5)
                converted = True
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetAtomicNum == 6 and neighbor.GetNumImplicitHs() == 2:
                        neighbor.SetAtomicNum(8)
                        converted = True
        if converted == True:
            return True
        else:                
            return False   
            
#    print(output_file_location)
    heavy_mutant_location_both = output_file_location.replace(".mol", "_both_HEAVY_ATOMS.mol")    
    output_file_location_both = output_file_location.replace(".mol", "_both.mol")    
    individual_mol_both = chem.MolFromMolFile(individual.heavy_mol_file_location)
    
    heavy_mutant_location_atom_only = output_file_location.replace(".mol", "_atom_only_HEAVY_ATOMS.mol")
    output_file_location_atom_only = output_file_location.replace(".mol", "_atom_only.mol")
    individual_mol_atom_only = chem.MolFromMolFile(individual.heavy_mol_file_location)
    
    heavy_mutant_location_ring_only = output_file_location.replace(".mol", "_ring_only_HEAVY_ATOMS.mol")
    output_file_location_ring_only = output_file_location.replace(".mol", "_ring_only.mol")
    individual_mol_ring_only = chem.MolFromMolFile(individual.heavy_mol_file_location)

    mols = [(individual_mol_both, heavy_mutant_location_both, output_file_location_both), 
            (individual_mol_atom_only, heavy_mutant_location_atom_only, output_file_location_atom_only), 
            (individual_mol_ring_only, heavy_mutant_location_ring_only, output_file_location_ring_only)]    
    
    atom_converted = ConvertToBoronAtoms(individual_mol_both)                        
    mol_ring_object = individual_mol_both.GetRingInfo().AtomRings()  
    atom_ring_list = [ [individual_mol_both.GetAtomWithIdx(atom) for atom in ring] for ring in mol_ring_object]
    for atom_ring in atom_ring_list:        
        ring_converted = False        
        ring_conversion = ConvertToBoronRing(atom_ring)
        if ring_conversion == 1:
            ring_converted = True
    
    if atom_converted == False and ring_converted == False:
        raise NameError("No atoms converted to Boron.")
        
    ConvertToBoronAtoms(individual_mol_atom_only)
    mol_ring_object = individual_mol_ring_only.GetRingInfo().AtomRings()  
    atom_ring_list = [ [individual_mol_ring_only.GetAtomWithIdx(atom) for atom in ring] for ring in mol_ring_object]
    for atom_ring in atom_ring_list:               
        ConvertToBoronRing(atom_ring)
     
    mutants = []
    for individual_mol, heavy_mutant_location, output_file_location in mols:    
        chem.MolToMolFile(individual_mol, heavy_mutant_location)
        pg.final_sub(heavy_mutant_location, output_file_location)

        bb_mol = chem.MolFromSmiles(individual.bb_smiles)
        link_mol = chem.MolFromSmiles(individual.link_smiles)        
        if "both" or "atom" in heavy_mutant_location:                
            print(bb_mol is None)            
            ConvertToBoronAtoms(bb_mol)
            print(bb_mol is None)
            print(link_mol is None)
            ConvertToBoronAtoms(link_mol)
            print(link_mol is None)
        
        if "both" or "ring" in heavy_mutant_location:        
            bb_ring_object = bb_mol.GetRingInfo().AtomRings()    
            bb_atom_ring_list = [ [bb_mol.GetAtomWithIdx(atom) for atom in ring] for ring in bb_ring_object]
            for atom_ring in bb_atom_ring_list:    
                ConvertToBoronRing(atom_ring)  
            
            link_ring_object = link_mol.GetRingInfo().AtomRings()    
            link_atom_ring_list = [ [link_mol.GetAtomWithIdx(atom) for atom in ring] for ring in link_ring_object]
            for atom_ring in link_atom_ring_list:    
                ConvertToBoronRing(atom_ring)    
            
        print(bb_mol is None, link_mol is None)    
        mutant_bb_smiles = chem.MolToSmiles(bb_mol)
        print(mutant_bb_smiles, mutant_bb_smiles is None, "bb")
        a = grf.ReverseChangeFunctionalGroupAtom(mutant_bb_smiles)
        a = chem.MolFromMolBlock(a)
        mutant_bb_smiles_prist = chem.MolToSmiles(a)             
                        
        mutant_link_smiles = chem.MolToSmiles(link_mol)
        print(mutant_link_smiles, mutant_link_smiles is None, "lk")
        a = grf.ReverseChangeFunctionalGroupAtom(mutant_link_smiles)
        a = chem.MolFromMolBlock(a)
        mutant_link_smiles_prist = chem.MolToSmiles(a)           
                
        
        mutant = Cage(mutant_bb_smiles_prist, mutant_bb_smiles, mutant_link_smiles_prist, 
                      mutant_link_smiles, output_file_location, heavy_mutant_location, individual.index, "mutant")
        mutant.origin = "mutation"
        print(atom_converted, ring_converted)
        if atom_converted == True and ring_converted == True:        
            print("...")            
            mutants.append(mutant)
        if atom_converted == True and ring_converted == False and ("atom" in mutant.heavy_mol_file_location):
            print("???")            
            mutants.append(mutant)
        if atom_converted == False and ring_converted == True and ("ring" in mutant.heavy_mol_file_location):
            print("!!!")            
            mutants.append(mutant)
            
    return mutants


def RelaxCage(individual):
    """
    This function RELAXES the geometry of a cage.....expain better.....
    """
    

    com_file_content = (
                        "{name.mae}\n"
                        "{name-out.maegz}\n"
                        " MMOD       0      1      0      0     0.0000     0.0000     0.0000     0.0000\n"
                        " DEBG      55      0      0      0     0.0000     0.0000     0.0000     0.0000\n"  
                        " FFLD      16      1      0      0     1.0000     0.0000     0.0000     0.0000\n"
                        " BDCO       0      0      0      0    41.5692 99999.0000     0.0000     0.0000\n"
                        " CRMS       0      0      0      0     0.0000     0.5000     0.0000     0.0000\n"
                        " BGIN       0      0      0      0     0.0000     0.0000     0.0000     0.0000\n"
                        " READ       0      0      0      0     0.0000     0.0000     0.0000     0.0000\n"
                        " CONV       2      0      0      0     0.1000     0.0000     0.0000     0.0000\n"
                        " MINI       1      0   5000      0     0.0000     0.0000     0.0000     0.0000\n"
                        " END        0      0      0      0     0.0000     0.0000     0.0000     0.0000"
                        )
    
    mae_location = individual.mol_file_location.replace(".mol", ".mae")   
    mae_h_location = mae_location.replace(".mae", "H.mae")
    mae_min_location = mae_h_location.replace(".mae", "-out.maegz")
    com_location = mae_location.replace(".mae", ".com")
    
    com_file_content = com_file_content.replace("{name.mae}", mae_h_location)
    com_file_content = com_file_content.replace("{name-out.maegz}", mae_min_location)    
    
    com_file = open(com_location, "w")
    com_file.write(com_file_content)    
    com_file.close()    
    
    file_present = False
    while file_present == False:       
        list_of_files = os.listdir(os.getcwd())
        if individual.mol_file_location.split("/")[-1] in list_of_files:
            file_present = True

    
    print(sp.check_output(["/opt/schrodinger/suites2015-3/utilities/structconvert", 
                           individual.mol_file_location, mae_location]).decode("utf-8"))

    os.remove(individual.mol_file_location)    
    
    file_present = False
    while file_present == False:       
        list_of_files = os.listdir(os.getcwd())
        if mae_location.split("/")[-1] in list_of_files:
            file_present = True
    try:
        print(sp.check_output(["/opt/schrodinger/suites2015-3/ligprep","-R", "h",
                           "-imae", mae_location, "-omae", mae_h_location]).decode("utf-8"))
    except:
        print("THERE IS A PROBLEM WITH THE STRUCTURE")
        print(sp.check_output(["/opt/schrodinger/suites2015-3/ligprep","-R", "h",
                           "-imae", mae_location, "-omae", mae_h_location]).decode("utf-8"))
        pass

    file_present = False
    while file_present == False:        
        list_of_files = os.listdir(os.getcwd())
        if mae_h_location.split("/")[-1] in list_of_files:
            file_present = True

    try:
        print(sp.check_output(["/opt/schrodinger/suites2015-3/macromodel",
                               com_location]).decode("utf-8"))
    except:
        print(sp.check_output(["/opt/schrodinger/suites2015-3/macromodel",
                               com_location]).decode("utf-8"))
        pass
    
    while True:
        try:            
            print(sp.check_output(["/opt/schrodinger/suites2015-3/utilities/sdconvert", 
                                   "-v3", "-imae", mae_min_location, "-osd", 
                                   individual.mol_file_location]).decode("utf-8"))
            break
    
        except:        
            pass
            
    return 1


######################################################################################################
######################################################################################################
#************************ Functions for CAGE's PROPERTY CALCULATION  ********************************#
######################################################################################################
######################################################################################################

def two_points_distance(point_a,point_b):
    
    """
    This is a function that takes XYZ positions for two points A and B and calculates the distance between them.
    This function assumes a list as an input of kind [x, y, z] and x, y, z should be floats.
    The point can be atoms XYZ coordinates, or center of mass XYZ coordinates, therefore you can calculate:
    atom-atom, com-atom, com-com distances. This equation is faster than invoking numpy.linalg
    Output: float
    """
    
    return(((point_a[0]-(point_b[0]))**2 + (point_a[1]-(point_b[1]))**2 + (point_a[2]-(point_b[2]))**2)**0.5)

def center_of_mass(atom_list):
    
    """
    This function calculates the centre of mass (COM) of a given list of atoms. It requires a list of type:
    [id, no, x, y, z], where id is the atom key identifier (It should be in uppercase and for force fields
    atom keys, they should already be deciphered with an appropriate function. Periodic table notation is required.)
    no is an index number (not relevant for this function) and x, y, z are xyz atom coordinates, respectively,
    and should be floats.
    Output: list [a, b, c] where a, b, c are COM coordinates
    """
    
    mass = 0
    mass_x = 0
    mass_y = 0
    mass_z = 0
    for i in atom_list:
        if i[0] in pg.atom_mass:
            mass += pg.atom_mass[i[0]]                 #Total mass of the compound
            mass_x += pg.atom_mass[i[0]] * i[2]        #Coordinate multiplied by the atom mass
            mass_y += pg.atom_mass[i[0]] * i[3]        # -//-
            mass_z += pg.atom_mass[i[0]] * i[4]        # -//-
        else:                                       #Atom key is not recognised and is missing from dictionary
            print('No mass for atom {0} in atom mass dictionary'.format(i[0]))
            
    return([mass_x/mass, mass_y/mass, mass_z/mass]) #Each mass*coordinate has to be divided by total mass of compound
    
def com2zero(atom_list):
    
    """
    This function first calculates the center of mass of the molecule and than translate all the coordinates
    So that new center of mass ends up in origin.
    It is done for both self.atom_list and self.xyz_array instances of this class
    """
    
    com = center_of_mass(atom_list) 
    
    return([[i[0],i[1],i[2]-com[0],i[3]-com[1],i[4]-com[2]] for i in atom_list])
    

def center_of_geometry(atom_list):
    
    """
    This function calculates the centre of geometry (COG) of a given list of atoms. It requires a list of type:
    [id, no, x, y, z], where id is the atom key identifier (It should be in uppercase and for force fields
    atom keys, they should already be deciphered with an appropriate function. Periodic table notation is required.)
    no is an index number (not relevant for this function) and x, y, z are xyz atom coordinates, respectively,
    and should be floats.
    Output: list [a, b, c] where a, b, c are COG coordinates
    """
    
    no_of_atoms = len(atom_list)                    #We need total number of molecules to take an avarage
    coordinate_a = 0
    coordinate_b = 0
    coordinate_c = 0
    for i in atom_list:
        coordinate_a += i[2]                        #We sum all coordinates for each component x,y,z
        coordinate_b += i[3]
        coordinate_c += i[4]                        #Each sum needs to be avaraged over all atoms
        
    return([coordinate_a/no_of_atoms, coordinate_b/no_of_atoms, coordinate_c/no_of_atoms])

def max_dim(atom_list):
    
    """
    This function calculates maximum molecular diameter, as a distance between two most separated atoms.
    It also adds vdw radii for the atom pair.
    Output: float
    """
    
    dimension_list = []
    for i in atom_list:
        for j in atom_list: #Calculate all atom pair distances (includes atom radii)
            d = two_points_distance(i[2:],j[2:]) + pg.atom_vdw_radii[i[0]]
            + pg.atom_vdw_radii[j[0]]                      
            dimension_list.append([i[0],i[1],j[0],j[1],d])                  #[id1, no1, id2, no2, distance]
    maximum_dimension = sorted(dimension_list, key=op.itemgetter(4))[-1]    #Find the largest distance  
    
    return(maximum_dimension[4])

def void_diameter(atom_list):
    
    """
    This function calculates the minimal internal cavity/void diameter. It calculates the biggest sphere that can be
    inserted at the center of mass before any outer atom is reached. Vdw radii are considered for this calculation.
    Output: float 
    """
    
    void_diameter_list = []
    for i in atom_list:     #Calculate the shortest distance between COM and any atom
        d = two_points_distance(i[2:],center_of_mass(atom_list)) - pg.atom_vdw_radii[i[0]]
        void_diameter_list.append([i[0],i[1],d*2])
    void_diameter = sorted(void_diameter_list, key=op.itemgetter(2))[0]     #Find the closes atom to the COM
    
    return(void_diameter[2])

def vector_analysis(vector, atom_list, increment):
    
    """
    First part of this function calculates a set of points on a vector by specified increments, 
    which are drawn in the XYZ system starting at origin.
    Output: 'pathway' list of XYZ coordinates (floats) [[a1,b1,c1],[a2,b2,c2], ... ,[an,bn,cn]]
    
    Second part calculates the diameter of the biggest sphere that can be drawn on a single point in XYZ space,
    with respect to the atom positions and their vdw radii.
    Output: float
    
    Third part of this function returns only these vectors, that have assigned sphere of positive radius value
    for the whole vector pathway
    Otherways it returns 'None'
    Output: list [[inc1, s_d1, a1, b1, c1, x, y, z], ... , [incn, s_dn, an, bn, cn, x, y, z]]
    
    inc1 - the increment on the vector assigning a point that was analysed
    s_d1 - the sphere diameter for this point
    a, b, c - XYZ coordinates of this point
    x, y, z - XYZ coordinates of analysed vector (unchanging value)
    """
    
    #First part
    pathway = []
    chunks = int(round((((vector[0])**2 + (vector[1])**2 + (vector[2])**2)**0.5) / increment, 0))
    for i in range(1, chunks+1):
        pathway.append([vector[0]/chunks*i, vector[1]/chunks*i, vector[2]/chunks*i])
        
    #Second part
    increments = 0
    values = []
    sphere_sizes = []
    for i in pathway:
        sphere_diameter_li = []
        for j in atom_list:
            sphere_diameter_li.append([j[0],j[1],(two_points_distance(j[2:], i) - pg.atom_vdw_radii[j[0]])*2])
        sphere_diameter = sorted(sphere_diameter_li, key=op.itemgetter(2))[0]
        sphere_size = sphere_diameter[2]
        if sphere_size > 0:
            values.append([increments, sphere_size, i[0], i[1], i[2], vector[0], vector[1], vector[2]])
            sphere_sizes.append(sphere_size)
        else:
            break
        increments += increment
    
    #Third part
    try:
        if len(values) == len(pathway):
            return(values[min(enumerate(sphere_sizes), key=op.itemgetter(1))[0]])
        else:
            return(None)
    except:
        return(None)

def windowCalc(atom_list, verbose=False, start=1.0, scale=0.5, std1=0.5, std2=0.5, accuracy=0.05, 
             iterations_allowed=3, psd_output=False, eps=3.8):
                 
    """
    This function is responsible for the main analysis.
    (1) It requires a file_atom_list as input, which is an attribute of the MOL class.
    (2) If verbose = True: print output messages in the terminal.
    (3) start: number of samples per square angstrom of analysis sphere surface
    (4) scale: values for which the number of samples will change with each iteration
    (5) std1: standard deviation allowed for the first iteration (deviation between windows)
    (6) std2: standard deviation allowed between iterations (deviation for each window seperatly 
    in respect to previous iteration value).
    (7) accuracy: allowed difference between the maximum window diameter found in the first 
    analysis and the meanvalue of the windows for each iteration.
    (8) iterations_allowed: number of allowed iterations to be done if all the previous conditions 
    won't be met. This is to prevent going into an infinite loop or if it impossible for the structure
    to meet the conditions specified.
    (9) psd_output: if the txt file is ready to be plotted the PSD should be saved at the end.
    (10) eps: density of data points required to create a cluster from them in the DBSCAN 
    3.8 based on CC3 and MC6 mean values of min/max for 1 point per 1 angstrom.
    """
    
    com = center_of_mass(atom_list)                                 #Center of mass
    cog = center_of_geometry(atom_list)                             #Center of geometry (mass distribution can be uneven)
    atom_list = com2zero(atom_list)                                 #Shift COM to origin for the sake of simplicity
    com_zero = center_of_mass(atom_list)                            #Double check that it is shifted to the origin
    maximum_dimension = max_dim(atom_list)                          #Maximum dimension of the cage
    void_diam = void_diameter(atom_list)                            #Internal cavity diameter
    
    if verbose == True:
        print("    COM: {0:.2f} {1:.2f} {2:.2f}".format(*com))
        print("    COG: {0:.2f} {1:.2f} {2:.2f}".format(*cog))
        print("    COM_0: {0:.2f} {1:.2f} {2:.2f}".format(*com_zero))
        print("    Maximum dimension {0:.2f}".format(maximum_dimension))
        print("    Void diameter {0:.2f}".format(void_diam))
        print("    Kr volume % of void volume: {0:.2f}".format(3.69/void_diam*100))
        print("    Xe volume % of void volume: {0:.2f}".format(4.10/void_diam*100))
    
    iteration_main = 0                                              #Start counting the iterations     
    std_1 = 100                                                     #On its own. Further on these values are being modified and this allowed
    std_2 = 1                                                       #The script to decide, whether further iterations are neccessery
    biggest_window = 1
    windows = [0]
    
    while (std_1 > std1 and iteration_main < iterations_allowed or std_2 > std2 and iteration_main < iterations_allowed and iteration_main < iterations_allowed):
#        or biggest_window > max(windows) + accuracy*max(windows) 
        iteration_main += 1
        
        print("\n    ITERATION {0}, SAMPLING SCALE {1}".format(iteration_main,start))
        R = maximum_dimension/2                                     #Finding out the sphere radius as a half of maximum cage dimension
        sphere_surface_area = 4 * np.pi * R * R                     #Calculating surface area of a halo sphere
        
        #Number of sampling points on the halo sphere
        number_of_points = int(round(sphere_surface_area*start, 0)) #This can be scaled!
        print("    Number of sampling points: {0}".format(number_of_points))

        #Estimating golden angle between the points on the sphere and creating a halo points sphere
        golden_angle = np.pi * (3 - np.sqrt(5))
        theta = golden_angle * np.arange(number_of_points)
        z = np.linspace(1 - 1.0 / number_of_points, 1.0 / number_of_points - 1.0, number_of_points)
        radius = np.sqrt(1 - z * z)

        points = np.zeros((number_of_points, 3))
        points[:,0] = radius * np.cos(theta) * R
        points[:,1] = radius * np.sin(theta) * R
        points[:,2] = z * R

#        results = []
#        for x in points:
#            results.append(vector_analysis(x, atom_list, 1.0))
#        dataset = [x[5:8] for x in results if x is not None]
#        all_basins = [x[1] for x in results if x is not None]
#        biggest_window = max(all_basins)
#        smallest_window = min(all_basins)
#        print("  Biggest window: {0}".format(biggest_window))
#        print("  Smallest window: {0}".format(smallest_window))
#        print("  Sampling points {0}".format(len(dataset)))
        
        pool = mp.Pool(processes=12)
        parallel = [pool.apply_async(vector_analysis, args=(x, atom_list, 1.0)) for x in points]
        results = [p.get() for p in parallel if p.get() is not None]
#        print("\n\n\n****************RESULTS : **************************\n\n\n")
#        print(results)
        pool.terminate()
        dataset = [x[5:8] for x in results]
        all_basins = [x[1] for x in results]
#        print("\n\n\n****************DATASET : **************************\n\n\n")
#        print(dataset)
#        print("\n\n\n****************ALL BASINS : **************************\n\n\n")
#        print(all_basins)
#        print("\n\n\n******************************************\n\n\n")
        
        if len(dataset) and len(all_basins) != 0:
            biggest_window = max(all_basins)
            smallest_window = min(all_basins)
#            print("    Biggest window: {0}".format(biggest_window))
#            print("    Smallest window: {0}".format(smallest_window))
#            print("    Sampling points {0}".format(len(dataset)))
    
            X = np.array(dataset)
    
            # Compute DBSCAN
            db = DBSCAN(eps=eps).fit(X) 
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_
    
            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
#            print("    Estimated number of clusters: {0}".format( n_clusters_))
    
            clusters = []
            for i,j in zip(X, db.labels_):
                clusters.append([i,j])
    
            golden_vectors = []
            for i in range(n_clusters_):
                x = 0
                y = 0
                z = 0
                no_of_samples = 0
                for j in clusters:
                    if j[1] == i:
                        no_of_samples += 1
                        x += j[0][0]
                        y += j[0][1]
                        z += j[0][2]
                #print(no_of_samples)
                golden_vectors.append([x/no_of_samples, y/no_of_samples, z/no_of_samples])
            
            windows = []
            if iteration_main == 1:
                for i in golden_vectors:
                    window_d = vector_analysis(i, atom_list, 0.1)
                    if window_d is not None:
                        if window_d[1] >= 2.18/2: #Minimum molecular diameter of hydrogen
                            windows.append(window_d[1])
#                print("    Estimated number of windows: {0}".format(len(windows)))
                
                if len(windows) > 0:
                    windows = np.array(windows)
                    windows = np.sort(windows)
#                    print("    Windows diameters: {0}".format(windows))
#                    print("    Mean value: {0}".format(np.mean(windows)))
                
                    std_1 = np.std(windows) / np.mean(windows) * 100
#                    print("    STD1 {0}".format(std_1))
                    windows = windows.tolist()
                else:
                    windows = [0]
                    biggest_window = 1
                start += scale
                
            else:
                old_windows = windows
            
                for i in golden_vectors:
                    window_d = vector_analysis(i, atom_list, 0.1)
#                    print("\n\n **** WINDOW_D : ****** \n\n")
#                    print(window_d)
                    if window_d is not None:
                        if window_d[1] >= 2.18/2: #Minimum molecular diameter of hydrogen
                            windows.append(window_d[1])          #POTENTIAL PROBLEM HERE, should use np.append??
#                    print("    Estimated number of windows: {0}".format(len(windows)))
              
                    if len(windows) > 0:
                        windows = np.array(windows)
                        windows = np.sort(windows)
#                        print("    Windows diameters: {0}".format(windows))
#                        print("    Mean value: {0}".format(np.mean(windows)))
                    
                        std_1 = np.std(windows) / np.mean(windows) * 100
#                        print("    STD1 {0}".format(std_1))
                        windows = windows.tolist()
                        if std_1 <= 1:
                            std_2 = 0
                        else:
                            new_list = []
                            for i,j in zip(old_windows,windows):
                                new_list.append([i,j])
                            std_2 = 0
                            for i in new_list:
                                std_2 = std_2 + (np.std(np.array(i))/np.mean(np.array(i))*100)
                            if len(new_list) != 0:
                                std_2 = std_2 / len(new_list)
                            else:
                                std_2 = std_2
                            std_1 = 0.5
#                            print("    STD2 {0}".format(std_2))
                            #print('STD1 prim {}'.format(std_1))
                                            
                    elif len(windows) == 0 and len(old_windows) > len(windows):
                        windows = [0]
                        biggest_window = 1
                    else:
                        windows = [1]
                        biggest_window = 0
                    start += scale
#                    print('End time {0}\n'.format(time.strftime("%H:%M:%S")))

        else:       #Case in whicn Dataset and AllBasins are empty
            windows = [0]
                

    return(windows)    
        

def CalcEnergy(individual, bb_num, bb_sub_atoms, lk_num):
    """
    This function calculates the PM7 reaction energy (eV) for the cage formation
    Calling the mopac_run function and comparing the energy of the cage, the 
    number of molecules created during imine condensation with the energy of the bb and link.
    """
    def mopac_run(mol_input):
        """
        This function optimizes the molecular structure at the PM7 semiempirical
        level calling MOPAC2012. The function takes in as parameters the name of the
        input file and returns the energy of the optimized molecule.
        Babel needs to be installed on the machine, and its path should be defined 
        explicitly.
        MOPAC2012 should be 
        """
        
        mopac_input = mol_input.replace("mol", "mop")
        mopac_run_file = mol_input.replace(".mol", "_run.mop")
        mopac_run_out = mopac_run_file.replace(".mop", ".out")
        
        file_present = False
        while file_present == False:
            list_of_files = os.listdir(os.getcwd())
            if mol_input.split("/")[-1] in list_of_files:
                file_present = True
        
        # Here the path to babel should be stated explicitly
        print(sp.check_output(["babel", mol_input, mopac_input]).decode("utf-8"))
        
        file_present = False
        while file_present == False:          
            list_of_files = os.listdir(os.getcwd())
            if mopac_input.split("/")[-1] in list_of_files:
                file_present = True
        
        # Modifying the standard input file into a PM7 optimization
        with open(mopac_run_file, "w") as new_mop:    
            with open(mopac_input, "r") as mop:
                for line in mop:
                    new_mop.write(line.replace("PUT KEYWORDS HERE", "PM7 XYZ 1SCF"))
        mop.close()
        new_mop.close()
        
        # Here the path to MOPAC should be stated explicitly
        print(sp.check_output(["/opt/mopac/MOPAC2012.exe", mopac_run_file]).decode("utf-8"))
        
        file_present = False
        while file_present == False:            
            list_of_files = os.listdir(os.getcwd())
            if mopac_run_out.split("/")[-1] in list_of_files:
                file_present = True
        
        # Looks for the out file and once that is generated stores the final energy        
        with open(mopac_run_out, "r") as out:
            for line in out:
                if "TOTAL ENERGY" in line:
                    energy = float(line.split()[3])
                    
        return energy    
    
    # Execute mopac_run on each different item (cage, bb, link, H2O)
    num_waters = bb_num * bb_sub_atoms
    bb_mol = chem.MolFromSmiles(individual.bb_smiles_prist)
    bb_mol = chem.AddHs(bb_mol)
    try:
        ac.EmbedMolecule(bb_mol)
        ac.MMFFOptimizeMolecule(bb_mol)
        bb_file_location = individual.mol_file_location.replace(".mol","_bb.mol")
#        print(bb_file_location)    
        chem.MolToMolFile(bb_mol, bb_file_location)
        energy_bb = mopac_run(bb_file_location)
    except:
        return 0
    
    link_mol = chem.MolFromSmiles(individual.link_smiles_prist)
    link_mol = chem.AddHs(link_mol)
    try:
        ac.EmbedMolecule(link_mol)    
        ac.MMFFOptimizeMolecule(link_mol)
        link_file_location = individual.mol_file_location.replace(".mol","_lk.mol")
        chem.MolToMolFile(link_mol, link_file_location)    
#        print(link_file_location)
        energy_link = mopac_run(link_file_location)
    except:
        return 0
    
    try:
        energy_cage = mopac_run(individual.mol_file_location)
#        print(individual.mol_file_location)
    except:
        return 0
    
    # H2O Energy calculated at PM7 level (eV)
    energy_water = -322.67921
    
    """
    Calculate reaction energy. The number of water molecules is equivalent at
    the number of bb in the cage, multiplied for the number of functionalities
    in each bb.
    """
    try:
#        print("bb_num is: ", bb_num, "\n")
#        print("lk_num is: ", lk_num, "\n")
#        print("num_waters is: ", num_waters, "\n")
        final_energy =( (energy_cage + (int(num_waters) * energy_water)) -
        ((int(bb_num) * energy_bb) + (int(lk_num) * energy_link)) )/ num_waters
        
        individual.energy = final_energy    
      
        return individual.energy
    
    except:
        return 0

def InteractionEnergy(individual):

    """
    Check the quality of the OPLS3 parameters from the log file. If they are not good enough print a WARNING.
    Looks for the out file and once that is generated stores the final energy
    """
    log_min_location = individual.mol_file_location.replace(".mol", ".log")
    
    with open(log_min_location, "r") as out:
        for line in out:
            if "Numbers of high, medium and low quality stretch parameters" in line:
                stretch_high = float(line.split()[-3])
                stretch_med = float(line.split()[-2])
                stretch_low = float(line.split()[-1])
            if "Numbers of high, medium and low quality bend parameters" in line:
                bend_high = float(line.split()[-3])
                bend_med = float(line.split()[-2])
                bend_low = float(line.split()[-1])
            if "Numbers of high, medium and low quality torsion parameters" in line:
                tor_high = float(line.split()[-3])
                tor_med = float(line.split()[-2])
                tor_low = float(line.split()[-1])
        total_par = stretch_high + stretch_med + stretch_low + bend_high + bend_med + bend_low + tor_high + tor_med + tor_low
        high_par = ((stretch_high + bend_high + tor_high)/ total_par ) * 100
        med_par = ((stretch_med + bend_med + tor_med)/ total_par ) * 100
        low_par = ((stretch_low + bend_low + tor_low)/ total_par ) * 100
        
        ## Return a WARNING when medium + low parameters are higher than 20 %
        if (med_par + low_par) >= 20:
            print("\n\n  ***********************************************************")
            print("  ************************* WARNING *************************\n")
            print("  OPLS3 has {0} % of medium and low parameters for structure {1}\n".format((med_par + low_par), log_file_location.split("/")[-1]))
            print("  ***********************************************************\n\n")     
    """
    Collect all the interatction energy contributions (VDW and Electrostatic) from the log file. Energies are shown in kJ/mol.
    The function return the interaction energy
    """
    with open(log_min_location, "r") as out:
        for line in out:
            energy_vdw = []
            energy_elect = []
            if "                            VDW =     " in line:
                energy_vdw.append(float(line.split()[-2]))
            if "                  Electrostatic =     " in line:
                energy_elect.append(float(line.split()[-2]))
        
        energy_interaction = energy_vdw[-1] + energy_elect[-1]
    
    return(energy_interaction)
       
######################################################################################################
######################################################################################################
#************************ Functions for the FITNESS FUNCTION DEFINITION  ****************************#
######################################################################################################
######################################################################################################

def FitnessFunction(individual, desired_size, bb_num, bb_sub_atoms, lk_num):
    
    """
    Explain how this works, and improve the function allowing customisation of parameters
    """
    
    file_present = False
    start = time.time()    
    while file_present == False:
        list_of_files = os.listdir(os.getcwd())      
        end = time.time()               
        if individual.mol_file_location.split("/")[-1] in list_of_files:
            file_present = True
        if end - start > 120:
            individual.fitness_value = 0            
            return 0
    
    ## Calculation of window differences
    fileWCalc = MOL(individual.mol_file_location)
    
    individual.windows = windowCalc(fileWCalc.atom_list)
    
    ideal_ratio = 1/(desired_size ** 2)
    
    if individual.windows != None:
        try:
            max_window = max(individual.windows)
            individual.max_window = max_window
            if individual.max_window != 0:
                window_ratio = 1/(individual.max_window ** 2)
            else:
                window_ratio = 200
            window_ratio_tot = abs(ideal_ratio - window_ratio) * 100
            individual.window_ratio = window_ratio_tot
            
        except:
            individual.max_window = 0.5
            window_ratio = 1/(individual.max_window ** 2)
            window_ratio_tot = abs(ideal_ratio - window_ratio) * 100
            individual.window_ratio = window_ratio_tot
        
    if len(individual.windows) == 1 and any(individual.windows) == 0 or len(individual.windows) < individual.window_num:
        window_diff = 500
    else:    
        if len(individual.windows) > individual.window_num:
            individual.windows = individual.windows[:individual.window_num]
        else:
            pass
    
        window_diff = 0.0                   # Defining the initial window difference
        for i,j in itertools.combinations(range(len(individual.windows)), 2):
    
            window1 = individual.windows[i]
            window2 = individual.windows[j]
            window_diff += abs(window1 - window2)
        
        if len(individual.windows) > 1: 
            diff_num = 0
            i = len(individual.windows) - 1
        
            while i > 0:
                diff_num += i
                i -= 1
        else:
            diff_num = 1
        
        window_diff = ((window_diff/diff_num)** 2) * 100
    
    individual.window_diff = window_diff
        

    """
    Calculate the cage volume and divide it for the Cavity Size.
    The smaller the CageVolume/CavitySize is the better the structure is.
    The value will be substracted from the individual's val in order to obtain
    an improved fitness_value.
    """
    mae_min_location= individual.mol_file_location.replace(".mol", ".mae") 

    try:

        if desired_size <= 10:
            try:
                val = (abs(desired_size - grf.GetCavitySize(individual.mol_file_location)) * 5)**2
#                print("********************************DIFFERENCE WITH EXPECTED CAVITY ", val, " ***************************")
#                print("********************************WINDOW DIFF IS        ", window_diff, "   ***************************")
#                print("********************************WINDOW RATIO IS        ", window_ratio, "   ***************************")
                
            except:
                val = 500
        else:
            try:
                val = abs(desired_size - grf.GetCavitySize(individual.mol_file_location)) * 50
            except:
                val = 500
            
        individual.val = val
        
        # Divide the final fitness function by 1000 in order to normalize the value of the fitness function over 1
        individual.fitness_value  = (1000 - val - window_diff - window_ratio_tot) / 1000
        
#        if individual.fitness_value < 0.0:
#             individual.fitness_value = 0.0
        print("\n   INDIVIDUAL FITNESS: ", individual.mol_file_location.split("/")[-1], "\n", individual.fitness_value, "    SHAPE = ", individual.shape, "\n")

    except:

        individual.fitness_value = 0.0
        print("\n   INDIVIDUAL FITNESS: ", individual.mol_file_location.split("/")[-1], "\n", individual.fitness_value, "    SHAPE = ", individual.shape, "\n")
    
    return individual.fitness_value
    
def CalculatePopulationFitness(population, desired_size, bb_num, bb_sub_atoms, lk_num):
    
    for individual in population:        
        FitnessFunction(individual, desired_size, bb_num, bb_sub_atoms, lk_num)
    
    return 1


######################################################################################################
######################################################################################################
#******************************* Functions for POPULATION HANDLING  *********************************#
######################################################################################################
######################################################################################################


def AddPopulationToTrackingList(population, tracking_list, tracking_list_index):
    
    for individual in population:
        tracking_list.append(individual)
        individual.index = tracking_list_index
        tracking_list_index += 1            
        
    return 1


def GetFamilyTree(individual, individuals_list):  
    
    output = []
    
    def inside(ind):

        output.append((str(ind.index), str(ind.fitness_value), str(ind.parent1), 
                       str(ind.parent2), str(ind.origin), str(ind.origin_gen)))

        if ind.parent2 == "mutant":
            inside(individuals_list[ind.parent1])          
        
        if ind.parent1 != "initial" and ind.parent2 != "mutant":
            inside(individuals_list[ind.parent1])
            inside(individuals_list[ind.parent2])
            
       
    inside(individual)
    
    return output
        
        
def SelBest(list_of_individuals, selection_number):
    
    """
    This function selects the best individuals from the larger population.............     
    """
    
    sorted_ind_list = sorted(list_of_individuals, key=lambda individual: individual.fitness_value, reverse = True)
    
    no_duplicate_list = []
    info_list = []
    while len(no_duplicate_list) <= selection_number:
        for ind in sorted_ind_list:
            print("\n\n\n ************* SELECTION BEST STEP ************* \n\n\n")
            print("  Individual TEST: ", ind.mol_file_location)
            if (ind.bb_smiles_prist, ind.link_smiles_prist, ind.shape) not in info_list:
                no_duplicate_list.append(ind)
                print("\n\n  BB SMILE: {0}\n, LINK SMILE: {1}\n, SHAPE: {2}\n\n".format(ind.bb_smiles_prist, ind.link_smiles_prist, ind.shape))
                print("  Individual accepted: ", ind.mol_file_location)
                info_list.append((ind.bb_smiles_prist, ind.link_smiles_prist, ind.shape))
            else:
                print("  Individual rejected: ", ind.mol_file_location)
                
    sorted_individuals = sorted(no_duplicate_list, key=lambda individual: individual.fitness_value, reverse = True)
    return sorted_individuals[:(selection_number)]    
    
def StochasticSampling(list_of_individuals, selection_number):
    
    """
    This function selects the best individuals from the larger population.............     
    """
    
    no_duplicate_list = []
    info_list = []
    for ind in list_of_individuals:
        if (ind.bb_smiles_prist, ind.link_smiles_prist, ind.shape) not in info_list:
            no_duplicate_list.append(ind)
            info_list.append((ind.bb_smiles_prist, ind.link_smiles_prist, ind.shape))
            
    selected_pop = []
    
    no_duplicate_list = [individual for individual in no_duplicate_list if individual.fitness_value >= 0.0]
    print("SIZE OF NO DUPLICATE LIST IS = ", len(no_duplicate_list))
    
    total_fitness = sum(individual.fitness_value for individual in no_duplicate_list)

    distance_pointers = total_fitness/selection_number
    
    start = random.uniform(0.0, distance_pointers)

    fitness_count = 0
    
    while len(selected_pop) < selection_number:
        print("\n\n  **********************************")
        print("  POP = ", len(selected_pop), 
        "  SELECTION NUMBER =", selection_number, "DISTANCE POINTER =", distance_pointers)
        individual = random.choice(no_duplicate_list)
        fitness_count += individual.fitness_value
        print("  INDIVIDUAL FITNESS VALUE = ", individual.fitness_value, "FITNESS COUNT = ", fitness_count)
        print("  Initital START = ", start)
        if fitness_count >= start:
            selected_pop.append(individual)
            individual.fitness_value -= distance_pointers 
            start += distance_pointers
        print("  Final START = ", start)
        print("  Final POP = ", len(selected_pop))
        print("  **********************************\n\n")
    
    print("\n\n  ***THIS IS THE SIZE OF THE POPULATION SELECTED FOR MATING: ", len(selected_pop), "****\n\n")
    for item in selected_pop:
        print(item.mol_file_location, "\n\n")
        
    return selected_pop


######################################################################################################
######################################################################################################
#**** This function handles the PARALLELISATION of the relaxation step for each individual **********#
######################################################################################################
######################################################################################################

def RelaxPopulation(population, processors):
    """
    Explain how does this work...
    """
    
    pool = mp.Pool(processors)
    pool.map_async(RelaxCage, population)
    pool.close()
    pool.join()    
    return 1
