# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:45:58 2015

@author: Enrico
"""

import numpy as np
import operator as op



#Atom mass dictionary.
atom_mass = {"H"  :  1.0078, "He" :  4.0026, "Li" :  6.9410, "Be" :  9.0122,
            "B"  : 10.8110, "C"  : 12.0000, "N"  : 14.0031, "O"  : 15.9994,
            "F"  : 18.9984, "Ne" : 20.1797, "Na" : 22.9898, "Mg" : 24.3050,
            "Al" : 26.9815, "Si" : 28.0855, "P"  : 30.9738, "S"  : 32.0650,
            "Cl" : 35.4530, "Ar" : 39.9480, "K"  : 39.0983, "Ca" : 40.0780,
            "Sc" : 44.9559, "Ti" : 47.8670, "V"  : 50.9415, "Cr" : 51.9961,
            "Mn" : 54.9380, "Fe" : 55.8450, "Co" : 58.9331, "Ni" : 58.6934,
            "Cu" : 63.5460, "Zn" : 65.4090, "Ga" : 69.7230, "Ge" : 72.6400,
            "As" : 74.9216, "Se" : 78.9600, "Br" : 79.9040, "Kr" : 83.7980,
            "Rb" : 85.4678, "Sr" : 87.6200, "Y"  : 88.9059, "Zr" : 91.2240,
            "Nb" : 92.9060, "Mo" : 95.9400, "Tc" : 98.1000, "Ru" : 101.0700,
            "Rh" : 102.9050, "Pd" : 106.4200, "Ag" : 107.8682, "Cd" : 112.4110,
            "In" : 114.8180, "Sn" : 118.7100, "Sb" : 121.7600, "Te" : 127.6000,
            "I"  : 126.9040, "Xe" : 131.2930, "Cs" : 132.9055, "Ba" : 137.3270,
            "La" : 138.9055, "Ce" : 140.1160, "Pr" : 140.9077, "Nd" : 144.2420,
            "Pm" : 145.1000, "Sm" : 150.3600, "Eu" : 151.9640, "Gd" : 157.2500,
            "Tb" : 158.9254, "Dy" : 162.5000, "Ho" : 164.9300, "Er" : 167.2590,
            "Tm" : 168.9342, "Yb" : 173.0400, "Lu" : 174.9670, "Hf" : 178.4900,
            "Ta" : 180.9479, "W"  : 183.8400, "Re" : 186.2070, "Os" : 190.2300,
            "Ir" : 192.2170, "Pt" : 195.0840, "Au" : 196.9666, "Hg" : 200.5900,
            "Tl" : 204.3833, "Pb" : 207.2000, "Bi" : 208.9804
            }

#Atom vdW radii dictionary taken from www.ccdc.cam.ac.uk/Lists/ResourceFileList/Elemental_Radii.xlsx 13 Oct 2015
#Excluding unstable radioisotopes, dummy atom denoted X (and D) have atomic vdW radii equal 1
atom_vdw_radii = {
                  'Al': 2,    'Sb': 2,    'Ar': 1.88, 'As': 1.85, 'Ba': 2,    'Be': 2,    'Bi': 2, 
                  'B':  2,    'Br': 1.85, 'Cd': 1.58, 'Cs': 2,    'Ca': 2,    'C':  1.7,  'Ce': 2, 
                  'Cl': 1.75, 'Cr': 2,    'Co': 2,    'Cu': 1.4,  'Dy': 2,    'Er': 2,    'Eu': 2,
                  'F':  1.47, 'Gd': 2,    'Ga': 1.87, 'Ge': 2,    'Au': 1.66, 'Hf': 2,    'He': 1.4,
                  'Ho': 2,    'H':  1.09, 'In': 1.93, 'I':  1.98, 'Ir': 2,    'Fe': 2,    'Kr': 2.02,
                  'La': 2,    'Pb': 2.02, 'Li': 1.82, 'Lu': 2,    'Mg': 1.73, 'Mn': 2,    'Hg': 1.55,
                  'Mo': 2,    'Nd': 2,    'Ne': 1.54, 'Ni': 1.63, 'Nb': 2,    'N':  1.55, 'Os': 2,
                  'O':  1.52, 'Pd': 1.63, 'P':  1.8,  'Pt': 1.72, 'K':  2.75, 'Pr': 2,    'Pa': 2,
                  'Re': 2,    'Rh': 2,    'Rb': 2,    'Ru': 2,    'Sm': 2,    'Sc': 2,    'Se': 1.9,
                  'Si': 2.1,  'Ag': 1.72, 'Na': 2.27, 'Sr': 2,    'S':  1.8,  'Ta': 2,    'Te': 2.06,
                  'Tb': 2,    'Tl': 1.96, 'Th': 2,    'Tm': 2,    'Sn': 2.17, 'Ti': 2,    'W':  2,
                  'U':  1.86, 'V':  2,    'Xe': 2.16, 'Yb': 2,    'Y':  2,    'Zn': 1.29, 'Zr': 2,
                  'X':  1.0,  'D':  1.0
                 }
    
#Atom covalent radii dictionary taken from www.ccdc.cam.ac.uk/Lists/ResourceFileList/Elemental_Radii.xlsx 13 Oct 2015
#Excluding unstable radioisotopes, dummy atom denoted X (and D) have atomic cov radii equal 1
atom_cov_radii = {
                  'Al': 1.21, 'Sr': 1.39, 'Ar': 1.51, 'As': 1.21, 'Ba': 2.15, 'Be': 0.96, 'Bi': 1.48, 
                  'B':  0.83, 'Br': 1.21, 'Cd': 1.54, 'Cs': 2.44, 'Ca': 1.76, 'C':  0.68, 'Ce': 2.04, 
                  'Cl': 0.99, 'Cr': 1.39, 'Co': 1.26, 'Cu': 1.32, 'Dy': 1.92, 'Er': 1.89, 'Eu': 1.98,
                  'F':  0.64, 'Gd': 1.96, 'Ga': 1.22, 'Ge': 1.17, 'Au': 1.36, 'Hf': 1.75, 'He': 1.5,
                  'Ho': 1.92, 'H':  0.23, 'In': 1.42, 'I':  1.4,  'Ir': 1.41, 'Fe': 1.52, 'Kr': 1.5,
                  'La': 2.07, 'Pb': 1.46, 'Li': 1.28, 'Lu': 1.87, 'Mg': 1.41, 'Mn': 1.61, 'Hg': 1.32,
                  'Mo': 1.54, 'Nd': 2.01, 'Ne': 1.5,  'Ni': 1.24, 'Nb': 1.64, 'N':  0.68, 'Os': 1.44,
                  'O':  0.68, 'Pd': 1.39, 'P':  1.05, 'Pt': 1.36, 'K':  2.03, 'Pr': 2.03, 'Pa': 2,
                  'Re': 1.51, 'Rh': 1.42, 'Rb': 2.2,  'Ru': 1.46, 'Sm': 1.98, 'Sc': 1.7,  'Se': 1.22,
                  'Si': 1.2,  'Ag': 1.45, 'Na': 1.66, 'Sr': 1.95, 'S':  1.02, 'Ta': 1.7,  'Te': 1.47,
                  'Tb': 1.94, 'Tl': 1.45, 'Th': 2.06, 'Tm': 1.9,  'Sn': 1.39, 'Ti': 1.6,  'W': 1.62,
                  'U':  1.96, 'V':  1.53, 'Xe': 1.5,  'Yb': 1.87, 'Y':  1.9,  'Zn': 1.22, 'Zr': 1.75,
                  'X':  1.0,  'D':  1.0
                 }


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


     
def two_points_distance(point_a,point_b):
    """
    This is a function that takes XYZ positions for two points A and B and calculates the distance between them.
    This function assumes a list as an input of kind [x, y, z] and x, y, z should be floats.
    The point can be atoms XYZ coordinates, or center of mass XYZ coordinates, therefore you can calculate:
    atom-atom, com-atom, com-com distances. This equation is faster than invoking numpy.linalg
    Output: float
    """
    return(((point_a[0]-(point_b[0]))**2 + (point_a[1]-(point_b[1]))**2 + (point_a[2]-(point_b[2]))**2)**0.5)

def max_dim(atom_list):
    """
    This function calculates maximum molecular diameter, as a distance between two most separated atoms.
    It also adds vdw radii for the atom pair.
    Output: float
    """
    dimension_list = []
    for i in atom_list:
        for j in atom_list: #Calculate all atom pair distances (includes atom radii)
            d = two_points_distance(i[2:],j[2:]) + atom_vdw_radii[i[0].upper()]                 + atom_vdw_radii[j[0].upper()]                      
            dimension_list.append([i[0],i[1],j[0],j[1],d])                  #[id1, no1, id2, no2, distance]
    maximum_dimension = sorted(dimension_list, key=op.itemgetter(4))[-1]    #Find the largest distance  
    return(maximum_dimension[4])

def void_diameter(atom_list):
    """
    This function calculates the minimal internal cavity/void diameter. I calculates the biggest sphere that can be
    inserted in the place of center of mass, with respect to the closest atom. It considers vdw radii.
    Output: float 
    """
    void_diameter_list = []
    for i in atom_list: #Calculate the shortest distance between COM and any atom
        d = two_points_distance(i[2:],center_of_mass(atom_list)) - atom_vdw_radii[i[0].upper()]
        void_diameter_list.append([i[0],i[1],d*2])
    void_diameter = sorted(void_diameter_list, key=op.itemgetter(2))[0] #Find the closes atom to the COM
    return(void_diameter[2])
    
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
        if i[0] in atom_mass:
            mass += atom_mass[i[0]]                 #Total mass of the compound
            mass_x += atom_mass[i[0]] * i[2]        #Coordinate multiplied by the atom mass
            mass_y += atom_mass[i[0]] * i[3]        # -//-
            mass_z += atom_mass[i[0]] * i[4]        # -//-
        else:                                       #Atom key is not recognised and is missing from dictionary
            print('No mass for atom {1} in atom mass dictionary'.format(i[0]))
    return([mass_x/mass, mass_y/mass, mass_z/mass]) #Each mass*coordinate has to be divided by total mass of compound
    
def com2zero(atom_list):
    """
    This function first calculates the center of mass of the molecule and than translate all the coordinates
    So that new center of mass ends up in origin.
    It is done for both self.atom_list and self.xyz_array instances of this class
    """
    com = center_of_mass(atom_list) 
    return([[i[0],i[1],i[2]-com[0],i[3]-com[1],i[4]-com[2]] for i in atom_list])
    
path ="/Users/Enrico/Documents/PostDoc/Programming/Notebooks/GA/v14october_2015/test_CC1/cc1.mol"
test = MOL(path)

print("\n\n  The cage's void is: {0} Angstroms \n\n".format(void_diameter(test.atom_list)))
