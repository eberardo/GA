import os as os

from rdkit import Chem as chem
from rdkit.Chem import AllChem as ac
import math as m
import numpy as np
from scipy.spatial import distance 
from operator import itemgetter
import itertools
import random
import GA_rdkit_functions as grf

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


def center_of_mass(mol):

    """ 
    Function to calculate the center of mass of the system and translocate it to the origin
    of the coordinate system. This function assumes a list as an input of kind [id, x, y, z].
    It is expected for the atom ID to be a string containing the first upper case letter and
    second lower case letter for the atom element. x, y, z are expected to be floats.
    It returns (1) a list of three floats, which define the coordinates of the COM and 
    (2) the molecule [id, x, y, z] with all the coordinates translated so that the COM is 
    placed at the origin.
    """ 

    mass_atom_list = []
    new_mol = []
    for item in mol:
        mass_atom_list.append(float(atom_mass[item[0]]))

    mass_sum = sum(mass_atom_list)
    mass_x = 0
    mass_y = 0
    mass_z = 0
    mass_center_x = 0
    mass_center_y = 0
    mass_center_z = 0

    for item in range(len(mol)):
        mass_x += mass_atom_list[item] * mol[item][1]
        mass_y += mass_atom_list[item] * mol[item][2]
        mass_z += mass_atom_list[item] * mol[item][3]

    mass_center_x = mass_x/mass_sum
    mass_center_y = mass_y/mass_sum
    mass_center_z = mass_z/mass_sum

    ## Translocating the system with the center of mass as origin
    for item in mol:
        new_x = item[1] - mass_center_x
        new_y = item[2] - mass_center_y
        new_z = item[3] - mass_center_z
        new_mol.append([item[0], float(new_x), float(new_y), float(new_z)])

    return([mass_center_x, mass_center_y, mass_center_z], new_mol)



def center_of_geometry(mol):
    """
    This function calculates the centre of geometry (COG) of a given list of atoms. This function assumes 
    a list as an input of kind [id, x, y, z]. It is expected for the atom ID to be a string containing the 
    first upper case letter and second lower case letter for the atom element. x, y, z are expected to be 
    floats. 
    Output: list [a, b, c] where a, b, c are COG coordinates. **** 
    [TAKEN FROM MARCIN's CODE 20/10/2015]
    """
    no_of_atoms = len(mol)                    #We need total number of molecules to take an avarage
    coordinate_a = 0
    coordinate_b = 0
    coordinate_c = 0
    for i in mol:
        coordinate_a += i[2]                        #We sum all coordinates for each component x,y,z
        coordinate_b += i[3]
        coordinate_c += i[4]                        #Each sum needs to be avaraged over all atoms
    
    return([coordinate_a/no_of_atoms, coordinate_b/no_of_atoms, coordinate_c/no_of_atoms])


def mid_point(point_a, point_b):

    """
    This function takes the XYZ positions of two points A and B and calculates the mid point between them.
    The two points can represent atom positions or (XYZ coordinates), or center of mass XYZ coordinates.
    """

    point_x = (point_b[0] + point_a[0])/2
    point_y = (point_b[1] + point_a[1])/2
    point_z = (point_b[2] + point_a[2])/2

    return([point_x, point_y, point_z])


def final_sub(input_addition, output_structure_file_name):
    with open(input_addition, "r") as add:
        new_file= ""
        for line in add:
            if "Y" in line:
                new_file += line.replace("Y", "C")
            elif "Zr" in line:
                new_file += line.replace("Zr", "C")
            elif "Nb" in line:
                new_file += line.replace("Nb", "C")
            elif "Mo" in line:
                new_file += line.replace("Mo", "C")
            elif "Tc" in line:
                new_file += line.replace("Tc", "O")
            elif "Ru" in line:
                new_file += line.replace("Ru", "S")
            elif "Rh" in line:
                new_file += line.replace("Rh", "N")
            elif "Pd" in line:
                new_file += line.replace("Pd", "N")
            else:
                new_file += line

    f = open(output_structure_file_name, "w")
    f.write(new_file)
    f.close()

    return 1


def polygon(shape):

    """
    This function calculates vertices and midpoint edges of the specified polygon.
    Options: Trigonal Bipyramid [3 + 2], Cube, Tetrahedron, Dodecahedron and (Prism).
    """

    global new_poly, edge_points

    if shape == "Tetrahedron":

        ## Running the code for a tetrahedral polygon
        ## Create a list with the coordinates of fictuous atoms at the vertices of the tetrahedron
        poly_xyz = [["Pb",   -54.8550,    -39.8750,    -7.3850], ["Pb",   14.7850,    -36.7050,    12.4600],
                    ["Pb",   -13.1950,   29.2800,    1.6150], ["Pb",   -13.4100,   -10.3950,    -59.0400]]

#        with open('tetrahedron.xyz') as polyhedron:
#            poly_xyz = [item.split() for item in polyhedron.readlines()]

        atom_numbers = len(poly_xyz)
        poly_xyz_proc = [[i[0], float(i[1]), float(i[2]), float(i[3])] for i in poly_xyz]

        ## Generate a new polygon making sure that its com coincides with the origin
        edge_points = []
        new_poly = []
        for item in center_of_mass(poly_xyz_proc)[1]:
            new_poly.append([item[1], item[2], item[3]])
        for i, j in itertools.combinations(list(range(atom_numbers)), 2):
            edge_points.append(mid_point(new_poly[i], new_poly[j]))
            
    if shape == "Trigonal Bi":
        """This function creates the coordinates for a trigonal bipyramid, which follows the [3 + 2] symmetry,
        meaning that 3 linkers are positioned on the vertices of the triangle, whereas the 2 building blocks
        sit on the bipyramid vertices.
        """

        l = 50
        poly_xyz = [["Pb", 0.0,      0.0,     0.0], ["Pb", l,      0.0,     0.0], ["Pb", l/2,     (m.sqrt(3)*l)/2,        0.0]]

        edge_points = []

        for item in center_of_mass(poly_xyz)[1]:
            edge_points.append([item[1], item[2], item[3]])

#        center = center_of_mass(edge_points)[0]
#        print(center)

        new_poly_a = [0.0, 0.0, (m.sqrt(3)*l)/2]
        new_poly_b = [0.0, 0.0, -(m.sqrt(3)*l)/2]
        new_poly = [new_poly_a, new_poly_b]

    if shape == "Cube":

        """
        Function that defines the coordinates for the points of a cube. Here the bb is placed on the vertices and
        the linkers on the edges mid points.
        """
        x = 30
        cube_coords = [["Pb", 0.0, 0.0, 0.0], ["Pb", x, 0.0, 0.0], ["Pb", x, x, 0.0], ["Pb", 0.0, x, 0.0],
        ["Pb", 0.0, 0.0, x], ["Pb", x, 0.0, x], ["Pb", x, x, x], ["Pb", 0.0, x, x]]

        atom_numbers = len(cube_coords)

        new_poly = []

        for item in center_of_mass(cube_coords)[1]:
            new_poly.append([item[1], item[2], item[3]])

        d_lst = []

        for i, j in itertools.combinations(new_poly, 2):
            d_lst.append([new_poly.index(i), new_poly.index(j), distance.euclidean(i, j)])

        min_lst = sorted(d_lst, key = itemgetter(2))
        min_lst = min_lst[:12]

        edge_points = []
        for item in min_lst:
            mid_xyz = mid_point(new_poly[int(item[0])], new_poly[int(item[1])])
            edge_points.append(mid_point(new_poly[int(item[0])], new_poly[int(item[1])]))

    if shape == "Octahedron":
        
        """ 
        Funtion that defines the coordinates for the vertices and edges' mid points for an octahedron.
        """        

        x = 50

        oct_coords = [["Pb", 0.0, 0.0, 0.0], ["Pb", x, 0.0, 0.0], ["Pb", 0.0, x, 0.0], ["Pb", x, x, 0.0],

        ["Pb", x/2, x/2, (m.sqrt(3)*x)/2], ["Pb", x/2, x/2, -(m.sqrt(3)*x)/2]]

        atom_numbers = len(oct_coords)

        new_poly = []

        for item in center_of_mass(oct_coords)[1]:
            new_poly.append([item[1], item[2], item[3]])

        d_lst = []

        for i, j in itertools.combinations(new_poly, 2):
            d_lst.append([new_poly.index(i), new_poly.index(j), distance.euclidean(i, j)])

        min_lst = sorted(d_lst, key = itemgetter(2))
        min_lst = min_lst[:12]

        edge_points = []

        for item in min_lst:
            mid_xyz = mid_point(new_poly[int(item[0])], new_poly[int(item[1])])
            edge_points.append(mid_point(new_poly[int(item[0])], new_poly[int(item[1])]))


    if shape == "Dodecahedron":

        #These are the coordinates of vertices of the dodecahedron where x varies the size of the edge length

        x = 20
        phi = (1  + m.sqrt(5))/2
        dod_coord = [
                   [x*phi,0.0,x/phi], [x*-phi,0.0,x/phi], [x*-phi,0.0,x/-phi],    
                   [x*phi,0.0,x/-phi], [x/phi,x*phi,0.0], [x/phi,x*-phi,0.0],
                   [x/-phi,x*-phi,0.0], [x/-phi,x*phi,0.0], [0.0,x/phi,x*phi],   
                   [0.0,x/phi,x*-phi], [0.0,x/-phi,x*-phi], [0.0,x/-phi,x*phi],  
                   [x,x,x], [x,-x,x], [-x,-x,x], [-x,x,x], [-x,x,-x], [x,x,-x], 
                   [x,-x,-x], [-x,-x,-x]
                   ]

        poly_xyz_proc = [["Pd", float(i[0]), float(i[1]), float(i[2])] for i in dod_coord]

        # Make sure that the polygon has the com in the origin
        new_poly = []

        for item in center_of_mass(poly_xyz_proc)[1]:
            new_poly.append([item[1], item[2], item[3]])

        # Assigning each vertex to a letter for reference
        A = new_poly[0]
        B = new_poly[1]
        C = new_poly[2]
        D = new_poly[3]    
        E = new_poly[4]         
        F = new_poly[5] 
        G = new_poly[6] 
        H = new_poly[7]
        I = new_poly[8]
        J = new_poly[9]
        K = new_poly[10]
        L = new_poly[11]
        M = new_poly[12]
        N = new_poly[13]
        O = new_poly[14]
        P = new_poly[15]
        Q = new_poly[16]
        R = new_poly[17]
        S = new_poly[18]
        T = new_poly[19]

        # defining a function that calculates the vector from j to i

        def vec(i,j):
            a = np.asarray(i)-np.asarray(j)
            return a

        #calculates the mid point of i and j

        def mid_point2(i,j):
            a = (np.asarray(i)+np.asarray(j))/2
            return a        

        #this calculates the vector between adjacent vertices and hence determines the edge vectors
        AN = vec(A,N)
        AM = vec(A,M)
        AD = vec(A,D)
        BO = vec(B,O)
        BP = vec(B,P)
        BC = vec(B,C)
        CT = vec(C,T)
        CQ = vec(C,Q)
        DS = vec(D,S)
        DR = vec(D,R)
        EM = vec(E,M)
        EH = vec(E,H)
        ER = vec(E,R)
        FG = vec(F,G)
        FS = vec(F,S)
        FN = vec(F,N)
        GO = vec(G,O)
        GT = vec(G,T)
        HP = vec(H,P)
        HQ = vec(H,Q)
        IL = vec(I,L)
        IM = vec(I,M)
        IP = vec(I,P)
        JK = vec(J,K)
        JR = vec(J,R)
        JQ = vec(J,Q)
        KS = vec(K,S)
        KT = vec(K,T)
        LO = vec(L,O)
        LN = vec(L,N)


        #this is a list of the edge vectors
        edge_vec=[AN, AM, AD, BO, BP, BC, CT, CQ, DS, DR, EM, EH, ER, FG, FS,
        FN, GO, GT, HP, HQ, IL, IM, IP, JK, JR, JQ, KS, KT, LO, LN]

        edge_vec_final=[]

        for i in edge_vec:
            edge_vec_final.append(i.tolist())

        #arbitrarily assigning the mid point of each vertex(below) as the point where the linkers will be translated to
        AN_mid=mid_point2(A,N)
        AM_mid=mid_point2(A,M)
        AD_mid=mid_point2(A,D)
        BO_mid=mid_point2(B,O)
        BP_mid=mid_point2(B,P)
        BC_mid=mid_point2(B,C)
        CT_mid=mid_point2(C,T)
        CQ_mid=mid_point2(C,Q)
        DS_mid=mid_point2(D,S)
        DR_mid=mid_point2(D,R)
        EM_mid=mid_point2(E,M)
        EH_mid=mid_point2(E,H)
        ER_mid=mid_point2(E,R)
        FG_mid=mid_point2(F,G)
        FS_mid=mid_point2(F,S)
        FN_mid=mid_point2(F,N)
        GO_mid=mid_point2(G,O)
        GT_mid=mid_point2(G,T)
        HP_mid=mid_point2(H,P)
        HQ_mid=mid_point2(H,Q)
        IL_mid=mid_point2(I,L)
        IM_mid=mid_point2(I,M)
        IP_mid=mid_point2(I,P)
        JK_mid=mid_point2(J,K)
        JR_mid=mid_point2(J,R)
        JQ_mid=mid_point2(J,Q)
        KS_mid=mid_point2(K,S)
        KT_mid=mid_point2(K,T)
        LO_mid=mid_point2(L,O)
        LN_mid=mid_point2(L,N)

        #complilation of the midpoints of the edges
        edge_points = [ AN_mid, AM_mid, AD_mid, BO_mid, BP_mid, BC_mid,
        CT_mid, CQ_mid, DS_mid, DR_mid, EM_mid, EH_mid, ER_mid, FG_mid,
        FS_mid, FN_mid, GO_mid, GT_mid, HP_mid, HQ_mid, IL_mid, IM_mid,
        IP_mid, JK_mid, JR_mid, JQ_mid, KS_mid, KT_mid, LO_mid, LN_mid]

    return(new_poly, edge_points)


def BuildCage(shape, output_structure_file_name, bb_smile, bb_sub_group_num, lk_smile = "", lk_sub_group_num = 0):
    
    """
    This function reads the coordinates for the original bb and lk precursors, modifies their functionalities and 
    substitutes them with heavy atoms. The modified bb and lk are arranged in space following the polygon's symmetry
    (bb --> vertices, lk --> edge midpoints) and a new mol file is created, where however bb and lk are unlinked. 
    The heavy atoms are accordinly linked and then substituted with the original atoms. The cage is Ready to roll!
    """

    """
    Run the function ChangeFunctionalGroupAtom, which takes the smiles of an aldehyde as building block (3 functional groups), and an amine as linker
    (2 functional groups) and substitutes the functional groups with heavy atoms.
    """

    # Modify the name of the folder GA_output to the name you want 
    with open("bb_new.mol", "w") as bb_input:    
        bb_input.write(grf.ChangeFunctionalGroupAtom(bb_smile, bb_sub_group_num))
    bb_input.close()

    # Modify the name of the folder GA_output to the name you want
    ## Read in the coordinates for the building block and for the linker
    with open("bb_new.mol", "r") as bb:
        bb_raw = bb.readlines()
        bb_coords = []
        bb_bonds = []
        bb_heavy_atom_count = 0
        for line in bb_raw:
            columns = line.split()
            if ("Y" in line or "Zr" in line or "Nb" in line or "Mo" in line or "Tc" in line or "Ru" in line or "Rh" in line or "Pd" in line):
                bb_heavy_atom_count += 1
                bb_heavy_atom = columns[3]
            if len(columns) == 16:
                bb_coords.append([str(columns[3]), float(columns[0]), float(columns[1]), float(columns[2])])
            if len(columns) == 4:
                bb_bonds.append([int(columns[0]), int(columns[1]), int(columns[2])])
        bb_new = center_of_mass(bb_coords)[1]
    bb.close()

    """
    Generate as many building block molecules as the number of vertices of the polygon and translate their com to
    the polygon's vertices
    """

    bb_lst = []
    bb_bonds_lst = []

    ## bb_n corresponds to the number of atoms in bb
    bb_n = len(bb_coords)
    
    ## bb_n_bonds corresponds to the number of bonds in the bb
    bb_n_bonds = len(bb_bonds)
    
    ## bb_num corresponds to the number of bb in the final cage
    bb_num = len(polygon(shape)[0])

    for i in range(len(polygon(shape)[0])):
        for item in bb_new:
            bb_x = item[1] - polygon(shape)[0][i][0]
            bb_y = item[2] - polygon(shape)[0][i][1]
            bb_z = item[3] - polygon(shape)[0][i][2]
            bb_lst.append([float(bb_x), float(bb_y), float(bb_z), str(item[0]), ("0  " * 12)])
        for item in bb_bonds:
            bb_bonds_lst.append([int(item[0] + bb_n * i), int(item[1] + bb_n * i), item[2], str("0  " * 4)])

    # Check if lk smile is empty else operate normally on the link molecules

    if lk_smile == "" and lk_sub_group_num == "":
        lk_lst = []
        lk_bonds_lst = []

    else:
        # Modify the name of the folder GA_output to the name you want
        with open("lk_new.mol", "w") as lk_input:
            lk_input.write(grf.ChangeFunctionalGroupAtom(lk_smile, lk_sub_group_num))
        lk_input.close()

        # Modify the name of the folder GA_output to the name you want
        with open("lk_new.mol") as lk:
            lk_raw = lk.readlines()
            lk_coords = []
            lk_bonds = []
            lk_heavy_atom_count = 0
            for line in lk_raw:
                columns = line.split()
                if ("Y" in line or "Zr" in line or "Nb" in line or "Mo" in line or "Tc" in line or "Ru" in line or "Rh" in line):
                    lk_heavy_atom_count += 1
                    lk_heavy_atom = columns[3]
                if len(columns) == 16:
                    lk_coords.append([str(columns[3]), float(columns[0]), float(columns[1]), float(columns[2])])
                if len(columns) == 4:
                    lk_bonds.append([int(columns[0]), int(columns[1]), int(columns[2])])
            lk_new = center_of_mass(lk_coords)[1]
        lk.close()

        """
        Generate as many linkers molecules as the number of edges midpoints of the polygon and translate their com to
        the edges
        """

        lk_lst = []
        lk_bonds_lst = []

        lk_n = len(lk_coords)
        lk_n_bonds = len(bb_bonds)
        
        lk_num = len(polygon(shape)[1])
        for i in range(len(polygon(shape)[1])):
            for item in lk_new:
                lk_x = item[1] - polygon(shape)[1][i][0]
                lk_y = item[2] - polygon(shape)[1][i][1]
                lk_z = item[3] - polygon(shape)[1][i][2]
                lk_lst.append([float(lk_x), float(lk_y), float(lk_z), str(item[0]), ("0  " * 12)])
            for item in lk_bonds:
                lk_bonds_lst.append([(item[0] + len(bb_lst) + lk_n * i), int(item[1] + len(bb_lst) + lk_n * i), item[2], str("0  " * 4)])


    ### Join the bb and lk in the same list
    final_coords_lst = bb_lst + lk_lst
    final_bonds_lst = bb_bonds_lst + lk_bonds_lst

    # Modify the name of the folder GA_output to the name you want
    ### Write the coord and bond lists to a formatted mol V3000 file "input_atoms.mol"
    with open("input_atoms.mol", "w+") as f:
        f.write("\n     RDKit          3D\n\n")
        f.write("  0  0  0  0  0  0  0  0  0  0999 V3000\n")
        f.write("M  V30 BEGIN CTAB\n")
        f.write("M  V30 COUNTS {:<3} {:<3} 0 0 0\n".format(len(final_coords_lst), len(final_bonds_lst)))
        f.write("M  V30 BEGIN ATOM\n")
        for item in final_coords_lst:
            f.write("M  V30 {0:<} {1:<} {2:<} {3:<} {4:<} 0\n".format((final_coords_lst.index(item) + 1), item[3], item[0], item[1], item[2]))
        f.write("M  V30 END ATOM\n")
        f.write("M  V30 BEGIN BOND\n")
        for item in final_bonds_lst:
            f.write("M  V30 {0:<} {1:<} {2:<} {3:<}\n".format((final_bonds_lst.index(item) + 1), item[2], item[0], item[1]))
        f.write("M  V30 END BOND\n")
        f.write("M  V30 END CTAB\n")
        f.write("M  END")


    """
    Run the function addition contained in the SubstituterDeAromatic library, which links the substituted bb and lk that
    are contained in the file input_atoms, but as sepatare molecule. The function takes as arguments:
    1) file generated by this script "input_atoms.mol", which is printed as a V2000 mol and should be in the same 
    folder where this script is executed.
    2) The heavy atom of the first molecule 
    3) Number of heavy atoms in the first molecule
    4) The heavy atom of the second molecule
    5) Number of heavy atoms in the second molecule
    ****this should be automated!
    """

    

    # Modify the name of the folder GA_output to the name you want
    output_file_name= "" + output_structure_file_name 
    heavy_atom_file_name = output_file_name[:-4] + "HEAVY_ATOMS.mol"

    

    # Modify the name of the folder GA_output to the name you want
    if lk_sub_group_num == 0:
        grf.AdditionSame("input_atoms.mol", bb_heavy_atom, bb_heavy_atom_count, heavy_atom_file_name)

    else:
        grf.AdditionDifferent("input_atoms.mol", bb_heavy_atom, bb_heavy_atom_count, 
                 lk_heavy_atom, lk_heavy_atom_count, heavy_atom_file_name)
    final_sub(heavy_atom_file_name, output_file_name)   

    return heavy_atom_file_name, bb_num, bb_heavy_atom_count, lk_num, lk_heavy_atom_count



# Run the Codes for specific bb and lk on Dodecahedral symmetry

#BuildCage("Dodecahedron", "testnitroso.mol", "CC(C1=CC=C(C=C1)N=O)(C1=CC=C(C=C1)N=O)C1=CC=C(C=C1)N=O", 8)
#os.chdir("/home/lukas/")
#BuildCage("Tetrahedron", "test2.mol", "c1(C(=O)[H])cc(C(=O)[H])cc(C(=O)[H])c1", 2, "NCCC1=CC=CC=C1CN", 1)
#BuildCage("Tetrahedron", "test3.mol", "O=CC1CCCC(CCCC(CCC1)C=O)C=O", 2, "[H]N([H])C1CCCCC1N([H])[H]", 1)
#BuildCage("Tetrahedron", "test4.mol", "O=CC1CCCC(CCCC(CCC1)C=O)C=O", 2, "NCCC1=CC=CC=C1CN", 1)

# Convert .mol file into .mae with the /opt/schrodinger/suites2015-3/utilities/structconvert to open with maestro





    

