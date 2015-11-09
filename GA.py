import GA_functions as ga_f
import polygon as pg
import os as os
from rdkit.Chem import AllChem as ac
from rdkit import Chem as chem
import random as random
import GA_rdkit_functions as grf
import time as time
import shutil as shutil
import itertools
import numpy as np
import GA_plotter as gap
import copy as cp

def CreateInitialPopulation(population_size, bb_folder, bb_grp_sub_num,
                            link_folder, link_grp_sub_num, shape):
    
    present_directory = os.getcwd()
    
    """Make initial population."""
    population = []    
    
    #create initial Cages
    
    list_of_bb_files = os.listdir(bb_folder)
    list_of_link_files = os.listdir(link_folder)       
    
    iter_count = 0
    while len(population) < population_size:
        structure_file_location = present_directory + "/" + str(iter_count) + ".mol"
        bb_file_name = random.choice(list_of_bb_files) 
        if ".mol" not in bb_file_name:
            continue
        bb_smiles = chem.MolFromMolFile(bb_folder+bb_file_name)
        bb_smiles = chem.MolToSmiles(bb_smiles)
        bb_smiles_prist = bb_smiles
        link_file_name = random.choice(list_of_link_files)
        if ".mol" not in link_file_name:
            continue
        link_smiles = chem.MolFromMolFile(link_folder+link_file_name)
        link_smiles = chem.MolToSmiles(link_smiles)
        lk_smiles_prist = link_smiles
 
        if ("+" in bb_smiles or "+" in link_smiles or "-" in bb_smiles or 
            "-" in link_smiles or "." in bb_smiles or "." in link_smiles):
                continue

        if shape == "Random_3":
            list_of_shapes = ["Tetrahedron", "Dodecahedron", "Trigonal Bi", "Cube"]
            chosen_shape = random.choice(list_of_shapes)
                    
        else:
            chosen_shape = shape
        
        try:
            print("shape before polygon", shape, chosen_shape)            
            heavy_structure_file_location, bb_n, bb_sub_atoms, lk_n, lk_sub_atoms = pg.BuildCage(chosen_shape, structure_file_location, bb_smiles, 
                                                                                                 bb_grp_sub_num, link_smiles, link_grp_sub_num)
                                                                                                
        except:
            print("EXCEPTION!!!!!!!!")
            continue
        
        bb_file_location = present_directory + "/bb_new.mol"
        link_file_location = present_directory + "/lk_new.mol"        
        bb_smiles = chem.MolFromMolFile(bb_file_location)
        bb_smiles = chem.MolToSmiles(bb_smiles)
        link_smiles = chem.MolFromMolFile(link_file_location)
        link_smiles = chem.MolToSmiles(link_smiles)
        b = open("new.log", "a")
        b.write(bb_smiles + "\n")
        b.write(link_smiles + "\n")        
        b.close()        
        individual = ga_f.Cage(bb_smiles_prist, bb_smiles, lk_smiles_prist, link_smiles,
                               structure_file_location, heavy_structure_file_location, 
                               "initial", "initial", chosen_shape)
                
        individual.origin = "initial"
        individual.origin_gen = 0       
        population.append(individual)
        iter_count += 1       
        
    return population, bb_n, bb_sub_atoms, lk_n, lk_sub_atoms
    
def MatePopulation(population, cur_ga_gen_dir, gen_counter, bb_grp_sub_num, link_grp_sub_num):
    offspring_population = []

    print("\n\n******** this is the size of the population chosen for mating: ", len(population), "\n\n")
    for i, j in itertools.combinations(range(len(population)), 2):

        individual1 = population[i]
        individual2 = population[j]

       # while True:
        #    ind_bb_same = False
         #   ind_lk_same = False
            
       #     if individual1.bb_smiles == individual2.bb_smiles:
       #         print("  BB SMILES: \n\n {0}    {1}".format(individual1.bb_smiles, individual2.bb_smiles))
       #         ind_bb_same = True
            
       #     if individual1.link_smiles == individual2.link_smiles:
       #         print("  LINK SMILES: \n\n {0}    {1}".format(individual1.link_smiles, individual2.link_smiles))
       #         ind_lk_same = True
                
       #     if ind_lk_same == True and ind_bb_same == True:
       #         continue
            
       #     break
        
        offspring_name = cur_ga_gen_dir + "/gen_" + str(gen_counter) + "_mating_" + str(i) + "_" + str(j) + ".mol"
        try:
            offspring1, offspring2 = ga_f.CageMating(individual1, individual2, bb_grp_sub_num, link_grp_sub_num, offspring_name)
        except:
            continue

        offspring_population.extend([offspring1, offspring2])
        offspring1.origin_gen = gen_counter
        offspring2.origin_gen = gen_counter
        
        print("\n\n ********** OFFSPRING1 SHAPE =  ", individual1.shape, "  ", individual2.shape, "  ", offspring1.shape, offspring1.mol_file_location)
        print(" ********** OFFSPRING2 SHAPE =  ", individual1.shape, "  ", individual2.shape, "  ", offspring2.shape, offspring2.mol_file_location, "\n\n")

    return offspring_population

       
def MutatePopulation(population, number_of_mutations, bb_folder, bb_grp_sub_num, link_folder, link_grp_sub_num, cur_ga_gen_dir, gen_counter):
    mutant_population = []    

    mutation_counter = 0
#    mutation_functions = [ga_f.FragmentMutation, ga_f.TopologyMutation, ga_f.TieMutation]
    mutation_functions = [ga_f.FragmentMutation, ga_f.TieMutation]
    while mutation_counter < number_of_mutations:                              
        individual_for_mutation = random.choice(population)
        individual_for_mutation = cp.deepcopy(individual_for_mutation)
        mutation_source_folder, grp_sub_num = random.choice([(bb_folder, bb_grp_sub_num), 
                                                             (link_folder, link_grp_sub_num)])            
        
        mutation_function = np.random.choice(a=mutation_functions , p=[0.9, 0.1])         
        mutant_name = cur_ga_gen_dir + "/gen_" + str(gen_counter) + "_mutation_" + str(mutation_counter) + ".mol"

        try:
            if mutation_function is ga_f.FragmentMutation:
                mutants = mutation_function(individual_for_mutation, mutant_name, mutation_source_folder, grp_sub_num, bb_folder)
                
            elif mutation_function is ga_f.TopologyMutation:
                mutants = mutation_function(individual_for_mutation, mutant_name, bb_grp_sub_num, link_grp_sub_num)
                
            elif mutation_function is ga_f.TieMutation:
                mutants = mutation_function(individual_for_mutation, mutant_name)
                
        except:
            print("PROBLEM WITH SOME MUTATION")            
            continue                
                
        for mutant in mutants:
            mutant_population.append(mutant)
            mutant.origin_gen = gen_counter            
        
        mutation_counter += 1    
    
    return mutant_population

def CopyPopulationToNextGenFolder(population, cur_ga_gen_dir):
    
    for individual in population:
        if individual.mol_file_location.split("/")[-2] != cur_ga_gen_dir.split("/")[-1]:               
            shutil.copy(individual.mol_file_location, cur_ga_gen_dir)
    
    return 1

def WriteGenToOutputFile(output_file_name, population, gen_number):
    
    fitness_list = [x.fitness_value for x in population]
    max_pop_fitness = max(fitness_list)
    min_pop_fitness = min(fitness_list)
    avg_pop_fitness = np.mean(fitness_list)
    sorted_population = sorted(population, key = lambda x: x.fitness_value, reverse=True)
    
    
    b = open(output_file_name, "a")
    b.write("\n\n")
    b.write("  ********************************************************************\n\n")
    b.write("  The size of the chosen population is: " + str(len(population)) + "\n")
    b.write("  GENERATION " + str(gen_number) + "\n")
    b.write("  Max " + str(round(max_pop_fitness, 2)) + "\n")
    b.write("  Min " + str(round(min_pop_fitness, 2)) + "\n")
    b.write("  Avg " + str(round(avg_pop_fitness, 2)) + "\n\n")
    
    for x in sorted_population:
        try:
            b.write("  {0:<40s}   Shape = {1:12s} Val = {2:>6.2f}  Window diff = {3:>6.2f}  Window ratio = {4:>6.2f}  {5:>10.3f}\n".format(x.mol_file_location.split("/")[-1], x.shape, x.val, x.window_diff, x.window_ratio, x.fitness_value))
#            b.write("{0:<40s}   Shape = {1:12s} Val = {2:>6.2f}  Window diff = {3:>6.2f}  {4:>10.2f}\n".format(x.mol_file_location.split("/")[-1], x.shape, x.val, x.window_diff, x.fitness_value))
        except:
            b.write("  {0:<40s}   Shape = {1:12s} Val = ------  Window diff = ------ Window ratio = ------  {2:>10.2f}\n\n".format(x.mol_file_location.split("/")[-1], x.shape, x.fitness_value))
    
    b.write("  ********************************************************************")    
    b.close()    
    
    return 1

def WriteEtpToOutputFile(output_file_name, population, tracking_list):
    
    b = open(output_file_name, "a")
    b.write("EPP_END\n")
    b.write("index\tfitness\tparent1\tparent2\torigin\torigin_gen\n")
    b.write("ETP_START\n")    
    
    for x in population:
        b.write("new_ind" + "\n")        
        etp_data = ga_f.GetFamilyTree(x, tracking_list)
        for y in etp_data:
            b.write(str(y) + "\n")
            
    b.write("ETP_END")    
    b.close()    
    return 1



def GeneticAlgorithm(population_size, number_of_generations, number_of_matings, 
                     number_of_mutations, bb_folder, bb_grp_sub_num, link_folder,
                     link_grp_sub_num, processors, desired_size, shape):
      
    ga_py_dir = os.getcwd()
    ga_output_dir = ga_py_dir + "/GA_output/"
    ga_initial_gen_dir = ga_output_dir + "gen_0"
    
    if "GA_output" in os.listdir(ga_py_dir):
        shutil.move(ga_output_dir, ga_py_dir + "/GA_output" + str(time.strftime("%c").replace(" ", "_")))    
    
    os.mkdir(ga_output_dir)    
    os.mkdir(ga_initial_gen_dir)
    
    os.chdir(ga_initial_gen_dir)       
    
    population, bb_n, bb_sub_atoms, lk_n, lk_sub_atoms = CreateInitialPopulation(population_size, bb_folder, bb_grp_sub_num,
                                                                                 link_folder, link_grp_sub_num, shape)
    ga_f.RelaxPopulation(population, processors)
    ga_f.CalculatePopulationFitness(population, desired_size, bb_n, bb_sub_atoms, lk_n)
    
    all_individuals_list = []
    all_individuals_list_index = 0     
    ga_f.AddPopulationToTrackingList(population, all_individuals_list, all_individuals_list_index)

          
    ga_gen_dir = ga_output_dir + "gen_"
    gen_counter = 1
    while gen_counter <= number_of_generations:
        cur_ga_gen_dir = ga_gen_dir + str(gen_counter)
        os.mkdir(cur_ga_gen_dir)
        os.chdir(cur_ga_gen_dir)
        
        selected_pop = population
        selected_pop = list(map(cp.deepcopy, selected_pop))        
        
        new_members = []
        
        refined_pop = ga_f.StochasticSampling(selected_pop, number_of_matings)        
        
        offspring_population = MatePopulation(refined_pop, cur_ga_gen_dir, gen_counter, bb_grp_sub_num, link_grp_sub_num)
        new_members.extend(offspring_population)
        
        print("\n\n\nSIZE OF NEWMEMBERS BEFORE MUTATION: ", len(new_members))        
        
        mutant_population = MutatePopulation(population, number_of_mutations, bb_folder, bb_grp_sub_num, link_folder, 
                                             link_grp_sub_num, cur_ga_gen_dir, gen_counter)
        
        print("SIZE OF MUTANT POPULATION: ", len(mutant_population))         

        new_members.extend(mutant_population)
        
        print("SIZE OF NEWMEMBERS AFTER MUTATION: ", len(new_members), "\n\n\n")
        
        ga_f.RelaxPopulation(new_members, processors)
        ga_f.CalculatePopulationFitness(new_members, desired_size, bb_n, bb_sub_atoms, lk_n)                 
        ga_f.AddPopulationToTrackingList(new_members, all_individuals_list, all_individuals_list_index)
        
        population.extend(new_members)
        
                
        for individual in population:
            print(individual.mol_file_location, "SHAPE = ", individual.shape, "  ", individual.fitness_value)
            
        print("SIZE OF POPULATION AFTER MUTATION: ", len(population), "\n\n\n")
        
        
        population = ga_f.SelBest(population, population_size)
        
        CopyPopulationToNextGenFolder(population, cur_ga_gen_dir)
        
        os.chdir(ga_output_dir)
    
        WriteGenToOutputFile("file.log", population, gen_counter)
        
        gen_counter += 1
        
    os.chdir(ga_output_dir)

    WriteEtpToOutputFile("file.log", population, all_individuals_list)
    
    return population
    

pop = GeneticAlgorithm(30, 200, 5, 25, "/Users/Enrico/Dropbox (Imperial College)/GA/db_new/aldehydes3f/", 2,
                       "/Users/Enrico/Dropbox (Imperial College)/GA/db_new/amines2f/", 1, processors = 12, desired_size = 10.32, shape= "Tetrahedron")

#os.chdir("GA_output/")
###gap.PlotETP("file.log")
#gap.PlotEPP("file.log", 200)



