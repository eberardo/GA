import tkinter as tk
import GA as ga



def run():
    pop_size = int(population_size_entry.get())
    num_of_gen = int(number_of_generations_entry.get())
    num_of_mat = int(number_of_matings_entry.get())
    num_of_mut = int(number_of_mutations_entry.get())
    bb_fold = bb_folder_entry.get()
    b_g_s_n = int(bb_grp_sub_num_var.get())
    lk_fold = link_folder_entry.get()
    l_g_s_n = int(lk_grp_sub_num_var.get())
    prcsrs = int(processors_entry.get())
    


    return ga.GeneticAlgorithm(population_size=pop_size, number_of_generations=num_of_gen,
                               number_of_matings=num_of_mat, number_of_mutations=num_of_mut,
                               bb_folder=bb_fold, bb_grp_sub_num=b_g_s_n, 
                               link_folder=lk_fold, link_grp_sub_num=l_g_s_n,
                               processors=prcsrs)

main = tk.Tk()

population_size_label = tk.Label(main, text="Population Size")
population_size_label.grid(row=2, column=0)

population_size_entry = tk.Entry(main)
population_size_entry.grid(row=2, column=1)

number_of_generations_label = tk.Label(main, text="Number of Generations")
number_of_generations_label.grid(row=3, column=0)

number_of_generations_entry = tk.Entry(main)
number_of_generations_entry.grid(row=3, column=1)

number_of_matings_label = tk.Label(main, text="Number of Matings")
number_of_matings_label.grid(row = 4, column=0)

number_of_matings_entry = tk.Entry(main)
number_of_matings_entry.grid(row=4, column=1)

number_of_mutations_label = tk.Label(main, text="Number of Mutations")
number_of_mutations_label.grid(row=5, column=0)

number_of_mutations_entry = tk.Entry(main)
number_of_mutations_entry.grid(row=5, column=1)

bb_folder_label = tk.Label(main, text="Building Block Database Path")
bb_folder_label.grid(row=0, column=0)

bb_folder_entry = tk.Entry(main)
bb_folder_entry.grid(row=0, column=1)

link_folder_label = tk.Label(main, text="Linker Database Path")
link_folder_label.grid(row=1, column=0)

link_folder_entry = tk.Entry(main)
link_folder_entry.grid(row=1, column=1)

bb_grp_sub_num_label = tk.Label(main, text="Building Block Functional Group")
bb_grp_sub_num_label.grid(row=9, column=0)

bb_grp_sub_num_var = tk.IntVar()
bb_grp_sub_num_op1 = tk.Radiobutton(main, variable = bb_grp_sub_num_var, value=1, text="[N]([H])[H]")
bb_grp_sub_num_op1.grid(row=10, column=0)

bb_grp_sub_num_op2 = tk.Radiobutton(main, variable = bb_grp_sub_num_var, value=2, text="C(=O)[H]")
bb_grp_sub_num_op2.grid(row=11, column=0)

bb_grp_sub_num_op3 = tk.Radiobutton(main, variable = bb_grp_sub_num_var, value=3, text="C(=O)O[H]")
bb_grp_sub_num_op3.grid(row=12, column=0)

bb_grp_sub_num_op4 = tk.Radiobutton(main, variable = bb_grp_sub_num_var, value=4, text="C(=O)N([H])[H]")
bb_grp_sub_num_op4.grid(row=10, column=1)

bb_grp_sub_num_op5 = tk.Radiobutton(main, variable = bb_grp_sub_num_var, value=5,text="C(=O)S[H]")
bb_grp_sub_num_op5.grid(row=11, column=1)

bb_grp_sub_num_op6 = tk.Radiobutton(main, variable = bb_grp_sub_num_var, value=6, text="O[H]")
bb_grp_sub_num_op6.grid(row=12, column=1)

bb_grp_sub_num_op7 = tk.Radiobutton(main, variable = bb_grp_sub_num_var, value=7, text="[S][H]")
bb_grp_sub_num_op7.grid(row=13, column=0)


lk_grp_sub_num_label = tk.Label(main, text="Linker Functional Group")
lk_grp_sub_num_label.grid(row=14, column=0)

lk_grp_sub_num_var = tk.IntVar()
lk_grp_sub_num_op1 = tk.Radiobutton(main, variable=lk_grp_sub_num_var, value=1, text="[N]([H])[H]")
lk_grp_sub_num_op1.grid(row=15, column=0)

lk_grp_sub_num_op2 = tk.Radiobutton(main, variable=lk_grp_sub_num_var, value=2, text="C(=O)[H]")
lk_grp_sub_num_op2.grid(row=16, column=0)

lk_grp_sub_num_op3 = tk.Radiobutton(main, variable=lk_grp_sub_num_var, value=3, text="C(=O)O[H]")
lk_grp_sub_num_op3.grid(row=17, column=0)

lk_grp_sub_num_op4 = tk.Radiobutton(main, variable=lk_grp_sub_num_var, value=4, text="C(=O)N([H])[H]")
lk_grp_sub_num_op4.grid(row=15, column=1)

lk_grp_sub_num_op5 = tk.Radiobutton(main, variable=lk_grp_sub_num_var, value=5,text="C(=O)S[H]")
lk_grp_sub_num_op5.grid(row=16, column=1)

lk_grp_sub_num_op6 = tk.Radiobutton(main, variable=lk_grp_sub_num_var, value=6, text="O[H]")
lk_grp_sub_num_op6.grid(row=17, column=1)

lk_grp_sub_num_op147 = tk.Radiobutton(main, variable=lk_grp_sub_num_var, value=7, text="[S][H]")
lk_grp_sub_num_op147.grid(row=18, column=0)

lk_grp_sub_num_op8 = tk.Radiobutton(main, variable=lk_grp_sub_num_var, value=0, text="None")
lk_grp_sub_num_op8.grid(row=18, column=1)

processors_label = tk.Label(main, text="Number of Processors")
processors_label.grid(row=8, column=0)

processors_entry = tk.Entry(main)
processors_entry.grid(row=8, column=1)

button = tk.Button(main, text="Run", command=run)
button.grid(row=20, column=0)

main.mainloop()
