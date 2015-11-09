from matplotlib import pyplot as plt

def PlotEPP(output_file_location, generation_number):
    output_file = open(output_file_location, "r")

    gen_numbers = []    
    max_numbers = []
    min_numbers = []
    avg_numbers = []
    for line in output_file:
        if "gen " in line:
            line = line.split()            
            gen_numbers.append(int(line[1]))
        
        if "max " in line:
            line = line.split()
            max_numbers.append(float(line[1]))
            
        if "min " in line:
            line = line.split()
            min_numbers.append(float(line[1]))
            
        if "avg " in line:
            line = line.split()
            avg_numbers.append(float(line[1]))
            
    output_file.close()
    #plt.plot(gen_numbers, max_numbers)    
    
    graph_figure = plt.figure(1, figsize = (6, 30))
    axes = graph_figure.add_subplot(111)
    title = axes.set_title("Evolutionary Progress Plot\n") 
    xlabel = axes.set_xlabel("Generation Number")
    ylabel = axes.set_ylabel("Fitness Value")    
    max_line, = axes.plot(gen_numbers, max_numbers, '-x', label="Maximum")    
    min_line, = axes.plot(gen_numbers, min_numbers, '-x', label="Minimum")
    avg_line, = axes.plot(gen_numbers, avg_numbers, '-x', label="Mean")
       
    title.set_weight('bold')
    title.set_fontsize(15)
    """
    axes.set_xlim(1, generation_number+1)
    axes.xaxis.set_ticks(range(1, generation_number+1))    
    axes.xaxis.set_ticks_position('bottom')
    axes.yaxis.set_ticks_position('left')
    
    for axis in ['left','bottom']:
        axes.spines[axis].set_linewidth(2)
    
    xlabel.set_weight('bold')
    xlabel.set_fontsize(12)
    ylabel.set_weight('bold')
    ylabel.set_fontsize(12)
    
    plt.tick_params(axis='x', which='major', labelsize=12)
    plt.tick_params(axis='y', which='major', labelsize=12)    
    plt.tick_params(axis='both', which='minor', labelsize=8)

    
    max_line.set_lw(2)
    max_line.set_markersize(10)
    max_line.set_markeredgewidth(1.5)
    min_line.set_lw(2)
    min_line.set_markersize(10)
    min_line.set_markeredgewidth(1.5)
    avg_line.set_lw(2)
    avg_line.set_markersize(10)
    avg_line.set_markeredgewidth(1.5)
    """
    plt.legend(bbox_to_anchor=(1, 0, 0, 0.25))
    plt.show()
    graph_figure.savefig("EPP.png", dpi=300)
    plt.close()
    
def PlotETP(output_file_location):
    output_file = open(output_file_location, "r")
    
    point_positions = []    
    raw_data = []    
    
    for line in output_file:
        if "(" in line: 
            line = line.split("'")
            
            line.remove("(")
            while ", " in line:   
                line.remove(", ")
            line.remove(")\n")
            
            if line not in raw_data:
                raw_data.append(line)
                print(line)
                
    for data in raw_data:
        point_positions.append([int(data[-1]), float(data[1])])
    
    for point in point_positions:
        plt.scatter(*point, marker='x', color='black')
        
    for line in raw_data:
        if line[2] == 'initial':
            continue
        if line[3] == 'mutant':
            parent = line[2]
            
            for line2 in raw_data:
                if line2[0] == parent:
                    plt.plot([int(line[-1]),int(line2[-1])],[float(line[1]),float(line2[1])], color='green')
                    
        if line[3] not in ['initial','mutant']:
            parent1 = line[2]
            parent2 = line[3]
            
            for line2 in raw_data:
                if line2[0] == parent1 or line2[0] == parent2:
                    print(line[0], line2[0])
                    plt.plot([int(line[-1]),int(line2[-1])],[float(line[1]),float(line2[1])], color='red')
                
        
    
       
            
        
        
        
    output_file.close()
 





















   