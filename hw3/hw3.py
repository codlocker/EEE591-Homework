################################################################################
# Created on Fri Aug 24 13:36:53 2018                                          #
#                                                                              #
# @author: olhartin@asu.edu; updates by sdm                                    #
#                                                                              #
# Program to solve resister network with voltage and/or current sources        #
################################################################################

import numpy as np                     # needed for arrays
from numpy.linalg import solve         # needed for matrices
from read_netlist import read_netlist  # supplied function to read the netlist
import comp_constants as COMP          # needed for the common constants

# this is the list structure that we'll use to hold components:
# [ Type, Name, i, j, Value ]

################################################################################
# How large a matrix is needed for netlist? This could have been calculated    #
# at the same time as the netlist was read in but we'll do it here             #
# Input:                                                                       #
#   netlist: list of component lists                                           #
# Outputs:                                                                     #
#   node_cnt: number of nodes in the netlist                                   #
#   volt_cnt: number of voltage sources in the netlist                         #
################################################################################

def get_dimensions(netlist):           # pass in the netlist

    ### EXTRA STUFF HERE!
    nodes = set()
    volt_nodes = 0

    for net in netlist:
        if net[0] == 1:
            volt_nodes += 1
        elif net[0] == 0:
            nodes.add(net[2])
            nodes.add(net[3])

    print(' Nodes ', len(nodes), ' Voltage sources ', volt_nodes)
    return len(nodes), volt_nodes

################################################################################
# Function to stamp the components into the netlist                            #
# Input:                                                                       #
#   y_add:    the admittance matrix                                            #
#   netlist:  list of component lists                                          #
#   currents: the matrix of currents                                           #
#   node_cnt: the number of nodes in the netlist                               #
# Outputs:                                                                     #
#   node_cnt: the number of rows in the admittance matrix                      #
################################################################################

def stamper(y_add, netlist, currents, node_n, node_v):
    # return the total number of rows in the matrix for
    # error checking purposes
    # add 1 for each voltage source...

    voltage_index = 0

    for comp in netlist:                  # for each component...
        #print(' comp ', comp)            # which one are we handling...

        # extract the i,j and fill in the matrix...
        # subtract 1 since node 0 is GND and it isn't included in the matrix
        i = comp[COMP.I] - 1
        j = comp[COMP.J] - 1

        if comp[COMP.TYPE] == COMP.R:           # a resistor
            if i >= 0:
                y_add[i, i] += 1.0 / comp[COMP.VAL]         # add on the diagonal
            if j >= 0:
                y_add[j, j] += 1.0 / comp[COMP.VAL]
            if i >= 0 and j >= 0:                            
                y_add[i, j] -= 1.0 / comp[COMP.VAL]
                y_add[j, i] -= 1.0 / comp[COMP.VAL]
        elif comp[COMP.TYPE] == COMP.IS: # a current source
            # Add on the current column
            if i >= 0:
                currents[i] -= comp[COMP.VAL]
            elif j >= 0:
                currents[j] += comp[COMP.VAL]
        elif comp[COMP.TYPE] == COMP.VS: # a voltage source
            # Add on the newest row created for the matrix and continue for
            # voltage source.
            currents[node_n + voltage_index] = comp[COMP.VAL]
            if i >= 0:
                y_add[node_n + voltage_index][i] = 1
                y_add[i][node_n + voltage_index] = 1
            elif j >= 0:
                y_add[node_n + voltage_index][j] = -1
                y_add[j][node_n + voltage_index] = -1
            voltage_index += 1

    print("Admittance Matrix:\n", admittance_matrix)
    print("Current Matrix:\n", current_matrix)

    return node_n + node_v  # should be same as number of rows!

################################################################################
# Start the main program now...                                                #
################################################################################

# Read the netlist!
netlist = read_netlist()

#EXTRA STUFF HERE!
# Get the dimensions of the matrix formed from the given netlist
node_n, node_v = get_dimensions(netlist=netlist)

# Initialize and set the admittance, voltage and current matrix.
# The size are set as node_n + node_v - 1 because we are removing the Ground Node from
# the calculation.
admittance_matrix = np.zeros((node_n + node_v - 1, node_n + node_v - 1))
voltage_matrix = np.zeros((node_n + node_v - 1, 1))
current_matrix = np.zeros((node_n + node_v - 1, 1))

# print("Shape : Admittance Matrix", admittance_matrix.shape)
# print("Shape : Voltage Matrix", voltage_matrix.shape)
# print("Shape : Current Matrix", current_matrix.shape)

stamper(
    currents=current_matrix,
    netlist=netlist,
    y_add=admittance_matrix,
    node_n=node_n - 1,
    node_v=node_v
)

voltage_matrix = np.matmul(np.linalg.inv(admittance_matrix), current_matrix)

print("Voltage Matrix:\n", np.transpose(voltage_matrix))