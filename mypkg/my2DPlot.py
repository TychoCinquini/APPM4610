import matplotlib.pyplot as plt
import numpy as np
import math

class my2DPlot:
    def __init__(self,f,a,b):
        # Creates a pyplot object of function f
        #
        # Inputs:
        #   f : function to graph
        #   a : left endpoint of function domain
        #   b : right endpoint of function domain
        # Outputs: None

        # Create domain vector with a resolution of 0.01
        self.x = np.arange(a,b,0.01) 
        # Create vector of function value along entire domain
        y = f(self.x)
        # Create plot object and plot function
        self.p = plt.plot(self.x,y)

    def show(self):
        # Displays the plot
        #
        # Inputs: None
        # Outputs: None

        # Display plot
        plt.show()

    def dotted(self):
        # Sets the linestyle of the most recent plot to be dotted
        # 
        # Inputs:None
        # Outputs: None

        # Set the most recent line to be dotted
        self.p[-1].set_linestyle('dotted')

    def labels(self,x,y):
        # Adds labels to the plot
        #
        # Inputs:
        #   x : x-axis label
        #   y : y-axis label
        # Outputs: None

        # Label the x-axis
        plt.xlabel(x)
        # Label the y-axis
        plt.ylabel(y)

    def addPlot(self,f):
        # Adds a new line to the plot
        #
        # Inputs:
        #   f : function describing the line to add to the plot
        # Outputs: None

        # Add the new line to the plot
        self.p = self.p + plt.plot(self.x,f(self.x))

    def color(self,colorName):
        # Changes the color of the most recent plot
        #
        # Inputs:
        #   colorName : new color of the plot
        # Outputs: None

        # Set line color to desired color
        self.p[-1].set_color(colorName)

    def logx(self):
        # Sets the x-axis to have a logarithmic scale
        #
        # Inputs: None
        # Outputs: None
        
        # Set the x-axis scale to be logarithmic
        plt.xscale("log")

    def logy(self):
        # Sets the y-axis to have a logarithmic scale
        #
        # Inputs: None
        # Outputs: None
        
        # Set the x-axis scale to be logarithmic
        plt.yscale("log")

    def save(self,filename):
        # Saves the plot to the current directory
        #
        # Inputs:
        #   filename : desired filename
        # Outputs: None

        # Save the plot
        plt.savefig(filename)