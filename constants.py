"""
    Store all the global constants/definitions used in the project here. 
"""
import seaborn as sns 

# constant used in the stopping condition for the sequential 
# test of Balasubramani-Ramdas (2016)
# 1.1 --> approximate control of type-I error 
# 2.0 --> exact control of type-I error 
LIL_THRESHOLD_CONSTANT = 1.1


# constant factor used in the confidence sequence constructed by
# Manole-Ramdas~(2021). The theoretically valid value of the 
# constant is 1.0. In practice,  using a smaller value 
# often leads to a test with a better power without violating 
# the type-I constraint. 
MR_THRESHOLD_CONSTANT = 0.5 


# Reimann's zeta function value at 2 
ZETA_2 = 1.6449 ## pi**2/6

# LinestyleDict: A dictionary with the linestyles to be used for 
# plotting the results of different methods 
LineStylesDict = {
'Betting': 'solid',
'Batch': (0, (5,1)), # densely dashed
'MR': (0, (1, 1)), # densely dotted 
'LC': (0, (3, 1, 1, 1)), # densely dash-dotted 
'DR': (0, (3, 1, 1, 1, 1, 1)), # densely dash-dot-dotted
'HR': (0, (5, 5)), # dashed 
'BR': (0, (5,1)), # densely dashed
'LC19':(0, (5,1)), # densely dashed
'alpha':(0, (5, 10)), # loosely dashed
'Bernoulli-SPRT': (0, (1, 1)) # densely dotted 
}

# ColorsDict : A dictionary with the colors to be used for
# plotting the results of  different methods 
palette = sns.color_palette('tab10', 10)

ColorsDict = {
'Betting':palette[0], 
'Batch':palette[1],
'MR':palette[2],
'DR': palette[3], 
'HR':palette[4],
'LC':palette[5],
'BR':palette[6],
'alpha':palette[7], 
'LC19':palette[8], 
}