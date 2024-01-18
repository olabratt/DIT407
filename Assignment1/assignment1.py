import numpy as np
import pandas as pd
from matplotlib import pyplot


ratio = lambda children, elderly, labor: 100 * (children + elderly) / labor
total = lambda children, elderly, labor: children + elderly + labor
fraction = lambda part, total: part / total

population = pd.read_csv('swedish_population_by_year_and_sex_1860-2022.csv',  sep=',',)
# Drop sex column, we don't need it
populationNoSex = population.drop(columns=['sex'])

# Set age to numeric
populationNoSex.at[220,'age'] = 110
populationNoSex.at[221,'age'] = 110
# Convert to numeric
populationNoSex['age'] = pd.to_numeric(populationNoSex['age'], errors='coerce', downcast='float')


# Group by age
classes = populationNoSex.groupby(pd.cut(populationNoSex['age'], [-1, 14, 64, 110])).sum()
# Drop age column, we don't need it anymore
classes = classes.drop(columns=['age'])
# Transpose
classesT = classes.transpose()
# Apply lambda functions
classesT['ratio'] = classesT.apply(lambda row: ratio(row.iat[0], row.iat[2], row.iat[1]), axis=1) 
classesT['total'] = classesT.apply(lambda row: total(row.iat[0], row.iat[2], row.iat[1]), axis=1) 
classesT['fraction_children'] = classesT.apply(lambda row: fraction(row.iat[0], row.iat[4]), axis=1) 
classesT['fraction_elderly'] = classesT.apply(lambda row: fraction(row.iat[2], row.iat[4]), axis=1) 
classesT['fraction_labor'] = classesT.apply(lambda row: fraction(row.iat[1], row.iat[4]), axis=1) 

# Convert index to float
years = np.asarray(classesT.index.values, float)

# Plot ratio
fig1, ax1 = pyplot.subplots(figsize=(5, 2.7), layout='constrained')
ax1.plot(years, classesT['ratio'], label='ratio')
ax1.set_xlabel('Year')  # Add an x-label to the axes.
ax1.set_ylabel('Ratio')  # Add a y-label to the axes.
ax1.set_title("Dependency Ratio")  # Add a title to the axes.
ax1.legend()

# Plot fractions
fig2, ax2 = pyplot.subplots(figsize=(5, 2.7), layout='constrained')
ax2.plot(years, classesT['fraction_children'], label='fraction children')
ax2.plot(years, classesT['fraction_elderly'], label='fraction elderly')
ax2.plot(years, classesT['fraction_labor'], label='fraction labor')
ax2.set_xlabel('Year')  # Add an x-label to the axes.
ax2.set_ylabel('Fraction')  # Add a y-label to the axes.
ax2.set_title("Fraction of total population")  # Add a title to the axes.
ax2.legend()