#//=============================================================
#//(c) 2011 Distributed under MIT-style license. 
#//(see LICENSE.txt or visit http://opensource.org/licenses/MIT)
#//=============================================================


# Do not modify next 6 lines
import bnomics
import sys
from synthetic_generator import synthesize
from synthetic_generator import generate_random_synthetic_data
import copy

# First load the datafile
filename=sys.argv[1]
dt=bnomics.dutils.loader(filename)

# import pandas as pd
# import numpy as np

# # Convert to DataFrame
# df = pd.DataFrame(dt.data, columns=dt.variables)

# # Ensure numeric types
# for col in ['juv_fel_count', 'juv_misd_count', 'juv_other_count']:
#     df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

# # Compute juv_crime
# df['juv_crime'] = df['juv_fel_count'] + df['juv_misd_count'] + df['juv_other_count']

# # Select only the relevant features
# selected_cols = [
#     'race', 'sex', 'two_year_recid', 'score_text', 'priors_count',
#     'age_cat', 'v_score_text', 'juv_crime', 'c_charge_degree'
# ]
# df = df[selected_cols].copy()

# # Assign back to dt
# dt.data = df.values
# dt.variables = df.columns.tolist()
# dt.arity = np.array([len(df[col].unique()) for col in df.columns])



# Modifications possible below

##############################
# Now discretize the variables with
dt.quantize_all()

# Other options include 
# dt.bin_discretize(var_list=[1,2,3],bins=3)
# which takes a list of variables and bin number as arguments.
# To discretize 1st, 2nd, and 3rd variables (count is from 0) do
# dt.bin_discretize([1,2,3])
# To discretize a range of variables [i,...,j], you can use the command
# range(i,j+1) like so
# dt.bin_discretize(range(1,11))


#######################
# Initialize the search
#srch=bnomics.search(dt,ofunc=bnomics.cmdlb)

srch = bnomics.search( dt )

# Parameters for the objective function (metric)
# can be added to the search argument list like so
# srch = bnomics.search( dt, ofunc=bnomics.bdm)
# The default is bnomics.mdl
# If ofext.cpp is compiled bnomics.cmdla (AIC) 
#and bnomics.cmdlb (BIC) represent a faster alternative.


####################
# Perform the search
srch.restarts( nrestarts=20 )

# There are 2 arguments that can be passed into the gsrestarts function
# nrestarts - number of restarts performed during the search
# tol - lower bound of the numerical tolerance
# Both arguments are used as a stopping criterion for the searchin algorithm.


################################################################################
# Save the reconstructed BN structure in dotfile.dot and generate a rendering of
# the result in outpdf.pdf (if Graphviz is properly installed).
srch.dot(filename="law_original_bn", connected_only=False)

# srch_copy = copy.deepcopy(srch)

# synthetic_df = generate_random_synthetic_data(srch, sample_size=10000)
# synthetic_df.to_csv("dutch_synthetic_data.csv", index=False)
# print("✅ synthetic data written to dutch_synthetic_data.csv")

# # ─── 3) Generate debiased synthetic data ─────────────────
# synthetic_debiased_df = synthesize(srch_copy, sample_size=10000)
# # save to CSV
# synthetic_debiased_df.to_csv("dutch_synthetic_debiased_data.csv", index=False)
# print("✅ Debiased synthetic data written to dutch_synthetic_debiased_data.csv")



