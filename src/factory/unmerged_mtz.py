import reciprocalspaceship as rs

# Load the MTZ file
ds = rs.read_mtz("/n/holylabs/LABS/hekstra_lab/Users/fgiehr/creat_dials_unmerged/unmerged.mtz")

# Show the column labels
print("Columns:", ds.columns)

# Preview first few rows
print(ds.head())
