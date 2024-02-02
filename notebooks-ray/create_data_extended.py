# Databricks notebook source
# MAGIC %md
# MAGIC ### Create simulated data for training EvoGraphNet GAN
# MAGIC #### @author stephen.offer@databricks.com
# MAGIC #### date 02-Feb-2024
# MAGIC

# COMMAND ----------

import numpy as np
import os 

write_dir = "/dbfs/ml/blogs/gan/evographnet/data-extended"

if not os.path.exists(write_dir):
  os.mkdir(write_dir)
  
mean, std = np.random.rand(), np.random.rand()

n_roi = 155   ### number of ROIs ### default (n_roi,n_roi) ROIs
n_sub = 400   ### default 113 subjects

for i in range(1, n_sub):

	# Create adjacency matrices

	t0 = np.abs(np.random.normal(mean, std, (n_roi,n_roi))) % 1.0
	mean_s = mean + np.random.rand() % 0.1
	std_s = std + np.random.rand() % 0.1
	t1 = np.abs(np.random.normal(mean_s, std_s, (n_roi,n_roi))) % 1.0
	mean_s = mean + np.random.rand() % 0.1
	std_s = std + np.random.rand() % 0.1
	t2 = np.abs(np.random.normal(mean_s, std_s, (n_roi,n_roi))) % 1.0

	# Make them symmetric

	t0 = (t0 + t0.T)/2
	t1 = (t1 + t1.T)/2
	t2 = (t2 + t2.T)/2

	# Clean the diagonals
	t0[np.diag_indices_from(t0)] = 0
	t1[np.diag_indices_from(t1)] = 0
	t2[np.diag_indices_from(t2)] = 0

	# Save them
	s = "cortical.lh.ShapeConnectivityTensor_OAS2_"
	if i < 10:
		s += "0"
	s += "00" + str(i) + "_MR1"

	t0_s = os.path.join(write_dir, s + "_t0.txt")
	t1_s = os.path.join(write_dir,s + "_t1.txt")
	t2_s = os.path.join(write_dir,s + "_t2.txt")

	np.savetxt(t0_s, t0)
	np.savetxt(t1_s, t1)
	np.savetxt(t2_s, t2)
print(t0_s)
print(t1_s)
print(t2_s)

# COMMAND ----------

import numpy as np
print(help(np.savetxt))

# COMMAND ----------


