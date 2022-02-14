# CNRMppe_save
"Investigating parametric dependence of climate feedbacks in the atmospheric component of CNRM-CM6-1." (S.Peatier, B.M.Sanderson, L.Terray, R.Roehrig) 

This repository contains the data files and code for the analyses presented in the paper. 

The Jupyter Notebooks to reproduce the paper's analyses are [here](https://github.com/speatier/CNRMppe_save/tree/main/PPE/PPE_paper_1/MultiLinReg), with each Notebook considering an individual metric ([E<sub>tas</sub>](https://github.com/speatier/CNRMppe_save/blob/main/PPE/PPE_paper_1/MultiLinReg/Final_tas.ipynb), [E<sub>pr</sub>](https://github.com/speatier/CNRMppe_save/blob/main/PPE/PPE_paper_1/MultiLinReg/Final_pr.ipynb), [E<sub>SW</sub>](https://github.com/speatier/CNRMppe_save/blob/main/PPE/PPE_paper_1/MultiLinReg/Final_SW.ipynb), [E<sub>LW</sub>](https://github.com/speatier/CNRMppe_save/blob/main/PPE/PPE_paper_1/MultiLinReg/Final_LW.ipynb)).

The PPE and emulated data files (in .npy) are contains [here](https://github.com/speatier/CNRMppe_save/tree/main/PPE/ENSEMBLE2/files/npy) and the optimal candidates data files are here : CNRMppe_save/PPE/ENSEMBLE4_selection/files/npy/ or here : CNRMppe_save/PPE/ENSEMBLE2/files/npy/CNRMppe/



The Notebook for estimating and analyzing the aggregated metric E<sub>tot</sub> is here :  CNRMppe_save/PPE/PPE_paper_1/MultiLinReg/Final_total.ipynb

The same analyse has been conducted considering another type of linear emulator (the LASSO model), you can find the data files, codes and figures here : CNRMppe_save/PPE/PPE_paper_1/LASSO/
