# food_trade_shocks
Repository associated to the paper: Diversification of global food trade partners increased inequalities in the exposure to shock risks. 
The folder ``Scripts`` contains all the scripts needed to reproduce the main results of the analysis, numbered in sequential order. 

They allow the user to:
1. Pre-process the trade and production data from FAO: scripts from``1_Download_&_filter_data`` to ``4_PreprocessFoodEx_&_production_data``.
2. Implement the topological analysis of the food trade multiplex:``5_Multilayer_Year_map`` and ``6_Plot_alluvial``.
3. Run the stochastic shock propagation model: ``7_Multiproduct_shocks_years_demand`` (parallel implementation).
4. Extract the vulnerability estimate for each country and year: ``8_Minimal_tolerance_extraction``.
5. Reproduce the main plots of the study: ``9_Plots_paper``.
6. Reproduce some Supplementary Analyses: ``10_IPR`` and ``11_Supplementary_analyses``.

Before executing the scripts, the Data must be downloaded from the different sources described in the article and then it should be saved in the ``Data``folder. 
