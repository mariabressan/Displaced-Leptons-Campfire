# Displaced-Leptons-Campfire
Displaced Leptons Mini Analysis for CAMPFIRE

## Ntuples:
signal: `/eos/atlas/atlascerngroupdisk/phys-susy/displacedleptonsRun3_ANA-SUSY-2022-11/ntuples/v2.1/oneEM/signal_ltrw/unskimmed/SelSelLLP_100_0_10ns.root` <br />
background: `/eos/atlas/atlascerngroupdisk/phys-susy/displacedleptonsRun3_ANA-SUSY-2022-11/ntuples/v1p9/oneEM/user.ancsmith.data22_13p6TeV.00437756.physics_Main.deriv.v9conf_VL_23__trees.root/user.ancsmith.00437756.f1305_m2142_p6000.36817393._000004.trees.root`

## Energyflow Documentation
[https://energyflow.network/]([https://energyflow.network/])

## Looking at ROOT files in Command Line
Open a root file with `root /path/to/file.root` <br />
Check what's in the root file with `.ls` <br />
Print all the branches in a tree with `TREE_NAME->Print()` <br />
Check the values of a variable with `TREE_NAME->Scan(VAR_NAME)` <br/>
You can also look at multiple vairables and place cuts with `TREE_NAME->Scan(VAR_NAME1:VARN_NAME2,cut)` <br />
An example of a cut would be `VARN_NAME1>0` <br />
Plot the histogram of a variable with `TREE_NAME->Draw(VAR_NAME)` <br/>
Make a 2D histogram with `TREE_NAME->Draw(YVAR_NAME:XVAR_NAME)`