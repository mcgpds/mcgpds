% MLTOOLS toolbox
% Version 0.138		22-Jul-2011
% Copyright (c) 2011, Neil D. Lawrence
% 
, Neil D. Lawrence
% DNETEXPANDPARAM Update dnet model with new vector of parameters.
% MODELPOINTLOGLIKELIHOOD Compute the log likelihood of a given point.
% ISOMAPOPTIONS Options for a isomap.
% MODELWRITERESULT Write a model to file.
% DEMOILLLE3 Demonstrate LLE on the oil data.
% DNETUPDATEOUTPUTWEIGHTS Do an M-step (update parameters) on an Density Network model.
% MLPLOGLIKELIHOOD Multi-layer perceptron log likelihood.
% RBFCREATE Wrapper for NETLAB's rbf `net'.
% SPECTRALUPDATELAPLACIAN Update the Laplacian using graph connections.
% MOGESTEP Do an E-step on an MOG model.
% DNETOBJECTIVE Wrapper function for Density Network objective.
% KBRPARAMINIT KBR model parameter initialisation.
% SMALLRANDEMBED Embed data set with small random values.
% VECTORMODIFY Helper code for visualisation of vectorial data.
% MODELOPTIONS Returns a default options structure for the given model.
% MOGPROJECT Project a mixture of Gaussians to a low dimensional space.
% PPCAOUT Output of an PPCA model.
% MODELLOGLIKELIHOOD Compute a model log likelihood.
% MODELWRITETOFID Write to a stream a given model.
% DNETRECONSTRUCT Reconstruct an DNET form component parts.
% MODELPOSTERIORVAR variances of the posterior at points given by X.
% IMAGEVISUALISE Helper code for showing an image during 2-D visualisation.
% MOGUPDATECOVARIANCE Update the covariances of an MOG model.
% MODELOUTPUTGRAD Compute derivatives with respect to params of model outputs.
% RBFPERIODICOUTPUTGRAD Evaluate derivatives of RBFPERIODIC model outputs with respect to parameters.
% DEMSWISSROLLLLE3 Demonstrate LLE on the oil data.
% LINEARCREATE Create a linear model.
% MODELOUTPUTGRADX Compute derivatives with respect to model inputs of model outputs.
% LVMSCATTERPLOTNEIGHBOURS 2-D scatter plot of the latent points with neighbourhood.
% LEOPTIMISE Optimise an LE model.
% KPCAEMBED Embed data set with kernel PCA.
% LLERECONSTRUCT Reconstruct an LLE form component parts.
% DISTANCEWARP Dynamic Time Warping Algorithm
% PLOT3MODIFY Helper code for visualisation of 3-d data.
% KBRDISPLAY Display parameters of the KBR model.
% KBREXPANDPARAM Create model structure from KBR model's parameters.
% SPRINGDAMPERSMODIFY Helper code for visualisation of springDamper data.
% MVUCREATE Maximum variance unfolding embedding model.
% MULTIMODELPARAMINIT MULTIMODEL model parameter initialisation.
% LINEARDISPLAY Display a linear model.
% MOGLOWERBOUND Computes lower bound on log likelihood for an MOG model.
% LVMCLASSVISUALISE Callback function for visualising data.
% RBFOPTIONS Default options for RBF network.
% ISOMAPDECONSTRUCT break isomap in pieces for saving.
% DNETWRITERESULT Write a DNET result.
% MODELEXPANDPARAM Update a model structure with parameters.
% DNETDECONSTRUCT break DNET in pieces for saving.
% DNETOPTIMISE Optimise an DNET model.
% LVMVISUALISE Visualise the manifold.
% DNETLOGLIKELIHOOD Density network log likelihood.
% MULTIMODELOPTIONS Create a default options structure for the MULTIMODEL model.
% MOGUPDATEPRIOR Update the priors of an MOG model.
% LEOPTIONS Options for a Laplacian eigenmaps.
% MOGSAMPLE Sample from a mixture of Gaussians model.
% MODELGRADIENTCHECK Check gradients of given model.
% MLPCREATE Multi-layer peceptron model.
% KBREXTRACTPARAM Extract parameters from the KBR model structure.
% PPCADECONSTRUCT break PPCA in pieces for saving.
% MODELPOSTERIORMEANVAR Mean and variances of the posterior at points given by X.
% MLPLOGLIKEGRADIENTS Multi-layer perceptron gradients.
% LLEOPTIMISE Optimise an LLE model.
% LVMRESULTSDYNAMIC Load a results file and visualise them.
% PPCACREATE Density network model.
% MODELGRADIENT Gradient of error function to minimise for given model.
% MOGUPDATEMEAN Update the means of an MOG model.
% SPECTRUMVISUALISE Helper code for showing an spectrum during 2-D visualisation.
% MVUOPTIMISE Optimise an MVU model.
% LINEARLOGLIKEGRADIENTS Linear model gradients.
% DOUBLEMATRIXREADFROMFID Read a full matrix from an FID.
% MODELLOADRESULT Load a previously saved result.
% KBRCREATE Create a KBR model.
% MLPOUTPUTGRAD Evaluate derivatives of mlp model outputs with respect to parameters.
% DNETTEST Test some settings for the density network.
% MODELSETOUTPUTWEIGHTS Wrapper function to return set output weight and bias matrices.
% MODELGETOUTPUTWEIGHTS Wrapper function to return output weight and bias matrices.
% PPCAPOSTERIORVAR Mean and variances of the posterior at points given by X.
% LVMSCATTERPLOTCOLOR 2-D scatter plot of the latent points with color.
% LVMSCATTERPLOT 2-D scatter plot of the latent points.
% LVMCLICKVISUALISE Visualise the manifold using clicks.
% MODELHESSIAN Hessian of error function to minimise for given model.
% DEMSWISSROLLFULLLLE1 Demonstrate LLE on the oil data.
% DNETUPDATEBETA Do an M-step (update parameters) on an Density Network model.
% MODELSAMP Give a sample from a model for given X.
% MATRIXREADFROMFID Read a matrix from an FID.
% MULTIMODELEXPANDPARAM Create model structure from MULTIMODEL model's parameters.
% RBFPERIODICLOGLIKELIHOOD Log likelihood of RBFPERIODIC model.
% DEMSWISSROLLLLE4 Demonstrate LLE on the oil data.
% LMVUEMBED Embed data set with landmark MVU
% RBFEXPANDPARAM Update rbf model with new vector of parameters.
% MLPOUT Output of an MLP model.
% DOUBLEMATRIXWRITETOFID Writes a double matrix to an FID.
% LFMVISUALISE Visualise the outputs in a latent force model
% RBFOUTPUTGRAD Evaluate derivatives of rbf model outputs with respect to parameters.
% RBFPERIODICDISPLAY Display parameters of the RBFPERIODIC model.
% SWISSROLLSCATTER 3-D scatter plot with colors.
% DNETOPTIONS Options for a density network.
% PPCARECONSTRUCT Reconstruct an PPCA form component parts.
% LVMLOADRESULT Load a previously saved result.
% MULTIMODELEXTRACTPARAM Extract parameters from the MULTIMODEL model structure.
% MULTIMODELLOGLIKELIHOOD Log likelihood of MULTIMODEL model.
% LINEARLOGLIKELIHOOD Linear model log likelihood.
% MODELLOGLIKEGRADIENTS Compute a model's gradients wrt log likelihood.
% LINEAROUTPUTGRAD Evaluate derivatives of linear model outputs with respect to parameters.
% MLTOOLSTOOLBOXES Load in the relevant toolboxes for the MLTOOLS.
% MLPDISPLAY Display the multi-layer perceptron model.
% ISOMAPOPTIMISE Optimise an ISOMAP model.
% ISOMAPRECONSTRUCT Reconstruct an isomap form component parts.
% LINEAROUTPUTGRADX Evaluate derivatives of linear model outputs with respect to inputs.
% LLEDECONSTRUCT break LLE in pieces for saving.
% MVUDECONSTRUCT break MVU in pieces for saving.
% SPRINGDAMPERSVISUALISE Helper code for showing an spring dampers during 2-D visualisation.
% LFMRESULTSDYNAMIC Load a results file and visualise them.
% ISOMAPCREATE isomap embedding model.
% DEMOILLLE2 Demonstrate LLE on the oil data.
% LFMCLASSVISUALISE Callback function to visualize LFM in 2D
% KBROPTIMISE Optimise a KBR model.
% LINEAREXTRACTPARAM Extract weights from a linear model.
% LVMTHREEDPLOT Helper function for plotting the labels in 3-D.
% DEMSWISSROLLFULLLLE4 Demonstrate LLE on the oil data.
% DEMSWISSROLLLLE2 Demonstrate LLE on the oil data.
% MOGCREATE Create a mixtures of Gaussians model.
% MLPEXPANDPARAM Update mlp model with new vector of parameters.
% LLEOPTIONS Options for a locally linear embedding.
% LVMNEARESTNEIGHBOUR Give the number of errors in latent space for 1 nearest neighbour.
% RBFOPTIMISE Optimise RBF for given inputs and outputs.
% RBFPERIODICOUTPUTGRADX Evaluate derivatives of a RBFPERIODIC model's output with respect to inputs.
% DEMSWISSROLLFULLLLE5 Demonstrate LLE on the oil data.
% DNETPOSTERIORMEANVAR Mean and variances of the posterior at points given by X.
% DNETOUTPUTGRADX Evaluate derivatives of DNET model outputs with respect to inputs.
% FINDACYCLICNEIGHBOURS find the k nearest neighbours for each point in Y preventing cycles in the graph.
% MODELOBJECTIVE Objective function to minimise for given model.
% PARAMNAMEREVERSELOOKUP Returns the index of the parameter with the given name.
% KBROPTIONS Create a default options structure for the KBR model.
% LINEAROUT Obtain the output of the linear model.
% DNETLOADRESULT Load a previously saved result.
% DNETOUT Output of an DNET model.
% LVMTWODPLOT Helper function for plotting the labels in 2-D.
% MLPOPTIONS Options for the multi-layered perceptron.
% MAPMODELREADFROMFID Load from a FID produced by C++ code.
% MVUOPTIONS Options for a MVU.
% DNETLOWERBOUND Computes lower bound on log likelihood for an DNET model.
% RBFPERIODICCREATE Create a RBFPERIODIC model.
% LINEARPARAMINIT Initialise the parameters of an LINEAR model.
% MLPOPTIMISE Optimise MLP for given inputs and outputs.
% LEDECONSTRUCT break LE in pieces for saving.
% DNETLOGLIKEGRADIENTS Density network gradients.
% MODELDISPLAY Display a text output of a model.
% RBFEXTRACTPARAM Wrapper for NETLAB's rbfpak.
% RBFPERIODICEXPANDPARAM Create model structure from RBFPERIODIC model's parameters.
% MVURECONSTRUCT Reconstruct an MVU form component parts.
% RBFOUTPUTGRADX Evaluate derivatives of a RBF model's output with respect to inputs.
% FINDDIRECTEDNEIGHBOURS find the k nearest neighbours for each point in Y preventing cycles in the graph.
% RBFPERIODICLOGLIKEGRADIENTS Gradient of RBFPERIODIC model log likelihood with respect to parameters.
% LVMCLASSVISUALISEPATH Latent variable model path drawing in latent space.
% LVMSETPLOT Sets up the plot for visualization of the latent space.
% MODELPARAMINIT Initialise the parameters of the model.
% RBFDISPLAY Display an RBF network.
% MOGOPTIMISE Optimise an MOG model.
% MODELCREATE Create a model of the specified type.
% LVMRESULTSCLICK Load a results file and visualise them with clicks
% LLEEMBED Embed data set with LLE.
% DNETGRADIENT Density Network gradient wrapper.
% MODELEXTRACTPARAM Extract the parameters of a model.
% KBROUTPUTGRAD Evaluate derivatives of KBR model outputs with respect to parameters.
% MOGLOGLIKELIHOOD Mixture of Gaussian's log likelihood.
% MODELADDDYNAMICS Add a dynamics kernel to the model.
% MAPPINGOPTIMISE Optimise the given model.
% RBFPERIODICOUT Compute the output of a RBFPERIODIC model given the structure and input X.
% RBFOUT Output of an RBF model.
% KBROUT Compute the output of a KBR model given the structure and input X.
% LINEAROPTIMISE Optimise a linear model.
% MOGPRINTPLOT Print projection of MOG into two dimensions.
% LINEAROPTIONS Options for learning a linear model.
% DEMOILLLE4 Demonstrate LLE on the oil data.
% MULTIMODELDISPLAY Display parameters of the MULTIMODEL model.
% ISOMAPEMBED Embed data set with Isomap.
% MODELREADFROMFID Load from a FID produced by C++ code.
% LINEAREXPANDPARAM Update linear model with vector of parameters.
% RBFPERIODICOPTIONS Create a default options structure for the RBFPERIODIC model.
% SPECTRUMMODIFY Helper code for visualisation of spectrum data.
% MLPOUTPUTGRADX Evaluate derivatives of mlp model outputs with respect to inputs.
% VECTORVISUALISE  Helper code for plotting a vector during 2-D visualisation.
% MLPPARAMINIT Initialise the parameters of an MLP model.
% FINDACYCLICNEIGHBOURS2 find the k nearest neighbours for each point in Y preventing cycles in the graph.
% MLPEXTRACTPARAM Extract weights and biases from an MLP.
% DEMOILLLE1 Demonstrate LLE on the oil data.
% MOGMEANCOV Project a mixture of Gaussians to a low dimensional space.
% PARAMNAMEREGULAREXPRESSIONLOOKUP Returns the indices of the parameter containing the given regular expression.
% DEMSWISSROLLFULLLLE2 Demonstrate LLE on the oil data.
% DEMSWISSROLLFULLLLE3 Demonstrate LLE on the oil data.
% VITERBIALIGN Compute the Viterbi alignment.
% DNETCREATE Density network model.
% LECREATE Laplacian eigenmap model.
% DEMSWISSROLLLLE1 Demonstrate LLE on the oil data.
% MULTIMODELLOGLIKEGRADIENTS Gradient of MULTIMODEL model log likelihood with respect to parameters.
% MVUEMBED Embed data set with MVU.
% MODELOUT Give the output of a model for given X.
% LVMCLASSCLICKVISUALISE Callback function for visualising data in 2-D with clicks.
% LVMSCOREMODEL Score model with a GP log likelihood.
% DEMMPPCA1 Demonstrate MPPCA on a artificial dataset.
% MOGTWODPLOT Helper function for plotting the labels in 2-D.
% MODELOPTIMISE Optimise the given model.
% PPCAOPTIONS Options for probabilistic PCA.
% SPECTRALUPDATEX Update the latent representation for spectral model.
% DNETEXTRACTPARAM Extract weights and biases from an DNET.
% MODELTEST Run some tests on the specified model.
% MLPLOGLIKEHESSIAN Multi-layer perceptron Hessian.
% DNETESTEP Do an E-step (update importance weights) on an Density Network model.
% MODELREADFROMFILE Read model from a file FID produced by the C++ implementation.
% IMAGEMODIFY Helper code for visualisation of image data.
% MODELTIEPARAM Tie parameters of a model together.
% DNETOUTPUTGRAD Evaluate derivatives of dnet model outputs with respect to parameters.
% PPCAEMBED Embed data set with probabilistic PCA.
% RBFPERIODICEXTRACTPARAM Extract parameters from the RBFPERIODIC model structure.
% LERECONSTRUCT Reconstruct an LE form component parts.
% MULTIMODELCREATE Create a MULTIMODEL model.
% PLOT3VISUALISE  Helper code for plotting a plot3 visualisation.
% FINDNEIGHBOURS find the k nearest neighbours for each point in Y.
% RBFPERIODICPARAMINIT RBFPERIODIC model parameter initialisation.
% LVMPRINTPLOT Print latent space for learnt model.
% PPCAPOSTERIORMEANVAR Mean and variances of the posterior at points given by X.
% LLECREATE Locally linear embedding model.
% MOGOPTIONS Sets the default options structure for MOG models.
% DEMSWISSROLLLLE5 Demonstrate LLE on the oil data.
