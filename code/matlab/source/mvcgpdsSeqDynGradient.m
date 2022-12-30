function g = mvcgpdsSeqDynGradient(x, model, modelAll, y, samd)
% mvcgpdsSeqDynGradient Gradients of the mvcgpds for test data.

   vardistx = model.vardistx;
   vardistx = vardistExpandParam(vardistx, x);
   
g = - mvcgpdsSeqDynLogLikeGradient(model, modelAll, vardistx, y, samd);
