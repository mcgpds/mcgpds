function [ neglikelihood ] = calculateMSLL( YtsOriginal,  Varmu, Varcovars)
likelihood = 0;
for n = 1:size(YtsOriginal,1)
    likelihood = likelihood - size(YtsOriginal,2)/2*log(2*pi)-0.5*sum(log((Varcovars(n,:))))-0.5*sum((YtsOriginal(n,:)-Varmu(n,:)).*...
        (1./Varcovars(n,:)).*(YtsOriginal(n,:)-Varmu(n,:)));
end
neglikelihood = - likelihood/prod(size(YtsOriginal));

end

