# install.packages('depmixS4')
# install.packages('quantmod')
library('depmixS4')
library('quantmod')
set.seed(1)

##################Hidden Markov Model with simulated data#########################
Nk_lower <- 50
Nk_upper <- 150
bull_mean <- 0.1
bull_var <- 0.1
bear_mean <- -0.05
bear_var <- 0.2

days <- replicate(5, sample(Nk_lower:Nk_upper, 1))
market_bull_1 <- rnorm( days[1], bull_mean, bull_var ) 
market_bear_2 <- rnorm( days[2], bear_mean, bear_var ) 
market_bull_3 <- rnorm( days[3], bull_mean, bull_var ) 
market_bear_4 <- rnorm( days[4], bear_mean, bear_var ) 
market_bull_5 <- rnorm( days[5], bull_mean, bull_var )

true_regimes <- c( rep(1,days[1]), rep(2,days[2]), rep(1,days[3]),
                   rep(2,days[4]), rep(1,days[5]))
returns <- c( market_bull_1, market_bear_2, market_bull_3, 
              market_bear_4, market_bull_5)

plot(returns, type="l", xlab='', ylab="Returns") 

#Hidden Markov Model
hmm <- depmix(returns ~ 1, family = gaussian(), nstates = 2, data=data.frame(returns=returns))
hmmfit <- fit(hmm, verbose = FALSE)

# posterior probabilities of the regimes
post_probs <- posterior(hmmfit)
layout(1:2)
plot(post_probs$state, type='s', main='True Regimes', xlab='', ylab='Regime')
matplot(post_probs[,-1], type='l', main='Regime Posterior Probabilities', ylab='Probability')
legend(x='topright', c('Bull','Bear'), fill=1:2, bty='n')



############Hidden Markov Model with actual data##################################
getSymbols( "^NSEI", from="2016-01-01" )
gspcRets = diff( log( Cl( NSEI ) ) )
returns = as.numeric(gspcRets)

plot(gspcRets)

hmm <- depmix(returns ~ 1, family = gaussian(), nstates = 2, data=data.frame(returns=returns))
hmmfit <- fit(hmm, verbose = FALSE)
post_probs <- posterior(hmmfit)

layout(1:2)
plot(index(gspcRets),returns, type='l', main='Regime Detection Nifty 50', xlab='', ylab='Returns')
matplot(index(gspcRets),post_probs[,-1], type='l', main='Regime Posterior Probabilities', ylab='Probability')
legend(x='bottomleft', c('Regime #1','Regime #2'), fill=1:2, bty='n')
