import warnings; warnings.simplefilter('ignore')  # hide warnings

# standard libraries
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sb

# modules from particles
import particles  # core module
from particles import distributions as dists  # where probability distributions are defined
from particles import state_space_models as ssm  # where state-space models are defined

class StochVol(ssm.StateSpaceModel):
    def PX0(self):  # Distribution of X_0
        return dists.Normal(loc=self.mu, scale=self.sigma / np.sqrt(1. - self.rho**2))
    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma)
    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=0., scale=np.exp(x))

my_model = StochVol(mu=-1., rho=.9, sigma=.1)  # actual model
true_states, data = my_model.simulate(100)  # we simulate from the model 100 data points

plt.style.use('ggplot')
plt.figure()
plt.plot(data)

fk_model=ssm.Bootstrap(ssm=my_model,data=data)
pf=particles.SMC(fk=fk_model,N=100,resampling='stratified',moments=True,store_history=True)
pf.run()
plt.figure()
plt.plot([yt**2 for yt in data],label='data-squared')
plt.plot([m['mean'] for m in pf.summaries.moments], label='filtered volatility')
plt.legend()

#results=particles.multiSMC(fk=fk_model,N=100,nruns=30,qmc={'SMC':False,'SQMC':True})
#results = particles.multiSMC(fk=fk_model, N=100, nruns=30, qmc={'SMC':False, 'SQMC':True})
#plt.figure()
#sb.boxplot(x=[r['output'].logLt for r in results], y=[r['qmc'] for r in results])

smooth_trajectories = pf.hist.backward_sampling(10)
plt.figure()
plt.plot(smooth_trajectories);


prior_dict = {'mu':dists.Normal(),
              'sigma': dists.Gamma(a=1., b=1.),
              'rho':dists.Beta(9., 1.)}
my_prior = dists.StructDist(prior_dict)

from particles import mcmc  # where the MCMC algorithms (PMMH, Particle Gibbs, etc) live
pmmh = mcmc.PMMH(ssm_cls=StochVol, prior=my_prior, data=data, Nx=50, niter = 1000)
pmmh.run()  # Warning: takes a few seconds


plt.figure()
# plot the marginals
burnin = 100  # discard the 100 first iterations
for i, param in enumerate(prior_dict.keys()):
    plt.subplot(2, 2, i+1)
    sb.distplot(pmmh.chain.theta[param][burnin:], 40)
    plt.title(param)


plt.show()


