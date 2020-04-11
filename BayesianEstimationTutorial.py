import warnings; warnings.simplefilter('ignore')  # hide warnings

from matplotlib import pyplot as plt
import numpy as np

from particles import distributions as dists

prior_dict = {'mu': dists.Normal(scale=2.),
              'rho': dists.Uniform(a=-1., b=1.),
              'sigma':dists.Gamma()}
my_prior = dists.StructDist(prior_dict)

theta = my_prior.rvs(size=500)  # sample 500 theta-parameters

plt.style.use('ggplot')
plt.hist(theta['sigma'], 30);
plt.xlabel('sigma')

plt.figure()
z = my_prior.logpdf(theta)
plt.hist(z, 30)
plt.xlabel('log-pdf');


plt.figure()
another_prior_dict = {'rho': dists.Uniform(a=-1., b=1.),
                      'log_sigma':dists.LogD(dists.Gamma())}
another_prior = dists.StructDist(another_prior_dict)
another_theta = another_prior.rvs(size=100)

plt.hist(another_theta['log_sigma'], 20)
plt.xlabel('log-sigma');



from collections import OrderedDict

dep_prior_dict = OrderedDict()
dep_prior_dict['rho'] = dists.Uniform(a=0., b=1.)
dep_prior_dict['sigma'] = dists.Cond( lambda theta: dists.Gamma(b=1./theta['rho']))
dep_prior = dists.StructDist(dep_prior_dict)
dep_theta = dep_prior.rvs(size=2000)

plt.figure()

plt.scatter(dep_theta['rho'], dep_theta['sigma'])
plt.axis([0., 1., 0., 8.])
plt.xlabel('rho')
plt.ylabel('sigma');



reg_prior_dict = OrderedDict()
reg_prior_dict['sigma2'] = dists.InvGamma(a=2., b=3.)
reg_prior_dict['beta'] = dists.MvNormal(cov=np.eye(20))
reg_prior = dists.StructDist(reg_prior_dict)
reg_theta = reg_prior.rvs(size=200)




from particles import state_space_models as ssm

class StochVol(ssm.StateSpaceModel):
    default_parameters = {'mu':-1., 'rho':0.95, 'sigma': 0.2}
    def PX0(self):  # Distribution of X_0
        return dists.Normal(loc=self.mu, scale=self.sigma / np.sqrt(1. - self.rho**2))
    def PX(self, t, xp):  # Distribution of X_t given X_{t-1}=xp (p=past)
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma)
    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x (and possibly X_{t-1}=xp)
        return dists.Normal(loc=0., scale=np.exp(x))

from particles import mcmc

# real data
raw_data = np.loadtxt('./datasets/GBP_vs_USD_9798.txt', skiprows=2, usecols=(3,), comments='(C)')
full_data = np.diff(np.log(raw_data))
data = full_data[:50]

my_pmmh = mcmc.PMMH(ssm_cls=StochVol, prior=my_prior, data=data, Nx=200,
                    niter=1000)
my_pmmh.run();  # may take several seconds...


for p in prior_dict.keys():  # loop over mu, theta, rho
    plt.figure()
    plt.plot(my_pmmh.chain.theta[p])
    plt.xlabel('iter')
    plt.ylabel(p)

class PGStochVol(mcmc.ParticleGibbs):
    def update_theta(self, theta, x):
        new_theta = theta.copy()
        sigma, rho = 0.2, 0.95  # fixed values
        xlag = np.array(x[1:] + [0.,])
        dx = (x - rho * xlag) / (1. - rho)
        s = sigma / (1. - rho)**2
        new_theta['mu'] = self.prior.laws['mu'].posterior(dx, sigma=s).rvs()
        return new_theta

pg = PGStochVol(ssm_cls=StochVol, data=data, prior=my_prior, Nx=200, niter=1000)
pg.run()  # may take several seconds...

plt.plot(pg.chain.theta['mu'])
plt.xlabel('iter')
plt.ylabel('mu')

plt.figure()
plt.hist(pg.chain.theta['mu'][20:], 50)
plt.xlabel('mu');


import particles
from particles import smc_samplers as ssp

fk_smc2 = ssp.SMC2(ssm_cls=StochVol, data=data, prior=my_prior,init_Nx=50,
                   ar_to_increase_Nx=0.1)
alg_smc2 = particles.SMC(fk=fk_smc2, N=500)
alg_smc2.run()

plt.figure()
plt.scatter(alg_smc2.X.theta['mu'], alg_smc2.X.theta['rho'])
plt.xlabel('mu')
plt.ylabel('rho');



plt.show()