
import warnings; warnings.simplefilter('ignore')  # hide warnings

# the usual imports
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np

# imports from the package
import particles
from particles import state_space_models as ssm
from particles import distributions as dists


class ThetaLogistic(ssm.StateSpaceModel):
    """ Theta-Logistic state-space model (used in Ecology).
    """
    default_params = {'tau0':.15, 'tau1':.12, 'tau2':.1, 'sigmaX': 0.47, 'sigmaY': 0.39}

    def PX0(self):  # Distribution of X_0
        return dists.Normal()

    def f(self, x):
        return (x + self.tau0 - self.tau1 * np.exp(self.tau2 * x))

    def PX(self, t, xp):  #  Distribution of X_t given X_{t-1} = xp (p=past)
        return dists.Normal(loc=self.f(xp), scale=self.sigmaX)

    def PY(self, t, xp, x):  # Distribution of Y_t given X_t=x, and X_{t-1}=xp
        return dists.Normal(loc=x, scale=self.sigmaY)

my_ssm = ThetaLogistic()  # use default values for all parameters
x, y = my_ssm.simulate(100)

plt.style.use('ggplot')
plt.plot(y)
plt.xlabel('t')
plt.ylabel('data')

just_for_fun = ThetaLogistic(tau2=0.3, bogus=92.)



fk_boot = ssm.Bootstrap(ssm=my_ssm, data=y)
my_alg = particles.SMC(fk=fk_boot, N=100)
my_alg.run()


class ThetaLogistic_with_prop(ThetaLogistic):
    def proposal0(self, data):
        return self.PX0()
    def proposal(self, t, xp, data):
        prec_prior = 1. / self.sigmaX**2
        prec_lik = 1. / self.sigmaY**2
        var = 1. / (prec_prior + prec_lik)
        mu = var * (prec_prior * self.f(xp) + prec_lik * data[t])
        return dists.Normal(loc=mu, scale=np.sqrt(var))

my_better_ssm = ThetaLogistic_with_prop()

fk_guided = ssm.GuidedPF(ssm=my_better_ssm, data=y)


alg = particles.SMC(fk=fk_guided, N=100, seed=None, ESSrmin=0.5, resampling='systematic', store_history=False,
                    compute_moments=False, online_smoothing=None, verbose=False)

next(alg) # processes data-point y_0
next(alg)  # processes data-point y_1
for _ in range(8):
    next(alg)  # processes data-points y_3 to y_9
# alg.run()  # would process all the remaining data-points

plt.figure()
plt.hist(alg.X, 20, weights=alg.W);


plt.figure()
plt.plot(alg.summaries.ESSs)
plt.xlabel('t')
plt.ylabel('ESS')


plt.figure()
plt.plot(alg.summaries.logLts)
plt.xlabel('t')
plt.ylabel('log-likelihood')



plt.figure()
#outf = lambda pf: pf.logLt
#results = particles.multiSMC(fk={'boot':fk_boot, 'guid':fk_guided}, nruns=20, nprocs=1, out_func=outf)
alg_with_mom = particles.SMC(fk=fk_guided, N=100, moments=True)
alg_with_mom.run()

plt.plot([m['mean'] for m in alg_with_mom.summaries.moments], label='filtered mean')
plt.plot(y, label='data')
plt.legend()


plt.figure()
alg = particles.SMC(fk=fk_guided, N=100, store_history=True)
alg.run()

trajectories = alg.hist.backward_sampling(5, linear_cost=False)
plt.plot(trajectories)


plt.figure()
class ThetaLogistic_with_upper_bound(ThetaLogistic_with_prop):
    def upper_bound_log_pt(self, t):
        return -np.log(np.sqrt(2 * np.pi) * self.sigmaX)

my_ssm = ThetaLogistic_with_upper_bound()
alg = particles.SMC(fk=ssm.GuidedPF(ssm=my_ssm, data=y), N=100, store_history=True)
alg.run()
(more_trajectories, acc_rate) = alg.hist.backward_sampling(10, linear_cost=True, return_ar=True)

print('acceptance rate was %1.3f' % acc_rate)
plt.plot(more_trajectories)

plt.show()