+++
title = 'Diffusion Model Explain'
date = 2023-12-23T19:09:04+08:00
draft = false
tags = ["tech"]
categories = ["tutorial"]
math = true
+++
# Likelihood Maximization and ELBO
Assume data is generated from some latent variable $z$. It might include higher-level representations such as color and shape. The goal is to use this latent variable to get new samples. We can introduce a joint probability $p(x,z)$ and try to maximize likelihood of *p(x)* 
$$p(x)=\int p(x,z)dz$$
Since integration is intractable, we apply Bayes theorem instead.
$$p(x)=\frac{p(x,z)}{p(z|x)}$$
True posterior is unavailable to us so we use above equation to derive log likelihood. Firstly we introduce $q_\phi$ by multiply 1 to $p(x)$
$$log p(x)=log p(x)\int q_\phi (z|x)dz $$
Use definition of expectation and multiply by 1 again
$$=E_{q_\phi (z|x)}[log p(x)]$$
$$=E_{q_\phi (z|x)}[log\frac{p(x,z)q_\phi (z|x)}{p(z|x)q_\phi (z|x)}]$$
Split expectation and define KL divergence.
$$=E_{q_\phi (z|x)}[log\frac{p(x,z)}{q_\phi (z|x)}] + D_{KL}(q_\phi (z|x)||p(z|x))$$
$$\eqslantgtr E_{q_\phi (z|x)}[log\frac{p(x,z)}{q_\phi (z|x)}]$$
Above term is called **Evidence Lower Bound (ELBO)** ELBO is the lower bound because KL is non-negative so ELBO cannot exceeds evidence. From above a model $q_\phi (z|x)$ is introduced to approximate the true posterior distribution which aligns with our goal to model the latent variable so we aim to minimize this KL term. Since we do not know the true distribution we can minimize KL by maximizing ELBO instead because left hand side likelihood is a constant respect to $\phi$.
# Variational Autoencoder (VAE)
Variational method is the approximation of distribution with some parameters, by introducing autoencoder structure, we come up with VAE. Let's derive ELBO further. Firstly split joint distribution by introducing another model $\theta$ to approximate real distribution.
$$E_{q_\phi (z|x)}[log\frac{p(x,z)}{q_\phi (z|x)}]=E_{q_\phi (z|x)}[log\frac{p_\theta(x|z)p(z)}{q_\phi (z|x)}]$$
By spliting expectation and define KL,
$$=E_{q_\phi (z|x)}[logp_\theta(x|z)]-D_{KL}(q_\phi (z|x)||p(z))$$
We define **decoder** and **encoder** respectively. Decoder aims to reconstruct input $x$ while encoder tries to match posterior with prior distribution. Ususally we define encoder as a multivariate Gaussian so KL can be estimated by MC sampling. Since latent $z$ is sampled from encoder, we use **reparameterization trick** to makes it differentiable.
$$z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$$
Where $\epsilon$ is a standard Gaussian $N$. During training, VAE tries to reconstruct the input $x$. During inference, we discard encoder and sample latent from the learnt $\mu_\phi$ and $\sigma_\phi$.
# Hierarchical VAE and Variational Diffusion Models
Hierarchical VAE (HVAE) is formed by series of VAE models, as we could construt the higher-level abstract latent for more complex task. We can also assign Markov property to get MHVAE. Also We can assign 3 properties to MHVAE so we can now call it Variational Diffusion Models(VDM). 

**(1)** Latent dimension equal to data dimension

**(2)** Encoder of latent layers is fixed (linear Gaussian that centers at previpus timestamp)

**(3)** Change encoder params so at step T image becomes a standard Gaussian.
Define the **noise schedule** that control noise at each timestep
$$\alpha_t+\beta_t=1$$
Define forward/encoder and reverse/decoder process
$$q(x_t|x_{t-1}) = N(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)$$
$$q(x_{1:T}|x_{0})=\prod_{t=1}^{T}q(x_t|x_{t-1})$$
$$p_\theta(x_{t-1}|x_{t}) = N(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$$
$$p_\theta(x_{0:T})=p(x_T)\prod_{t=1}^{T}p_\theta(x_{t-1}|x_{t})$$
Notice in reverse process we use the network to approximate the real mean and covariance. We could generate new samples from the Gaussian noise and run denoising transision T times. Similar to VAE we could optimzie by ELBO.
$$log p(x)=log\int p(x_{0:T})dx_{1:T}$$
Multiply with $q(x_{1:T}|x_0)$ (introduce q)and define expectation. Use Jensen Inequality.
$$=log E_{q(x_{1:T}|x_0)}[\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}]$$
$$\eqslantgtr E_{q(x_{1:T}|x_0)}[log\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}]$$
We get **ELBO** term again! After derivation it becomes several expectations and we can appy Monte Carlo estimates. However for each time step it involves $x_{t-1},x_{t+1}$ and we need to sum all $T-1$ terms so it cause high variance! How can we involve less variables? Firstly according to Markov property we have the form $q(x_{t}|x_{t-1})=q(x_{t}|x_{t-1},x_0)$, then by Bayes Rule,
$$q(x_{t}|x_{t-1},x_0) = \frac{q(x_{t}|x_0)q(x_{t-1}|x_{t},x_0)}{q(x_{t-1}|x_0)}$$
Derive from the above ELBO we get,
$$log p(x)\eqslantgtr E_{q(x_{1:T}|x_0)}[log\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}]$$
Expand $p,q$
$$=E_{q(x_{1:T}|x_0)}[log\frac{p(x_{T})p_\theta(x_{0}|x_{1})\prod_{t=2}^{T}p_\theta(x_{t-1}|x_{t})}{q(x_{1}|x_0)\prod_{t=2}^{T}q(x_{t}|x_{t-1})}]$$
Use above substitution for $q(x_{t}|x_{t-1})$ we get,
$$=E_{q(x_{1:T}|x_0)}[log\frac{p(x_{T})p_\theta(x_{0}|x_{1})\prod_{t=2}^{T}p_\theta(x_{t-1}|x_{t})}{q(x_{1}|x_0)\prod_{t=2}^{T}\frac{q(x_{t}|x_0)q(x_{t-1}|x_{t},x_0)}{q(x_{t-1}|x_0)}}]$$
$$=E_{q(x_{1:T}|x_0)}[log\frac{p(x_{T})p_\theta(x_{0}|x_{1})}{q(x_{1}|x_0)}+log\prod_{t=2}^{T}\frac{p_\theta(x_{t-1}|x_{t})}{\frac{q(x_{t}|x_0)q(x_{t-1}|x_{t},x_0)}{q(x_{t-1}|x_0)}}]$$
$q(x_{t-1}|x_0),q(x_{t}|x_0)$ terms cancel each others and few remain
$$=E_{q(x_{1:T}|x_0)}[log\frac{p(x_{T})p_\theta(x_{0}|x_{1})}{q(x_{1}|x_0)}+log\frac{q(x_{1}|x_0)}{q(x_{T}|x_0)}+log\prod_{t=2}^{T}\frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t-1}|x_{t},x_0)}]$$
Change products to sums, split and redefine range of expectations.
$$=E_{q(x_{1:T}|x_0)}[log\frac{p(x_{T})p_\theta(x_{0}|x_{1})}{q(x_{T}|x_0)}+\sum_{t=2}^{T}log\frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t-1}|x_{t},x_0)}]$$
$$=E_{q(x_{1}|x_0)}[log p_\theta(x_{0}|x_{1})]+E_{q(x_{T}|x_0)}[log\frac{p(x_{T})}{q(x_{T}|x_0)}]+\sum_{t=2}^{T}E_{q(x_{t},x_{t-1}|x_0)}[log\frac{p_\theta(x_{t-1}|x_{t})}{q(x_{t-1}|x_{t},x_0)}]$$
$$=E_{q(x_{1}|x_0)}[log p_\theta(x_{0}|x_{1})]-D_{KL}(q(x_{T}|x_0)||p(x_{T}))-\sum_{t=2}^{T}E_{q(x_{t}|x_0)}[D_{KL}(q(x_{t-1}|x_{t},x_0)||p_\theta(x_{t-1}|x_{t}))]$$
Notice we have less variables so lower variance for ELBO estimation. We can divide ELBO into 3 terms:

**Reconstruction term** Similar as VAE decoder, use Monte Carlo estimate.

**Prior matching term** Difference between input at the final step and standrad Gaussian, equal to 0.

**Denoising matching term** Estimate the ground truth denosing step at $t-1$ with a model. This is the major term we need to optimize. But how?

We further derive this term to makes it tractable by expression of Gaussian distribution. Notice **encoder $q$** looks tough to deal with so we firstly apply Bayes rule:
$$q(x_{t-1}|x_{t},x_0) = \frac{q(x_{t-1}|x_0)q(x_{t}|x_{t-1},x_0)}{q(x_{t}|x_0)}$$
Let's dive into these 3 terms individually. According to Markov property we have the form $q(x_{t}|x_{t-1})=q(x_{t}|x_{t-1},x_0)$ which has the closed form expression. We are left with $q(x_t|x_0)$ and $q(x_{t−1}|x_0)$. Since encoder are linear Gaussian models, we can derive recursively. Firstly, apply **reparameterization trick** again to one forward step, assume each noise is iid.
$$x_t=\sqrt{\alpha_t}x_{t-1}+\sqrt{1-\alpha_{t}}\epsilon_{t-1}$$
$$x_t=\sqrt{\alpha_t}(\sqrt{\alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_{t-1}}\epsilon_{t-2})+\sqrt{1-\alpha_{t}}\epsilon_{t-1}$$
$$x_t=\sqrt{\alpha_t \alpha_{t-1}}x_{t-2}+\sqrt{\alpha_t-\alpha_t \alpha_{t-1}}\epsilon_{t-2}+\sqrt{1-\alpha_{t}}\epsilon_{t-1}$$
Sum of two independent Gaussians equal to a Gaussian with mean = sum of means and variance = sum of variances.
$$x_t=\sqrt{\alpha_t \alpha_{t-1}}x_{t-2}+\sqrt{\alpha_t-\alpha_t \alpha_{t-1}+1-\alpha_{t}}\epsilon_{t-2}$$
$$x_t=\sqrt{\alpha_t \alpha_{t-1}}x_{t-2}+\sqrt{1-\alpha_t \alpha_{t-1}}\epsilon_{t-2}$$$$...$$
$$x_t=\sqrt{\prod_{i=1}^t\alpha_i}x_{0}+\sqrt{1-\prod_{i=1}^t\alpha_{i}}\epsilon_{0}$$
$$x_t=\sqrt{\overline{\alpha_i}}x_{0}+\sqrt{1-\overline{\alpha_{i}}}\epsilon_{0}$$
We get the closed form of $q(x_t|x_0)$ and $q(x_{t−1}|x_0)$. Recall the Bayes rule expansion derived above.
$$q(x_{t-1}|x_{t},x_0) = \frac{q(x_{t}|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_{t}|x_0)}$$
$$=\frac{N(x_t;\sqrt{\alpha_t}x_{t-1},(1-\alpha_t)I)N(x_{t-1};\sqrt{\overline{\alpha_{t-1}}}x_{0},(1-{\overline{\alpha_{t-1}}})I)}{N(x_t;\sqrt{\alpha_t}x_{0},(1-\overline{\alpha_t})I)}$$
$$\propto exp\lbrace-\frac{1}{2}[\frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{1-\alpha_t}+\frac{(x_{t-1}-\sqrt{\overline{\alpha_{t-1}}}x_{0})^2}{1-\overline{\alpha_{t-1}}}-\frac{(x_t-\sqrt{\alpha_t}x_{0})^2}{1-\overline{\alpha_{t}}}]\rbrace$$
$$=exp\lbrace-\frac{1}{2}[\frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{1-\alpha_t}+\frac{(x_{t-1}-\sqrt{\overline{\alpha_{t-1}}}x_{0})^2}{1-\overline{\alpha_{t-1}}}-\frac{(x_t-\sqrt{\alpha_t}x_{0})^2}{1-\overline{\alpha_{t}}}]\rbrace$$
Separate all terms with only $x_t,x_0,\alpha$ as a constant C (remaining are related with $x_{t-1}$)
$$=exp\lbrace-\frac{1}{2}[\frac{-2\sqrt{\alpha_t}x_{t}x_{t-1}+\alpha_t x_{t-1}^2}{1-\alpha_t}+\frac{x_{t-1}^2-2\sqrt{\overline{\alpha_{t-1}}}x_{t-1}x_{0}}{1-\overline{\alpha_{t-1}}}+C(x_t,x_0)]\rbrace$$
$$\propto exp\lbrace-\frac{1}{2}[(\frac{\alpha_t}{1-\alpha_t}+\frac{1}{1-\overline{\alpha_{t-1}}})x_{t-1}^2-2(\frac{\sqrt{\alpha_t}x_{t}}{1-\alpha_t}+\frac{\sqrt{\overline{\alpha_{t-1}}}x_{0}}{1-\overline{\alpha_{t-1}}})x_{t-1}]\rbrace$$
$$=exp\lbrace-\frac{1}{2}[(\frac{\alpha_t}{1-\alpha_t}+\frac{1}{1-\overline{\alpha_{t-1}}})x_{t-1}^2-2(\frac{\sqrt{\alpha_t}x_{t}}{1-\alpha_t}+\frac{\sqrt{\overline{\alpha_{t-1}}}x_{0}}{1-\overline{\alpha_{t-1}}})x_{t-1}]\rbrace$$
Use the fact $\overline{\alpha_{t-1}}\alpha_{t}=\overline{\alpha_{t}}$, combine first 2 terms
$$=exp\lbrace-\frac{1}{2}[(\frac{1-\overline{\alpha_{t}}}{(1-\alpha_t)(1-\overline{\alpha_{t-1}})})x_{t-1}^2-2(\frac{\sqrt{\alpha_t}x_{t}}{1-\alpha_t}+\frac{\sqrt{\overline{\alpha_{t-1}}}x_{0}}{1-\overline{\alpha_{t-1}}})x_{t-1}]\rbrace$$
Extract common factor and work on the second terms.
$$=exp\lbrace-\frac{1}{2}(\frac{1-\overline{\alpha_{t}}}{(1-\alpha_t)(1-\overline{\alpha_{t-1}})})[x_{t-1}^2-2\frac{(\frac{\sqrt{\alpha_t}x_{t}}{1-\alpha_t}+\frac{\sqrt{\overline{\alpha_{t-1}}}x_{0}}{1-\overline{\alpha_{t-1}}})}{\frac{1-\overline{\alpha_{t}}}{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}}x_{t-1}]\rbrace$$
$$=exp\lbrace-\frac{1}{2}(\frac{1-\overline{\alpha_{t}}}{(1-\alpha_t)(1-\overline{\alpha_{t-1}})})[x_{t-1}^2-2\frac{(\frac{\sqrt{\alpha_t}x_{t}}{1-\alpha_t}+\frac{\sqrt{\overline{\alpha_{t-1}}}x_{0}}{1-\overline{\alpha_{t-1}}})(1-\alpha_t)(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_{t}}}x_{t-1}]\rbrace$$
$$=exp\lbrace-\frac{1}{2}(\frac{1-\overline{\alpha_{t}}}{(1-\alpha_t)(1-\overline{\alpha_{t-1}})})[x_{t-1}^2-2\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})x_{0}}{1-\overline{\alpha_{t}}}x_{t-1}]\rbrace$$
$$=exp\lbrace-\frac{1}{2}(\frac{1}{\frac{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_{t}}}})[x_{t-1}^2-2\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})x_{0}}{1-\overline{\alpha_{t}}}x_{t-1}]\rbrace$$
Use the constant term above to complete square.
$$\propto N(x_{t-1};\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})x_{0}}{1-\overline{\alpha_{t}}},\frac{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_{t}}}I)$$
$$=N(x_{t-1};\mu_q(x_t,x_0),\Sigma_q(t))$$
Where mean and variance is
$$\mu_q(x_t,x_0)=\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})x_{0}}{1-\overline{\alpha_{t}}}$$
$$\sigma_q^2(t)=\frac{(1-\alpha_t)(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_{t}}}$$
Therefore $q(x_{t-1}|x_{t},x_0)$ itself is actually a Gaussian distribution as well. Notice variance is only related t $\alpha$ which is known. Recall we try to optimize **denoising matching term** above so now we need to construct estimate term $p_\theta(x_{t-1}|x_{t})$that approximate ground truth $q(x_{t-1}|x_{t},x_0)$. Analogously, we can model it with a Gaussian distribution. Since we can match their **variance** exactly, match 2 distributions equal to match **mean** value. Let's derive from the definition of KL between 2 Gaussians.
$$\displaystyle\argmin_{\theta} D_{KL}(q(x_{t-1}|x_{t},x_0)||p_\theta(x_{t-1}|x_{t}))$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2}[log\frac{|\Sigma_q(t)|}{|\Sigma_q(t)|}-d+tr(\Sigma_q(t)^{-1}\Sigma_q(t))+(\mu_\theta-\mu_q)^T\Sigma_q(t)^{-1}(\mu_\theta-\mu_q)]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2}[log1-d+d+(\mu_\theta-\mu_q)^T\Sigma_q(t)^{-1}(\mu_\theta-\mu_q)]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2}[(\mu_\theta-\mu_q)^T\Sigma_q(t)^{-1}(\mu_\theta-\mu_q)]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2}[(\mu_\theta-\mu_q)^T\(\sigma^2_q(t)I)^{-1}(\mu_\theta-\mu_q)]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}[||(\mu_\theta-\mu_q)||^2_2]$$
where $\mu_\theta=\mu_\theta(x_t,t), \mu_q = \mu_q(x_t,x_0)$, estimated mean can be further derived according to the true mean above :
$$\mu_\theta(x_t,t)=\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})\hat{x_{\theta}}(x_t,t)}{1-\overline{\alpha_{t}}}$$
Notice we use a network $\hat{x_{\theta}}(x_t,t)$ to predict $x_0$ so above optimization can be written as:
$$\displaystyle\argmin_{\theta} D_{KL}(q(x_{t-1}|x_{t},x_0)||p_\theta(x_{t-1}|x_{t}))$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}[||(\mu_\theta-\mu_q)||^2_2]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}[||(\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})\hat{x_{\theta}}(x_t,t)}{1-\overline{\alpha_{t}}}-\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})x_{0}}{1-\overline{\alpha_{t}}})||^2_2]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}[||(\frac{\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})}{1-\overline{\alpha_{t}}})(\hat{x_{\theta}}(x_t,t)-x_0)||^2_2]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}\frac{\overline{\alpha_{t-1}}(1-\alpha_{t})^2}{(1-\overline{\alpha_{t}})^2}||(\hat{x_{\theta}}(x_t,t)-x_0)||^2_2$$
Now we simplify the objective to predict the original image from a noisy version. Since we need to optimize summation over all time steps, it is equivalent to take expectation over time.
## Three Equivalent Interpretations
### Noise prediction & DDPM
There are  two other equivalent parameterizations other than predict original image $x_0$. What if we replace $x_0$ in above equations? Firstly let's reuse reparameterization trick.
$$x_t=\sqrt{\overline{\alpha_t}}x_{0}+\sqrt{1-\overline{\alpha_{t}}}\epsilon_{0}$$
$$x_0=\frac{x_t-\sqrt{1-\overline{\alpha_{t}}}\epsilon_{0}}{\sqrt{\overline{\alpha_t}}}$$
Plug into true denoising transition mean:
$$\mu_q(x_t,x_0)=\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})\frac{x_t-\sqrt{1-\overline{\alpha_{t}}}\epsilon_{0}}{\sqrt{\overline{\alpha_t}}}}{1-\overline{\alpha_{t}}}$$
$$=\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+(1-\alpha_{t})\frac{x_t-\sqrt{1-\overline{\alpha_{t}}}\epsilon_{0}}{\sqrt{{\alpha_t}}}}{1-\overline{\alpha_{t}}}$$
$$=\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}}{1-\overline{\alpha_{t}}}+\frac{(1-\alpha_{t})x_{t}}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}}-\frac{(1-\alpha_{t})\sqrt{1-\overline{\alpha_{t}}}}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}}\epsilon_{0}$$
$$=(\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_{t}}}+\frac{(1-\alpha_{t})}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}})x_t-\frac{(1-\alpha_{t})\sqrt{1-\overline{\alpha_{t}}}}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}}\epsilon_{0}$$
$$=\frac{\alpha_{t}-\overline{\alpha_{t}}+1-\alpha_{t}}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}}x_t-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha_{t}}}\sqrt{\alpha_t}}\epsilon_{0}$$
$$=\frac{1-\overline{\alpha_{t}}}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}}x_t-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha_{t}}}\sqrt{\alpha_t}}\epsilon_{0}$$
$$=\frac{1}{\sqrt{\alpha_t}}x_t-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha_{t}}}\sqrt{\alpha_t}}\epsilon_{0}$$
We use same trick to express estimated denoisising mean term by modelling $\mu_0$
$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}x_t-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha_{t}}}\sqrt{\alpha_t}}\hat{\epsilon_{\theta}}(x_t,t)$$
Plug into the optimization term above:
$$\displaystyle\argmin_{\theta} D_{KL}(q(x_{t-1}|x_{t},x_0)||p_\theta(x_{t-1}|x_{t}))$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}[||(\frac{1}{\sqrt{\alpha_t}}x_t-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha_{t}}}\sqrt{\alpha_t}}\hat{\epsilon_{\theta}}(x_t,t)-\frac{1}{\sqrt{\alpha_t}}x_t+\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha_{t}}}\sqrt{\alpha_t}}\epsilon_{0})||^2_2]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}[||\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha_{t}}}\sqrt{\alpha_t}}(\epsilon_{0}-\hat{\epsilon_{\theta}}(x_t,t))||^2_2]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}\frac{(1-\alpha_{t})^2}{(1-\overline{\alpha_{t}}){\alpha_t}}[||\epsilon_{0}-\hat{\epsilon_{\theta}}(x_t,t)||^2_2]$$
Notice we use a network $\hat{\epsilon_{\theta}}(x_t,t)$ to model source noise, which is equivalent to original image prediction. What if we ignore the whole prefix which include $\alpha$?
$$L_{simple}=\displaystyle\argmin_\theta[||\epsilon_{0}-\hat{\epsilon_{\theta}}(x_t,t)||^2_2]$$
This is equivalent to the objective of **Denoising Diffusion Probabilistic Models (DDPM)**, where the author propose to use a simple rescaled loss function that downplay the part where t is small. Although this is not equal to the original ELBO, but it performs surprising good when we use noise prediction scheme.

DDPM training starts from a sampling of a time step between 0~T, we can generate corresponding noisy image at this time step. Then, we feed $x_t,t$ to the network to predict the source noise and take gradient decent steps until converge.

During sampling, we sample a noise $x_T$ and use reverse process to get a clear image iteratively.
$$p_\theta(x_{t-1}|x_{t}) = N(x_{t-1};\mu_\theta(x_t,t),\Sigma_\theta(x_t,t))$$
$$x_{t-1} = \mu_\theta(x_t,t)+\sigma_t\epsilon$$
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}x_t-\frac{1-\alpha_{t}}{\sqrt{1-\overline{\alpha_{t}}}\sqrt{\alpha_t}}\hat{\epsilon_{\theta}}(x_t,t)+\sigma_t\epsilon$$
### Predict score function
#### Tweedie’s Formula
Tweedie’s Formula states exponential family distribution true mean = empirical mean + correction term.
$$E[\mu_z|z]=z+\Sigma_z \nabla_z log p(z)$$
According to, we can replace covariance.
$$x_t=\sqrt{\overline{\alpha_t}}x_{0}+\sqrt{1-\overline{\alpha_{t}}}\epsilon_{0}$$
$$E[\mu_{x_t}|x_t]=x_t+(1-\overline{\alpha_{i}})\nabla_{x_t} log p(x_t)$$
We now have 2 equivalent expression of true mean value.
$$\sqrt{\overline{\alpha_t}}x_{0}=x_t+(1-\overline{\alpha_{t}})\nabla log p(x_t)$$
$$x_{0}=\frac{x_t+(1-\overline{\alpha_{t}})\nabla log p(x_t)}{\sqrt{\overline{\alpha_t}}}$$
We then use the same trick to replace $x_0$ in the true transition mean.
$$\mu_q(x_t,x_0)=\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})x_{0}}{1-\overline{\alpha_{t}}}$$
$$\mu_q(x_t,x_0)=\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+\sqrt{\overline{\alpha_{t-1}}}(1-\alpha_{t})\frac{x_t+(1-\overline{\alpha_{t}})\nabla log p(x_t)}{\sqrt{\overline{\alpha_t}}}}{1-\overline{\alpha_{t}}}$$
$$=\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}+(1-\alpha_{t})\frac{x_t+(1-\overline{\alpha_{t}})\nabla log p(x_t)}{\sqrt{{\alpha_t}}}}{1-\overline{\alpha_{t}}}$$
$$=\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})x_{t}}{1-\overline{\alpha_{t}}}+\frac{(1-\alpha_{t})x_{t}}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}}+\frac{(1-\alpha_{t})(1-\overline{\alpha_{t}})\nabla log p(x_t)}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}}$$
$$=(\frac{\sqrt{\alpha_t}(1-\overline{\alpha_{t-1}})}{1-\overline{\alpha_{t}}}+\frac{(1-\alpha_{t})}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}})x_t+\frac{(1-\alpha_{t})}{\sqrt{\alpha_t}}\nabla log p(x_t)$$
$$=\frac{\alpha_{t}-\overline{\alpha_{t}}+1-\alpha_{t}}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}}x_t+\frac{(1-\alpha_{t})}{\sqrt{\alpha_t}}\nabla log p(x_t)$$
$$=\frac{1-\overline{\alpha_{t}}}{(1-\overline{\alpha_{t}})\sqrt{\alpha_t}}x_t+\frac{1-\alpha_{t}}{\sqrt{\alpha_t}}\nabla log p(x_t)$$
$$=\frac{1}{\sqrt{\alpha_t}}x_t+\frac{1-\alpha_{t}}{\sqrt{\alpha_t}}\nabla log p(x_t)$$
Use the similar form to construct estimated denoising mean term 
$$\mu_\theta(x_t,t)=\frac{1}{\sqrt{\alpha_t}}x_t+\frac{1-\alpha_{t}}{\sqrt{\alpha_t}}s_\theta(x_t,t)$$
Same optimization term before and mean value comparison.
$$\displaystyle\argmin_{\theta} D_{KL}(q(x_{t-1}|x_{t},x_0)||p_\theta(x_{t-1}|x_{t}))$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}[||(\mu_\theta-\mu_q)||^2_2]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}[||(\frac{1}{\sqrt{\alpha_t}}x_t+\frac{1-\alpha_{t}}{\sqrt{\alpha_t}}s_\theta(x_t,t)-\frac{1}{\sqrt{\alpha_t}}x_t-\frac{1-\alpha_{t}}{\sqrt{\alpha_t}}\nabla log p(x_t))||^2_2]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}[||\frac{1-\alpha_{t}}{\sqrt{\alpha_t}}(s_\theta(x_t,t)-\nabla log p(x_t))||^2_2]$$
$$=\displaystyle\argmin_{\theta}\frac{1}{2\sigma^2_q(t)}\frac{(1-\alpha_{t})^2}{\alpha_t}[||s_\theta(x_t,t)-\nabla log p(x_t)||^2_2]$$
$s_\theta(x_t,t)$ here is a neural network to predict **score function**, which is gradient of $x_t$. Notice the form is similar to the noise prediction form above, so we combine Tweedie’s Formula with reparameterization trick and get:
$$\nabla log p(x_t)=-\frac{1}{\sqrt{1-\overline{\alpha_{t}}}}\epsilon_0$$
which means optimizing score function equal to reverse noisy corruption. We have shown diffusion model have 3 equivalent optimization targets:

Predict **original image $x_0$**

Predict **source noise** $\epsilon_0$

Predict **Score function $\nabla log p(x_t)$** at a random noise level
# Improvements and Applications
## Guidance
## Faster Sampling