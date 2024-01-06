+++
title = 'Diffusion Model'
date = 2023-12-23T19:09:04+08:00
draft = false
tags = ["tech"]
categories = ["tutorial"]
math = true
+++
# Diffusion Model Explain
## Likelihood Maximization and ELBO
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
## Variational Autoencoder (VAE)
Variational method is the approximation of distribution with some parameters, by introducing autoencoder structure, we come up with VAE. Let's derive ELBO further. Firstly split joint distribution by introducing another model $\theta$ to approximate real distribution.
$$E_{q_\phi (z|x)}[log\frac{p(x,z)}{q_\phi (z|x)}]=E_{q_\phi (z|x)}[log\frac{p_\theta(x|z)p(z)}{q_\phi (z|x)}]$$
By spliting expectation and define KL,
$$=E_{q_\phi (z|x)}[logp_\theta(x|z)]-D_{KL}(q_\phi (z|x)||p(z))$$
We define **decoder** and **encoder** respectively. Decoder aims to reconstruct input $x$ while encoder tries to match posterior with prior distribution. Ususally we define encoder as a multivariate Gaussian so KL can be estimated by MC sampling. Since latent $z$ is sampled from encoder, we use **reparameterization trick** to makes it differentiable.
$$z=\mu_\phi(x)+\sigma_\phi(x)\odot\epsilon$$
Where $\epsilon$ is a standard Gaussian $N$. During training, VAE tries to reconstruct the input $x$. During inference, we discard encoder and sample latent from the learnt $\mu_\phi$ and $\sigma_\phi$.
## Hierarchical VAE and Variational Diffusion Models
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
$$q(x_{x}|x_{x-1},x_0) = \frac{q(x_{t}|x_0)q(x_{t-1}|x_{t},x_0)}{q(x_{t-1}|x_0)}$$
Derive from the above ELBO we get,
$$log p(x)\eqslantgtr E_{q(x_{1:T}|x_0)}[log\frac{p(x_{0:T})}{q(x_{1:T}|x_0)}]$$
Expand $p,q$
$$=E_{q(x_{1:T}|x_0)}[log\frac{p(x_{T})p_\theta(x_{0}|x_{1})\prod_{t=2}^{T}p_\theta(x_{t-1}|x_{t})}{q(x_{1}|x_0)\prod_{t=2}^{T}q(x_{t}|x_{t-1})}]$$
Use above substitution for $q(x_{t}|x_{t-1})$ we get,
$$=E_{q(x_{1:T}|x_0)}[log\frac{p(x_{T})p_\theta(x_{0}|x_{1})\prod_{t=2}^{T}p_\theta(x_{t-1}|x_{t})}{q(x_{1}|x_0)\prod_{t=2}^{T}\frac{q(x_{t}|x_0)q(x_{t-1}|x_{t},x_0)}{q(x_{t-1}|x_0)}}]$$