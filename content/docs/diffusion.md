+++
title = 'Diffusion Model'
date = 2023-12-23T19:09:04+08:00
draft = false
tags = ["tech"]
categories = ["tutorial"]
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