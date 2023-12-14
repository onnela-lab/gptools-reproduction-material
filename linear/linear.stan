// Text file "linear.stan" containing the model definition.

data {
    int n, p;
    matrix [n, p] X;
    vector[n] y;
}

parameters {
    vector[p] theta;
    real<lower=0> sigma;
}

model {
    theta ~ normal(0, 1);
    sigma ~ gamma(2, 2);
    y ~ normal(X * theta, sigma);
}
