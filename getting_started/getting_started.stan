functions {
    #include gptools/util.stan
    #include gptools/fft.stan
}

data {
    int n;
    real<lower=0> sigma, length_scale, period;
}

transformed data {
    vector [n %/% 2 + 1] cov_rfft =
        gp_periodic_exp_quad_cov_rfft(n, sigma, length_scale, period) + 1e-9;
}

parameters {
    vector [n] f;
}

model {
    f ~ gp_rfft(zeros_vector(n), cov_rfft);
}
