function unscented_kalman_filter(x, P, z, R, f, h, Q, alpha, beta, kappa)
    # Compute the sigma points for the state distribution
    n = length(x)
    lambda = alpha^2 * (n + kappa) - n
    sigma_points = zeros(2n+1, n)
    sigma_points[1, :] = x
    sigma_weights = zeros(2n+1)
    sigma_weights[1] = lambda / (n + lambda)
    W = 1 / (2 * (n + lambda)) * ones(2n)
    for i = 1:n
        sigma_points[i+1, :] = x + sqrt(n + lambda) * cholesky(P)'[i, :]
        sigma_points[n+i+1, :] = x - sqrt(n + lambda) * cholesky(P)'[i, :]
    end

    # Transform the sigma points through the process and measurement functions
    x_pred = f(sigma_points, W)
    z_pred = h(sigma_points, W)

    # Compute the predicted state mean and covariance
    x_pred = sum(sigma_weights .* x_pred)
    P_pred = Q
    for i = 1:2n+1
        P_pred = P_pred + sigma_weights[i] * (x_pred - x_pred[i]) * (x_pred - x_pred[i])'
    end

    # Compute the predicted measurement mean and covariance
    z_pred = sum(sigma_weights .* z_pred)
    S = R
    for i = 1:2n+1
        S = S + sigma_weights[i] * (z_pred - z_pred[i]) * (z_pred - z_pred[i])'
    end

    # Compute the cross-covariance between the state and measurement predictions
    P_xz = zeros(n, length(z))
    for i = 1:2n+1
        P_xz = P_xz + sigma_weights[i] * (x_pred - x_pred[i]) * (z_pred - z_pred[i])'
    end

    # Compute the Kalman gain
    K = P_xz * inv(S)

    # Update the state estimate based on the measurement
    x = x_pred + K * (z - z_pred)
    P = P_pred - K * S * K'

    # Return the updated state estimate
    return x, P
end
