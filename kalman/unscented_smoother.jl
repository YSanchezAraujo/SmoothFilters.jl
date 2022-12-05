function unscented_kalman_smoother(x, P, A, Q, alpha, beta, kappa)
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

    # Transform the sigma points through the process function
    x_pred = f(sigma_points, W)

    # Compute the predicted state mean and covariance
    x_pred = sum(sigma_weights .* x_pred)
    P_pred = Q
    for i = 1:2n+1
        P_pred = P_pred + sigma_weights[i] * (x_pred - x_pred[i]) * (x_pred - x_pred[i])'
    end

    # Compute the cross-covariance between the state and measurement predictions
    P_xz = zeros(n, n)
    for i = 1:2n+1
        P_xz = P_xz + sigma_weights[i] * (x_pred - x_pred[i]) * (x - x[i])'
    end

    # Compute the smoothed state estimate using the RTS algorithm
    x_smooth = x
    P_smooth = P
    for k = n-1:-1:1
        L = P_pred[k, k+1:n] * inv(P_pred[k+1:n, k+1:n])
        x_smooth[k] = x[k] + L * (x_smooth[k+1:n] - A[k+1:n, k] * x[k])
        P_smooth[k, k] = P[k, k] + L * (P_smooth[k+1:n, k+1:n] - P_pred[k+1:n, k+1:n]) * L'
        P_smooth[k, k+1:n] = P[k, k+1:n] - P[k, k+1:n] * L'
    end

    # Return the smoothed state estimates
    return x_smooth, P_smooth
end
