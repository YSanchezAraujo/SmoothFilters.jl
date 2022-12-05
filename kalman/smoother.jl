function kalman_smoother(x, P, A, Q)
    # Compute the smoothed state estimate using the RTS algorithm
    n = length(x)
    x_smooth = zeros(n)
    P_smooth = zeros(n, n)

    # Compute the initial smoothed state and covariance
    x_smooth[n] = x[n]
    P_smooth[n, n] = P[n, n]

    # Loop backwards through the states, applying the RTS equations
    for k = n-1:-1:1
        L = P[k, k+1:n] * inv(P[k+1:n, k+1:n])
        x_smooth[k] = x[k] + L * (x_smooth[k+1:n] - A[k+1:n, k] * x[k])
        P_smooth[k, k] = P[k, k] + L * (P_smooth[k+1:n, k+1:n] - P[k+1:n, k+1:n]) * L'
        P_smooth[k, k+1:n] = P[k, k+1:n] - P[k, k+1:n] * L'
    end

    # Return the smoothed state estimates
    return x_smooth, P_smooth
end
