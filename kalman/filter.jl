function filter_step(x, P, z, R, A, H, Q)
    # Predict the next state of the system
    x_pred = A * x
    P_pred = A * P * A' + Q

    # Compute the Kalman gain
    K = P_pred * H' * inv(H * P_pred * H' + R)

    # Update the state estimate based on the measurement
    x = x_pred + K * (z - H * x_pred)
    P = P_pred - K * H * P_pred

    # Return the updated state estimate
    return x, P
end
