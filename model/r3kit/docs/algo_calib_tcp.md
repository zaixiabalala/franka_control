# TCP Calibration Algorithm
To calibrate the offset from tcp to robot end-effector, where we can read the pose from robot end-effector to robot base, we make use of extrinsic point fixed under robot base.
We make contact between tcp and the fixed point under different pose, and read the end-effector pose $(\mathbf{R}_i, \mathbf{t}_i)$.
Since $T_{\text{t2b}} = T_{\text{e2b}} @ T_{\text{t2e}}$, so the fixed point translation relative to robot base remains the same. Therefore, we have $\mathbf{R}_i @ \hat{\mathbf{t}} + \mathbf{t}_i = \mathbf{R}_j @ \hat{\mathbf{t}} + \mathbf{t}_j$. Then we can use least squares method to calculate $\hat{\mathbf{t}}$.
