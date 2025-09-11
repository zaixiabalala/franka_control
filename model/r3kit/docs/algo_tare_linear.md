# Linear Tare Algorithm
Assume the raw ftsensor data $(\mathbf{f}, \mathbf{t})$(in ftsensor coordinate) contains initial offsets $(\mathbf{f}_0, \mathbf{t}_0)$(in ftsensor coordinate) and there exists additional attachments $m$ at offset $\mathbf{c}$(in ftsensor coordinate) to it.
We want to only get the pure interaction wrench.
They should obey to the physics law as balance of wrench when no other interaction, as 
$$\mathbf{R} \cdot (\mathbf{f} - \mathbf{f}_0) = m \mathbf{g}$$
$$\mathbf{R} \cdot (\mathbf{t} - \mathbf{t}_0) = (\mathbf{R} \cdot \mathbf{c}) \times m \mathbf{g}$$
where $\mathbf{R}$ represents the rotation matrix from ftsensor coordinate to gravity coordinate, $\mathbf{g}$ represents the gravitational acceleration in gravity coordinate.
What we can measure/know include $\mathbf{R}$ and $\mathbf{f}$ and $\mathbf{t}$ and $\mathbf{g}$.
So we can use least squares method to calculate $m$ and $\mathbf{f}_0$ from the first equation. Or measure the $m$ also to calculate $\mathbf{f}_0$ from the first equation. And then calculate $\mathbf{c}$ and $\mathbf{t}_0$ from the second equation.
Notice that the second equation cannot be directly solved by least squares method if $m$ unknown since nonlinear.
