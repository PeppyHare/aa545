Questions...

## Brett

- Q: How did you get matlab to generate such nice plots of 3D slices of the domain?
- A: I used the slice command in Matlab. it allows the used to set up 3-D space, identify a matrix, then select planes of constant x,y,z to display. It looks something like "slice(X,Y,Z, 3-D Data Set, x-plane, y-plane, z-plane)". You can set any of the plane inputs as a vector to select multiple planes at once as well.

## Jack

- Q: Did you run into any "missing" features in Julia that you wanted to use, but weren't implemented in the native libraries? I ran into issues with the Numba just-in-time compiler for Python, which supported pretty much everything I wanted to do except for some of the FFT libraries in Numpy.
- A: No, I didn't run into anything that I would have used, but wasn't available. Most of the native libraries are quite mature, and the compatibility with C makes easy use of existing high-performance linear algebra and scientific computing libraries.

## Simon

- Q: How did you initialize the perturbation in your initial particle states for the two-stream and the Dory-Guest-Harris instabilities? I had a lot of trouble initializing a perturbation that resulted in a smooth-looking charge density after passing through my first-order weighting function.
- A: For the two stream instability I initialized my particles uniformly, then added a small sinusoidal perturbation. The full function is below:

def generate_beam_particles(num_beam_particles, x_domain_endpoints):
    '''
    Generates the initial particle state of two counter streaming beams with a small perturbation (but otherwise uniform position)
    '''
    x_domain_length = x_domain_endpoints[1]-x_domain_endpoints[0]
    beam = linspace(x_domain_endpoints[0], x_domain_endpoints[1], num_beam_particles)
    positive_beam_x = beam+beam_displacement_amplitude*sin(beam*2*pi/x_domain_length)
    negative_beam_x = -beam+beam_displacement_amplitude*sin(beam*2*pi/x_domain_length*+pi)
    combined_beams_x = concatenate((positive_beam_x, negative_beam_x))
    particle_x = expand_dims(combined_beams_x, 1)
    positive_beam_vx = beam_velocity*ones((num_beam_particles, 1))
    negative_beam_vx = -beam_velocity*ones((num_beam_particles, 1))
    particle_vx = concatenate((positive_beam_vx, negative_beam_vx))
    initial_particle_state = [particle_x, particle_vx]
    return initial_particle_state

For the Dory Guest Harris instability I did essentially the same thing for position, where I sampled uniformly then added a sinusoidal perturbation but for velocity I uniformly sampled a length equal to the circumference of the ring then mapped this to x and y components using sine and cosine. The ring distribution implicitly has radius 1 in my code, which in retrospect I should have made explicit. Also for the position in this case I made sure that the perturbation didn't cause particles to leave the domain, which I should have done the first time. The function is below:

def generate_initial_particle_state(num_particles, v0):
    '''
    Generates an initial particle state according to a cold ring distribution

    The particle state is a list where the first component holds the particle location and the second component
    holds the particle velocities as a matrix with two rows, the first row contains the x components and the
    second row contains the y components of the velocity. The columns correspond to particles.

    The ring distribution is sampled by uniformly sampling around the ring then mapping to x and y components.
    The particle's positions are evenly distributed along the domain.
    '''
    particle_positions = linspace(x_domain_endpoints[0], x_domain_endpoints[1], num_particles)

    # Particle position perturbation
    particle_positions += perturbation_amplitude*sin(2*pi*particle_positions/x_domain_length)
    # Enforcing our domain boundaries
    left_boundary = x_domain_endpoints[0]
    right_boundary = x_domain_endpoints[1]
    if any(logical_or(particle_positions<left_boundary, particle_positions>right_boundary)):
        too_low_indices = nonzero(particle_positions<left_boundary)[0]
        too_high_indices = nonzero(particle_positions>right_boundary)[0]
        for i in range(len(too_low_indices)):
            particle_positions[too_low_indices[i]] += x_domain_length
        for i in range(len(too_high_indices)):
            particle_positions[too_high_indices[i]] -= x_domain_length

    particle_velocities = zeros((2, num_particles))
    particle_velocity_samples = uniform(0, 2*pi*v0, num_particles)
    particle_velocities[0, :] = cos(particle_velocity_samples)
    particle_velocities[1, :] = sin(particle_velocity_samples)
    return [particle_positions, particle_velocities]

## Brady

- Q: If you were to re-implement the MHD solver using the same hybrid python/c++ approach, would you make any major changes in how you pass data between the two (how you store the output data)? Might there be performance problems writing very large arrays out to .csv files?
- A: I generally try to pass bare minimum out of C++ code. So for the MHD I would have only saved the handful timesteps where I was going to plot the results. This would probably be defined with some record interval parameter.

It would definitely be faster to either pass the arrays directly without saving them or saving them with some sort of binary file. I think it's still worth saving them because I tend to rerun my plotting code more often than the model to adjustment to labels and such. I have a habit of sticking to CSVs because I've had to figure out how to import someone's strange (but usually efficient) file format a few too many times.

## Cameron

- Q: If you had unlimited time (which of course none of us do), what would you change about your implementation, either the PIC code or the MHD code? I know I definitely rushed and cut some corners for the assignments with tighter deadlines, and ended up having to clean some things up just so that I could generate nice visualizations for the presentation.
- A: I would definitely adjust my graphs for a better cool factor, and I would also explore my particle weightings in PIC as I think that was the root of most of my problems in the codes. Pretty sure there was some minor error somewhere that I just could not catch


## Josh

- Q: How did you choose to handle some of the divergences that appear in the curl of fields near the r=0 axis?
- A: I added a small positive shift to my grid positions. By calculating quantities on a shifted grid, I didn't have to calculate the curl at points with divergences.

## Reed

- Q: For the linear advection finite difference assignment, did you have multiple spatial dimensions, or did you solve the equations in 1D? I thought you had some 3D plots of solutions from the advection equation. I feel like that would have been a good choice to prepare for the 3D MHD finite difference assignment, even if it would require a longer time for the advection simulations to run.
