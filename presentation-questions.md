Questions...

## Brett

- Q: How did you get matlab to generate such nice plots of 3D slices of the domain?
- A: I used the slice command in Matlab. it allows the used to set up 3-D space, identify a matrix, then select planes of constant x,y,z to display. It looks something like "slice(X,Y,Z, 3-D Data Set, x-plane, y-plane, z-plane)". You can set any of the plane inputs as a vector to select multiple planes at once as well.

## Jack

- Q: Did you run into any "missing" features in Julia that you wanted to use, but weren't implemented in the native libraries? I ran into issues with the Numba just-in-time compiler for Python, which supported pretty much everything I wanted to do except for some of the FFT libraries in Numpy.
- A: No, I didn't run into anything that I would have used, but wasn't available. Most of the native libraries are quite mature, and the compatibility with C makes easy use of existing high-performance linear algebra and scientific computing libraries.

## Simon

- Q: How did you initialize the perturbation in your initial particle states for the two-stream and the Dory-Guest-Harris instabilities? I had a lot of trouble initializing a perturbation that resulted in a smooth-looking charge density after passing through my first-order weighting function.

## Brady

- Q: If you were to re-implement the MHD solver using the same hybrid python/c++ approach, would you make any major changes in how you pass data between the two (how you store the output data)? Might there be performance problems writing very large arrays out to .csv files?

## Cameron

- Q: If you had unlimited time (which of course none of us do), what would you change about your implementation, either the PIC code or the MHD code? I know I definitely rushed and cut some corners for the assignments with tighter deadlines, and ended up having to clean some things up just so that I could generate nice visualizations for the presentation.