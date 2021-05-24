Questions...

## Brett

- Q: How did you get matlab to generate such nice plots of 3D slices of the domain?
- A: I used the slice command in Matlab. it allows the used to set up 3-D space, identify a matrix, then select planes of constant x,y,z to display. It looks something like "slice(X,Y,Z, 3-D Data Set, x-plane, y-plane, z-plane)". You can set any of the plane inputs as a vector to select multiple planes at once as well.

## Jack

- Q: Did you run into any "missing" features in Julia that you wanted to use, but weren't implemented in the native libraries? I ran into issues with the Numba just-in-time compiler for Python, which supported pretty much everything I wanted to do except for some of the FFT libraries in Numpy.
- A: No, I didn't run into anything that I would have used, but wasn't available. Most of the native libraries are quite mature, and the compatibility with C makes easy use of existing high-performance linear algebra and scientific computing libraries.