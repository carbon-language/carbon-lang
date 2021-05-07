This is a README for the font compression reference code. There are several
compression related modules in this repository.

brotli/ contains reference code for the Brotli byte-level compression
algorithm. Note that it is licensed under the MIT license.

src/ contains the C++ code for compressing and decompressing fonts.

# Build & Run

This document documents how to run the compression reference code. At this
writing, the code, while it is intended to produce a bytestream that can be
reconstructed into a working font, the reference decompression code is not
done, and the exact format of that bytestream is subject to change.

The build process depends on the g++ compiler.

## Build

On a standard Unix-style environment:

```
git clone --recursive https://github.com/google/woff2.git
cd woff2
make clean all
```

Alternatively, if Brotli is already installed on your system you can use CMake
to build executables and libraries:

```
git clone https://github.com/google/woff2.git
cd woff2
mkdir out
cd out
cmake ..
make
make install
```

By default, shared libraries are built. To use static linkage, do:

```
cd woff2
mkdir out-static
cmake -DBUILD_SHARED_LIBS=OFF ..
make
make install
```

## Run

Ensure the binaries from the build process are in your $PATH, then:

```
woff2_compress myfont.ttf
woff2_decompress myfont.woff2
```

# References

http://www.w3.org/TR/WOFF2/
http://www.w3.org/Submission/MTX/

Also please refer to documents (currently Google Docs):

WOFF Ultra Condensed file format: proposals and discussion of wire format
issues (PDF is in docs/ directory)

WIFF Ultra Condensed: more discussion of results and compression techniques.
This tool was used to prepare the data in that document.
