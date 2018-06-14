BOLT
====

  BOLT is a post-link optimizer developed to speed up large applications.
  It achieves speed-ups by optimizing application's code layout based on an
  execution profile gathered by sampling profilers such as Linux `perf` tool.
  BOLT could operate on any binary with symbol table, but for maximum gains
  it utilizes relocations saved by a linker (--emit-relocs).
  
  NOTE: Currently BOLT support is limited to non-PIC/PIE binaries.

INSTALLATION
============

  BOLT heavily uses LLVM libraries and by design it is built as one of LLVM
  tools. The build process in not much different from regular LLVM.

  Start with cloning LLVM and BOLT repos:

  > git clone https://github.com/llvm-mirror/llvm llvm
  > cd llvm/tools
  > git checkout -b llvm-bolt f137ed238db11440f03083b1c88b7ffc0f4af65e
  > git clone https://github.com/facebookincubator/BOLT llvm-bolt
  > patch -p 1 < llvm-bolt/llvm.patch

  Proceed to a normal LLVM build:

  > cd ../..
  > mkdir build
  > cd build
  > cmake -G Ninja
  > ninja

  llvm-bolt will be available under bin/ .

  Note that we use a specific revision of LLVM as we currently rely on a set of
  patches that are not yet upstreamed.
