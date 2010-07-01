// Test that the -filelist option works correctly with -linker=c++.
// RUN: llvmc --dry-run -filelist DUMMY -linker c++ |& grep llvm-g++
// XFAIL: vg
