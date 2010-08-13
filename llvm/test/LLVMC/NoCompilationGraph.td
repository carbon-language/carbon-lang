// Check that the compilation graph is not required.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: %compile_cxx -x c++ %t
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"
