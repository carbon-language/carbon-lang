// Check that the compilation graph is not required.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: %compile_cxx -fexceptions -x c++ %t

include "llvm/CompilerDriver/Common.td"
