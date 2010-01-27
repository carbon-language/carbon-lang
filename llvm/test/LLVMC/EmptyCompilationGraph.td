// Check that the compilation graph can be empty.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: %compile_cxx -Qunused-arguments -fexceptions -x c++ %t

include "llvm/CompilerDriver/Common.td"

def Graph : CompilationGraph<[]>;
