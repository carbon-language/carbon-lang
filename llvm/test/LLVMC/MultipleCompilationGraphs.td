// Check that multiple compilation graphs are allowed.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: %compile_cxx -fexceptions -x c++ %t

include "llvm/CompilerDriver/Common.td"

def Graph1 : CompilationGraph<[]>;
def Graph2 : CompilationGraph<[]>;
def Graph3 : CompilationGraph<[]>;
