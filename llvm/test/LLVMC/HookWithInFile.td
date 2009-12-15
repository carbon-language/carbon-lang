// Check that a hook can be given $INFILE as an argument.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: grep Hook\\(inFile.c_str\\(\\)\\) %t | count 1
// RUN: %compile_cxx -fexceptions -x c++ %t

include "llvm/CompilerDriver/Common.td"

def dummy_tool : Tool<[
(cmd_line "$CALL(Hook, '$INFILE')/path $INFILE"),
(in_language "dummy"),
(out_language "dummy")
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
