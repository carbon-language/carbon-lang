// Check that a hook can be given $INFILE as an argument.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx -fexceptions -x c++ %t

include "llvm/CompilerDriver/Common.td"

def dummy_tool : Tool<[
// CHECK: Hook(inFile.c_str())
(cmd_line "$CALL(Hook, '$INFILE')/path $INFILE"),
(in_language "dummy"),
(out_language "dummy")
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
