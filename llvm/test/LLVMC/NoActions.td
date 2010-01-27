// Check that tools without associated actions are accepted.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx -Qunused-arguments -fexceptions -x c++ %t

include "llvm/CompilerDriver/Common.td"

// CHECK: class dummy_tool : public Tool {
def dummy_tool : Tool<[
(cmd_line "dummy_cmd $INFILE"),
(in_language "dummy"),
(out_language "dummy")
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
