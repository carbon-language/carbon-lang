// Check that tools without associated actions are accepted.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx %t
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"

// CHECK: class dummy_tool : public Tool {
def dummy_tool : Tool<[
(command "dummy_cmd"),
(in_language "dummy"),
(out_language "dummy")
]>;

def DummyGraph : CompilationGraph<[(edge "root", "dummy_tool")]>;
