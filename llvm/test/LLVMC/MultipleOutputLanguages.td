// Check that multiple output languages work.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx %t
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"

def dummy_tool : Tool<[
    (command "dummy_cmd"),
    (in_language "dummy_lang"),
    (out_language "another_dummy_lang", "yet_another_dummy_lang")
]>;

def another_dummy_tool : Tool<[
    (command "another_dummy_cmd"),
    (in_language "another_dummy_lang", "some_other_dummy_lang"),
    (out_language "executable"),
    (join)
]>;

// CHECK: new SimpleEdge("dummy_tool")
// CHECK: new SimpleEdge("another_dummy_tool")
def DummyGraph : CompilationGraph<[
    (edge "root", "dummy_tool"),
    (edge "dummy_tool", "another_dummy_tool")
]>;
