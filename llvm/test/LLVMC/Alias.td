// Test alias generation.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx -x c++ %t
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[

(switch_option "dummy1", (help "none")),
// CHECK: cl::alias Alias_dummy2
(alias_option "dummy2", "dummy1")
]>;

def dummy_tool : Tool<[
(command "dummy_cmd"),
(in_language "dummy_lang"),
(out_language "dummy_lang"),
(actions (case
         (switch_on "dummy1"), (forward "dummy1")))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
