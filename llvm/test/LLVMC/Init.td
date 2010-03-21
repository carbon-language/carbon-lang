// Check that (init true/false) and (init "str") work.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx -fexceptions -x c++ %t
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[
// CHECK: cl::init(true)
(switch_option "dummy1", (help "none"), (init true)),
// CHECK: cl::init("some-string")
(parameter_option "dummy2", (help "none"), (init "some-string"))
]>;

def dummy_tool : Tool<[
(command "dummy_cmd"),
(in_language "dummy_lang"),
(out_language "dummy_lang"),
(actions (case
         (switch_on "dummy1"), (forward "dummy1"),
         (not_empty "dummy2"), (forward "dummy2")))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
