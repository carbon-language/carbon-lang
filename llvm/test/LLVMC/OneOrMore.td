// Check that (one_or_more) and (zero_or_one) properties work.
// The dummy tool and graph are required to silence warnings.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx -fexceptions -x c++ %t
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[
    // CHECK: cl::OneOrMore
    (prefix_list_option "foo", (one_or_more)),
    // CHECK: cl::Optional
    (parameter_list_option "baz", (optional))]>;

def dummy_tool : Tool<[
(command "dummy_cmd"),
(in_language "dummy"),
(out_language "dummy"),
(actions (case
         (not_empty "foo"), (forward_as "foo", "bar"),
         (not_empty "baz"), (forward "baz")))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
