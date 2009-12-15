// Check that forward_transformed_value works.
// The dummy tool and graph are required to silence warnings.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: grep HookA %t | count 2
// RUN: grep HookB %t | count 2
// RUN: %compile_cxx -fexceptions -x c++ %t

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[(parameter_option "a", (extern)),
                          (prefix_list_option "b", (extern))]>;

def dummy_tool : Tool<[
(cmd_line "dummy_cmd $INFILE"),
(in_language "dummy"),
(out_language "dummy"),
(actions (case
         (not_empty "a"), (forward_transformed_value "a", "HookA"),
         (not_empty "b"), (forward_transformed_value "b", "HookB")))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
