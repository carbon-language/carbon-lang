// Check the fix for PR4159.
// http://llvm.org/bugs/show_bug.cgi?id=4159
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: grep unique_name %t

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[(parameter_option "dummy", (extern))]>;

def dummy_tool : Tool<[
(cmd_line "dummy_cmd"),
(in_language "dummy"),
(out_language "dummy"),
(actions (case
         (not_empty "dummy"), (forward_as "dummy", "unique_name")))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
