// Check that (init true/false) and (init "str") work.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: grep cl::init("some-string") %t
// RUN: grep cl::init(true) %t

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[
(switch_option "dummy1", (help "none"), (init true)),
(parameter_option "dummy2", (help "none"), (init "some-string"))
]>;

def dummy_tool : Tool<[
(cmd_line "dummy_cmd $INFILE"),
(in_language "dummy_lang"),
(out_language "dummy_lang"),
(actions (case
         (switch_on "dummy1"), (forward "dummy1"),
         (not_empty "dummy2"), (forward "dummy2")))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
