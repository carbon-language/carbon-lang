// Check that extern options work.
// The dummy tool and graph are required to silence warnings.
// RUN: tblgen -I $srcroot/include --gen-llvmc %s | grep extern

include "llvm/CompilerDriver/Common.td"

def OptList : OptionList<[(extern_switch "Wall"),
    (extern_parameter "std"), (extern_list "L")]>;

def dummy_tool : Tool<[
(cmd_line "dummy_cmd"),
(in_language "dummy"),
(out_language "dummy"),
(actions (case
         (switch_on "Wall"), (stop_compilation),
         (not_empty "std"), (stop_compilation),
         (not_empty "L"), (stop_compilation)))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
