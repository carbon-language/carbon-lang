// Check that hooks can be invoked from 'append_cmd'.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx -fexceptions -x c++ %t

include "llvm/CompilerDriver/Common.td"

// CHECK: std::string MyHook()

def OptList : OptionList<[
(switch_option "dummy1", (help "none")),
(switch_option "dummy2", (help "none"))
]>;

def dummy_tool : Tool<[
(cmd_line "dummy_cmd $INFILE"),
(in_language "dummy_lang"),
(out_language "dummy_lang"),
(actions (case
         // CHECK: push_back("-arg1")
         // CHECK: push_back("-arg2")
         (switch_on "dummy1"), (append_cmd "-arg1 -arg2"),
         // CHECK: push_back("-arg3")
         // CHECK: hooks::MyHook()
         (switch_on "dummy2"), (append_cmd "-arg3 $CALL(MyHook)")))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
