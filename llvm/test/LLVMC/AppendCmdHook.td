// Check that hooks can be invoked from 'append_cmd'.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx -x c++ %t
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"

// CHECK: std::string MyHook()

def OptList : OptionList<[
(switch_option "dummy1", (help "none")),
(switch_option "dummy2", (help "none"))
]>;

def dummy_tool : Tool<[
(command "dummy_cmd"),
(in_language "dummy_lang"),
(out_language "dummy_lang"),
(actions (case
         // CHECK: , "-arg1"));
         // CHECK: , "-arg2"));
         (switch_on "dummy1"), (append_cmd "-arg1 -arg2"),
         // CHECK: , "-arg3"));
         // CHECK: hooks::MyHook()
         (switch_on "dummy2"), (append_cmd "-arg3 $CALL(MyHook)")))
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
