// Check that hooks with arguments work.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx -fexceptions -x c++ %t
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"

// CHECK: Hook(const char* Arg0, const char* Arg1, const char* Arg2);
// CHECK: "/path"
// CHECK: std::getenv("VARIABLE")
// CHECK: "/2path"

def dummy_tool : Tool<[
(command "$CALL(Hook, 'Arg1',   'Arg2', 'Arg3 Arg3Cont')/path arg1 $ENV(VARIABLE)/2path arg2"),
(in_language "dummy"),
(out_language "dummy")
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
