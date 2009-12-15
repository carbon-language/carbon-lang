// Check that hooks with arguments work.
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: FileCheck -input-file %t %s
// RUN: %compile_cxx -fexceptions -x c++ %t

include "llvm/CompilerDriver/Common.td"

// CHECK: Hook(const char* Arg0, const char* Arg1, const char* Arg2);
// CHECK: std::getenv("VARIABLE")
// CHECK: "/2path"
// CHECK: "/path"

def dummy_tool : Tool<[
(cmd_line "$CALL(Hook, 'Arg1',   'Arg2', 'Arg3 Arg3Cont')/path arg1 $ENV(VARIABLE)/2path arg2 $INFILE"),
(in_language "dummy"),
(out_language "dummy")
]>;

def DummyGraph : CompilationGraph<[SimpleEdge<"root", "dummy_tool">]>;
