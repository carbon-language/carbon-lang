// Check the fix for PR4157.
// http://llvm.org/bugs/show_bug.cgi?id=4157
// RUN: tblgen -I %p/../../include --gen-llvmc %s -o %t
// RUN: not grep {FOO")));} %t
// RUN: %compile_cxx -x c++ %t
// XFAIL: vg_leak

include "llvm/CompilerDriver/Common.td"

def dummy_tool : Tool<[
(command "gcc $ENV(FOO)/bar"),
(in_language "dummy"),
(out_language "dummy")
]>;

def DummyGraph : CompilationGraph<[(edge "root", "dummy_tool")]>;

def Graph : CompilationGraph<[]>;
