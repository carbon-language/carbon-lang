// Check that multiple plugin priorities are not allowed.
// RUN: ignore tblgen -I %p/../../include --gen-llvmc %s |& grep "More than one 'PluginPriority' instance found"
// XFAIL: vg_leak

// Disable for Darwin PPC: <rdar://problem/7598390>
// XFAIL: powerpc-apple-darwin

// This also fails on i386-darwin9 for some reason
// XFAIL: i386-darwin9

include "llvm/CompilerDriver/Common.td"

def Graph : CompilationGraph<[]>;

def Priority1 : PluginPriority<1>;

def Priority2 : PluginPriority<2>;
