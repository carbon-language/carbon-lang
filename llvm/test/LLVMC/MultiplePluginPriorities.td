// Check that multiple plugin priorities are not allowed.
// RUN: ignore tblgen -I %p/../../include --gen-llvmc %s |& grep "More than one 'PluginPriority' instance found"

// Disable for Darwin PPC: <rdar://problem/7598390>
// XFAIL: powerpc-apple-darwin

// Generally XFAIL'ed for now, this is (sometimes?) failing on x86_64-apple-darwin10.
// RUN: false
// XFAIL: *

include "llvm/CompilerDriver/Common.td"

def Graph : CompilationGraph<[]>;

def Priority1 : PluginPriority<1>;

def Priority2 : PluginPriority<2>;
