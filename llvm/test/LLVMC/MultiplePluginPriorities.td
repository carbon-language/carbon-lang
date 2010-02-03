// Check that multiple plugin priorities are not allowed.
// RUN: ignore tblgen -I %p/../../include --gen-llvmc %s |& grep "More than one 'PluginPriority' instance found"

include "llvm/CompilerDriver/Common.td"

def Graph : CompilationGraph<[]>;

def Priority1 : PluginPriority<1>;

def Priority2 : PluginPriority<2>;
