include "llvm/Option/OptParser.td"

//===----------------------------------------------------------------------===//
/// Utility Functions
//===----------------------------------------------------------------------===//

def output : Separate<["-"], "o">, MetaVarName<"<path>">,
     HelpText<"Path to file to write output">;

def dynamic_linker : Separate<["-"], "dynamic-linker">,
     HelpText<"Which dynamic linker to use">;

def rpath : Separate<["-"], "rpath">,
     HelpText<"Add a DT_RUNPATH to the output">;

def shared : Flag<["-"], "shared">,
     HelpText<"Build a shared object">;
