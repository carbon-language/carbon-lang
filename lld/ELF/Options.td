include "llvm/Option/OptParser.td"

//===----------------------------------------------------------------------===//
/// Utility Functions
//===----------------------------------------------------------------------===//

def output : Separate<["-"], "o">, MetaVarName<"<path>">,
     HelpText<"Path to file to write output">;
