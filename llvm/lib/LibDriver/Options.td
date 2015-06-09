include "llvm/Option/OptParser.td"

// lib.exe accepts options starting with either a dash or a slash.

// Flag that takes no arguments.
class F<string name> : Flag<["/", "-", "-?"], name>;

// Flag that takes one argument after ":".
class P<string name, string help> :
      Joined<["/", "-", "-?"], name#":">, HelpText<help>;

def out    : P<"out", "Path to file to write output">;

//==============================================================================
// The flags below do nothing. They are defined only for lib.exe compatibility.
//==============================================================================
def nologo : F<"nologo">;
