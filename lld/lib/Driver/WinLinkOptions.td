include "llvm/Option/OptParser.td"

// link.exe accepts options starting with either a dash or a slash.

class F<string name> : Flag<["-", "/"], name>;

multiclass P<string name, string help> {
  def "" : Separate<["-", "/"], name>, HelpText<help>;
  def _c : Joined<["-", "/"], name#":">, Alias<!cast<Option>(name)>;
}

defm base    : P<"base", "Base address of the program">;
defm defaultlib : P<"defaultlib", "Add the library to the list of input files">;
defm entry   : P<"entry", "Name of entry point symbol">;
// No help text because /failifmismatch is not intended to be used by the user.
defm failifmismatch : P<"failifmismatch", "">;
defm heap    : P<"heap", "Size of the heap">;
defm libpath : P<"libpath", "Additional library search path">;
defm mllvm   : P<"mllvm", "Options to pass to LLVM">;
defm out     : P<"out", "Path to file to write output">;
defm stack   : P<"stack", "Size of the stack">;
defm subsystem : P<"subsystem", "Specify subsystem">;

// We cannot use multiclass P because class name "incl" is different
// from its command line option name. We do this because "include" is
// a reserved keyword in tablegen.
def incl : Separate<["-", "/"], "include">,
    HelpText<"Force symbol to be added to symbol table as undefined one">;
def incl_c : Joined<["-", "/"], "include:">, Alias<incl>;

def force : F<"force">,
    HelpText<"Allow undefined symbols when creating executables">;

def nxcompat : F<"nxcompat">,
    HelpText<"Enable data execution provention">;
def no_nxcompat : F<"nxcompat:no">,
    HelpText<"Disable data execution provention">;

def largeaddressaware : F<"largeaddressaware">,
    HelpText<"Enable large addresses">;
def no_largeaddressaware : F<"largeaddressaware:no">,
    HelpText<"Disable large addresses">;

def fixed : F<"fixed">, HelpText<"Disable base relocations">;
def no_fixed : F<"fixed:no">, HelpText<"Enable base relocations">;

def tsaware : F<"tsaware">,
    HelpText<"Create Terminal Server aware executable">;
def no_tsaware : F<"tsaware:no">,
    HelpText<"Create non-Terminal Server aware executable">;

// NOLOGO does nothing. It is defined only for link.exe compatibility.
def nologo : F<"nologo">;

def help : F<"help">;
def help_q : F<"?">, Alias<help>;
