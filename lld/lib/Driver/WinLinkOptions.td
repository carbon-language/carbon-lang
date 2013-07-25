include "llvm/Option/OptParser.td"

// link.exe accepts options starting with either a dash or a slash.

class _Flag<string name> : Flag<["-", "/"], name>;
class _Separate<string name> : Separate<["-", "/"], name>;
class _Joined<string name, Option alias> : Joined<["-", "/"], name>,
      Alias<alias>;

def mllvm : _Flag<"mllvm">, HelpText<"Options to pass to LLVM">;

def out : _Separate<"out">, HelpText<"Path to file to write output">;
def out_c : _Joined<"out:", out>;

def defaultlib : _Separate<"defaultlib">,
    HelpText<"Add the library to the list of input files">;
def defaultlib_c : _Joined<"defaultlib:", defaultlib>;

def base : _Separate<"base">, HelpText<"Base address of the program">;
def base_c: _Joined<"base:", base>;

def stack : _Separate<"stack">, HelpText<"Size of the stack">;
def stack_c: _Joined<"stack:", stack>;

def heap : _Separate<"heap">, HelpText<"Size of the heap">;
def heap_c: _Joined<"heap:", heap>;

def subsystem : _Separate<"subsystem">, HelpText<"Specify subsystem">;
def subsystem_c: _Joined<"subsystem:", subsystem>;

def entry : _Separate<"entry">, HelpText<"Name of entry point symbol">;
def entry_c: _Joined<"entry:", entry>;

def libpath : _Separate<"libpath">, HelpText<"Additional library search path">;
def libpath_c: _Joined<"libpath:", libpath>;

def force : _Flag<"force">,
    HelpText<"Allow undefined symbols when creating executables">;

def nxcompat : _Flag<"nxcompat">,
    HelpText<"Enable data execution provention">;
def no_nxcompat : _Flag<"nxcompat:no">,
    HelpText<"Disable data execution provention">;

def largeaddressaware : _Flag<"largeaddressaware">,
    HelpText<"Enable large addresses">;
def no_largeaddressaware : _Flag<"largeaddressaware:no">,
    HelpText<"Disable large addresses">;

def fixed : _Flag<"fixed">, HelpText<"Disable base relocations">;
def no_fixed : _Flag<"fixed:no">, HelpText<"Enable base relocations">;

def tsaware : _Flag<"tsaware">,
    HelpText<"Create Terminal Server aware executable">;
def no_tsaware : _Flag<"tsaware:no">,
    HelpText<"Create non-Terminal Server aware executable">;

def incl : _Separate<"include">,
    HelpText<"Force symbol to be added to symbol table as undefined one">;
def incl_c : _Joined<"include:", incl>;

// No help text because /failifmismatch is not intended to be used by the user.
def failifmismatch : _Separate<"failifmismatch">;
def failifmismatch_c : _Joined<"failifmismatch:", failifmismatch>;

// NOLOGO does nothing. It is defined only for link.exe compatibility.
def nologo : _Flag<"nologo">;

def help : _Flag<"help">;
def help_q : _Flag<"?">, Alias<help>;
