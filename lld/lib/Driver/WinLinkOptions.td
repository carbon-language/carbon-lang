include "llvm/Option/OptParser.td"

// link.exe accepts options starting with either a dash or a slash.

// Flag that takes no arguments.
class F<string name> : Flag<["/", "-", "-?"], name>;

// Flag that takes one argument after ":".
multiclass P<string name, string help> {
  def "" : Joined<["/", "-", "-?"], name#":">, HelpText<help>;
  def _c : Separate<["/", "-", "-?"], name>, Alias<!cast<Option>(name)>;
}

// Boolean flag suffixed by ":no".
multiclass B<string name, string help> {
  def "" : F<name>;
  def _no : F<name#":no">, HelpText<help>;
}

defm base    : P<"base", "Base address of the program">;
defm defaultlib : P<"defaultlib", "Add the library to the list of input files">;
defm entry   : P<"entry", "Name of entry point symbol">;
// No help text because /failifmismatch is not intended to be used by the user.
defm failifmismatch : P<"failifmismatch", "">;
defm heap    : P<"heap", "Size of the heap">;
defm align   : P<"align", "Section alignment">;
defm libpath : P<"libpath", "Additional library search path">;
defm mllvm   : P<"mllvm", "Options to pass to LLVM">;
defm out     : P<"out", "Path to file to write output">;
defm stack   : P<"stack", "Size of the stack">;
defm machine : P<"machine", "Specify target platform">;
defm version : P<"version", "Specify a version number in the PE header">;
defm subsystem : P<"subsystem", "Specify subsystem">;

// We cannot use multiclass P because class name "incl" is different
// from its command line option name. We do this because "include" is
// a reserved keyword in tablegen.
def incl : Joined<["/", "-"], "include:">,
    HelpText<"Force symbol to be added to symbol table as undefined one">;
def incl_c : Separate<["/", "-"], "include">, Alias<incl>;

def force : F<"force">,
    HelpText<"Allow undefined symbols when creating executables">;
def force_unresolved : F<"force:unresolved">;

def ref : F<"opt:ref">;
def no_ref : F<"opt:noref">,
    HelpText<"Keep unreferenced symbols to be included to the output">;

defm nxcompat : B<"nxcompat", "Disable data execution provention">;
defm largeaddressaware : B<"largeaddressaware", "Disable large addresses">;
defm allowbind: B<"allowbind", "Disable DLL binding">;
defm fixed    : B<"fixed", "Enable base relocations">;
defm tsaware  : B<"tsaware", "Create non-Terminal Server aware executable">;
defm dynamicbase : B<"dynamicbase",
     "Disable address space layout randomization">;

def help : F<"help">;
def help_q : F<"?">, Alias<help>;

def DASH_DASH : Option<["--"], "", KIND_REMAINING_ARGS>;

// The flags below do nothing. They are defined only for link.exe compatibility.
multiclass Q<string name> {
  def "" : Joined<["/", "-", "-?"], name#":">;
  def _c : Separate<["/", "-", "-?"], name>, Alias<!cast<Option>(name)>;
}
def nologo : F<"nologo">;
defm errorreport : Q<"errorreport">;
def incremental : F<"incremental">;
def no_incremental : F<"incremental:no">;
