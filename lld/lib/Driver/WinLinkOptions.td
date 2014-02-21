include "llvm/Option/OptParser.td"

// link.exe accepts options starting with either a dash or a slash.

// Flag that takes no arguments.
class F<string name> : Flag<["/", "-", "-?"], name>;

// Flag that takes one argument after ":".
class P<string name, string help> :
      Joined<["/", "-", "-?"], name#":">, HelpText<help>;

// Boolean flag suffixed by ":no".
multiclass B<string name, string help> {
  def "" : F<name>;
  def _no : F<name#":no">, HelpText<help>;
}

def alternatename : P<"alternatename", "Define weak alias">;
def base    : P<"base", "Base address of the program">;
def defaultlib : P<"defaultlib", "Add the library to the list of input files">;
def nodefaultlib : P<"nodefaultlib", "Remove a default library">;
def disallowlib : Joined<["/", "-", "-?"], "disallowlib:">, Alias<nodefaultlib>;
def entry   : P<"entry", "Name of entry point symbol">;
// No help text because /failifmismatch is not intended to be used by the user.
def export  : P<"export", "Export a function">;
def failifmismatch : P<"failifmismatch", "">;
def heap    : P<"heap", "Size of the heap">;
def align   : P<"align", "Section alignment">;
def libpath : P<"libpath", "Additional library search path">;
def mllvm   : P<"mllvm", "Options to pass to LLVM">;
def out     : P<"out", "Path to file to write output">;
def stack   : P<"stack", "Size of the stack">;
def machine : P<"machine", "Specify target platform">;
def version : P<"version", "Specify a version number in the PE header">;
def merge   : P<"merge", "Combine sections">;
def section : P<"section", "Specify section attributes">;
def subsystem : P<"subsystem", "Specify subsystem">;
def stub    : P<"stub", "Specify DOS stub file">;

def manifest : F<"manifest">;
def manifest_colon : P<"manifest", "Create manifest file">;
def manifestuac : P<"manifestuac", "User access control">;
def manifestfile : P<"manifestfile", "Manifest file path">;
def manifestdependency : P<"manifestdependency",
                           "Attributes for <dependency> in manifest file">;

// We cannot use multiclass P because class name "incl" is different
// from its command line option name. We do this because "include" is
// a reserved keyword in tablegen.
def incl : Joined<["/", "-"], "include:">,
    HelpText<"Force symbol to be added to symbol table as undefined one">;

// "def" is also a keyword.
def deffile : Joined<["/", "-"], "def:">,
    HelpText<"Use module-definition file">;

def nodefaultlib_all : F<"nodefaultlib">;
def noentry : F<"noentry">;
def dll : F<"dll">;
def verbose : F<"verbose">;
def debug : F<"debug">;
def swaprun_cd : F<"swaprun:cd">;
def swaprun_net : F<"swaprun:net">;

def force : F<"force">,
    HelpText<"Allow undefined symbols when creating executables">;
def force_unresolved : F<"force:unresolved">;

def ref : F<"opt:ref">;
def ref_no : F<"opt:noref">,
    HelpText<"Keep unreferenced symbols to be included to the output">;

defm nxcompat : B<"nxcompat", "Disable data execution provention">;
defm largeaddressaware : B<"largeaddressaware", "Disable large addresses">;
defm allowbind: B<"allowbind", "Disable DLL binding">;
defm fixed    : B<"fixed", "Enable base relocations">;
defm tsaware  : B<"tsaware", "Create non-Terminal Server aware executable">;
defm allowisolation : B<"allowisolation", "Set NO_ISOLATION bit">;
defm dynamicbase : B<"dynamicbase",
     "Disable address space layout randomization">;
defm safeseh : B<"safeseh", "Produce an image with Safe Exception Handler">;

def help : F<"help">;
def help_q : Flag<["/?", "-?"], "">, Alias<help>;

def DASH_DASH : Option<["--"], "", KIND_REMAINING_ARGS>;

//==============================================================================
// The flags below do nothing. They are defined only for link.exe compatibility.
//==============================================================================

class QF<string name> : Joined<["/", "-", "-?"], name#":">;

multiclass QB<string name> {
  def "" : F<name>;
  def _no : F<name#":no">;
}

def functionpadmin : F<"functionpadmin">;
def ignoreidl : F<"ignoreidl">;
def incremental : F<"incremental">;
def no_incremental : F<"incremental:no">;
def nologo : F<"nologo">;

def delay : QF<"delay">;
def delayload : QF<"delayload">;
def errorreport : QF<"errorreport">;
def idlout : QF<"idlout">;
def implib : QF<"implib">;
def pdb : QF<"pdb">;
def pdbaltpath : QF<"pdbaltpath">;
def tlbid : QF<"tlbid">;
def tlbout : QF<"tlbout">;
def verbose_all : QF<"verbose">;

defm wx : QB<"wx">;
