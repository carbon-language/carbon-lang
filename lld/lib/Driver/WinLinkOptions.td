include "llvm/Option/OptParser.td"

// link.exe accepts options starting with either a dash or a slash.

def mllvm : Separate<["-", "/"], "mllvm">,
    HelpText<"Options to pass to LLVM">;

def out : Separate<["-", "/"], "out">,
    HelpText<"Path to file to write output">;
def out_c : Joined<["-", "/"], "out:">, Alias<out>;

def defaultlib : Separate<["-", "/"], "defaultlib">,
    HelpText<"Add the library to the list of input files">;
def defaultlib_c : Joined<["-", "/"], "defaultlib:">, Alias<defaultlib>;

def base : Separate<["-", "/"], "base">,
    HelpText<"Base address of the program">;
def base_c: Joined<["-", "/"], "base:">, Alias<base>;

def stack : Separate<["-", "/"], "stack">,
    HelpText<"Size of the stack">;
def stack_c: Joined<["-", "/"], "stack:">, Alias<stack>;

def heap : Separate<["-", "/"], "heap">,
    HelpText<"Size of the heap">;
def heap_c: Joined<["-", "/"], "heap:">, Alias<heap>;

def subsystem : Separate<["-", "/"], "subsystem">,
    HelpText<"Specify subsystem">;
def subsystem_c: Joined<["-", "/"], "subsystem:">, Alias<subsystem>;

def entry : Separate<["-", "/"], "entry">,
    HelpText<"Name of entry point symbol">;
def entry_c: Joined<["-", "/"], "entry:">, Alias<entry>;

def libpath : Separate<["-", "/"], "libpath">,
    HelpText<"Additional library search path">;
def libpath_c: Joined<["-", "/"], "libpath:">, Alias<libpath>;

def force : Flag<["-", "/"], "force">,
    HelpText<"Allow undefined symbols when creating executables">;

def nxcompat : Flag<["-", "/"], "nxcompat">,
    HelpText<"Enable data execution provention">;
def no_nxcompat : Flag<["-", "/"], "nxcompat:no">,
    HelpText<"Disable data execution provention">;

def largeaddressaware : Flag<["-", "/"], "largeaddressaware">,
    HelpText<"Enable large addresses">;
def no_largeaddressaware : Flag<["-", "/"], "largeaddressaware:no">,
    HelpText<"Disable large addresses">;

def fixed : Flag<["-", "/"], "fixed">,
    HelpText<"Disable base relocations">;
def no_fixed : Flag<["-", "/"], "fixed:no">,
    HelpText<"Enable base relocations">;

def tsaware : Flag<["-", "/"], "tsaware">,
    HelpText<"Create Terminal Server aware executable">;
def no_tsaware : Flag<["-", "/"], "tsaware:no">,
    HelpText<"Create non-Terminal Server aware executable">;

def help : Flag<["-", "/"], "help">;
def help_q : Flag<["-", "/"], "?">, Alias<help>;
