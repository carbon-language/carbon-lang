include "llvm/Option/OptParser.td"

// link.exe accepts options starting with either a dash or a slash.

def mllvm : Separate<["-", "/"], "mllvm">,
    HelpText<"Options to pass to LLVM">;

def out : Separate<["-", "/"], "out">,
    HelpText<"Path to file to write output">;
def out_c : Joined<["-", "/"], "out:">, Alias<out>;

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

def help : Flag<["-", "/"], "help">;
def help_q : Flag<["-", "/"], "?">, Alias<help>;
