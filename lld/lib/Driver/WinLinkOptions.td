include "llvm/Option/OptParser.td"

// link.exe accepts options starting with either a dash or a slash.

def mllvm : Separate<["-", "/"], "mllvm">,
    HelpText<"Options to pass to LLVM">;

def out : Separate<["-", "/"], "out">,
    HelpText<"Path to file to write output">;
def out_c : Joined<["-", "/"], "out:">, Alias<out>;

def subsystem : Separate<["-", "/"], "subsystem">,
    HelpText<"Specify subsystem">;
def subsystem_c: Joined<["-", "/"], "subsystem:">, Alias<subsystem>;

def entry : Separate<["-", "/"], "entry">,
    HelpText<"Name of entry point symbol">;
def entry_c: Joined<["-", "/"], "entry:">, Alias<entry>;

def help : Flag<["-", "/"], "help">;
def help_q : Flag<["-", "/"], "?">, Alias<help>;
