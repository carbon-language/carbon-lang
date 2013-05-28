include "llvm/Option/OptParser.td"

// link.exe accepts options starting with either a dash or a slash.

def mllvm : Separate<["-", "/"], "mllvm">,
    HelpText<"Options to pass to LLVM">;

def help : Flag<["-", "/"], "help">;
def help_q : Flag<["-", "/"], "?">, Alias<help>;
