include "llvm/Option/OptParser.td"

// Select an optional flavor
def flavor: Separate<["-"], "flavor">,
     HelpText<"Flavor for linking, options are gnu/darwin/link">;

// Select the core flavor
def core : Flag<["-"], "core">,
     HelpText<"CORE linking">;

def target: Separate<["-"], "target">,
     HelpText<"Select the target">;

// Help message
def help : Flag<["-"], "help">,
     HelpText<"Display this help message">;
