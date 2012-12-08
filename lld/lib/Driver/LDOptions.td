include "llvm/Option/OptParser.td"

def flavor : Separate<["-"], "flavor">;
def target : Separate<["-"], "target">, HelpText<"Target triple to link for">;

def entry : Joined<["--"], "entry=">;
def entry_e : Separate<["-"], "e">, Alias<entry>;

def output : Joined<["--"], "output=">;
def output_e : Separate<["-"], "o">, Alias<output>;

def relocatable : Flag<["--"], "relocatable">;
def relocatable_r : Flag<["-"], "r">, Alias<relocatable>;
