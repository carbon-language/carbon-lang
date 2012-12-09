include "llvm/Option/OptParser.td"

def target : Separate<["-"], "target">, HelpText<"Target triple to link for">;

def output : Joined<["-"], "output=">;
def entry : Joined<["-"], "entry=">;
def relocatable : Flag<["-"], "relocatable">;

def OCTOTHORPE_OCTOTHORPE_OCTOTHORPE : Flag<["-"], "###">;
