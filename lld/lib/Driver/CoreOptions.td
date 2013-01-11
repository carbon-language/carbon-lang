include "llvm/Option/OptParser.td"

def flavor : Separate<["-"], "flavor">;
def target : Separate<["-"], "target">, HelpText<"Target triple to link for">;
def mllvm : Separate<["-"], "mllvm">, HelpText<"Options to pass to LLVM">;

def output : Joined<["-"], "output=">;
def entry : Joined<["-"], "entry=">;
def relocatable : Flag<["-"], "relocatable">;

def OCTOTHORPE_OCTOTHORPE_OCTOTHORPE : Flag<["-"], "###">;

def emit_yaml : Flag<["-"], "emit-yaml">;

def noinhibit_exec : Flag<["-"], "noinhibit-exec">, 
    HelpText<"Retain the executable output file whenever it is still usable">;
