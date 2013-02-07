include "llvm/Option/OptParser.td"

def flavor : Separate<["-"], "flavor">;
def target : Separate<["-"], "target">, HelpText<"Target triple to link for">;
def mllvm : Separate<["-"], "mllvm">, HelpText<"Options to pass to LLVM">;


def entry : Joined<["--"], "entry=">;
def entry_e : Separate<["-"], "e">, Alias<entry>;

def output : Joined<["--"], "output=">;
def output_e : Separate<["-"], "o">, Alias<output>;

def relocatable : Flag<["--"], "relocatable">;
def relocatable_r : Flag<["-"], "r">, Alias<relocatable>;

def OCTOTHORPE_OCTOTHORPE_OCTOTHORPE : Flag<["-"], "###">;

def emit_yaml : Flag<["-"], "emit-yaml">;

def m : Separate<["-"], "m">;
def z : Separate<["-"], "z">;

def static : Flag<["-"], "static">;
def start_group : Flag<["--"], "start-group">;
def end_group : Flag<["--"], "end-group">;
def build_id : Flag<["--"], "build-id">;

def L : Joined<["-"], "L">;
def l : Joined<["-"], "l">;
def hash_style : Joined <["--"], "hash-style=">;

def noinhibit_exec : Flag<["--"], "noinhibit-exec">,
    HelpText<"Retain the executable output file whenever it is still usable">;
