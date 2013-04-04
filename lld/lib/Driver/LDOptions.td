include "llvm/Option/OptParser.td"

def target : Separate<["-"], "target">,     
    MetaVarName<"<triple>">,
    HelpText<"Target triple to link for">;
  
def mllvm : Separate<["-"], "mllvm">, HelpText<"Options to pass to LLVM">;


def entry : Joined<["--"], "entry=">,
    HelpText<"Name of entry point symbol">;
def entry_e : Separate<["-"], "e">, Alias<entry>;

def output : Separate<["-"], "o">, 
    MetaVarName<"<path>">,
    HelpText<"Path to file to write output">;
    
def relocatable : Flag<["--"], "relocatable">;
def relocatable_r : Flag<["-"], "r">, Alias<relocatable>;

def dynamic_linker : Separate<["-"], "dynamic-linker">;


def m : Separate<["-"], "m">;
def z : Separate<["-"], "z">;
def t : Flag<["-"], "t">;
def rpath : Separate<["-"], "rpath">;
def soname : Separate<["-"], "soname">;

def static : Flag<["-"], "static">;
def shared : Flag<["-"], "shared">;
def start_group : Flag<["--"], "start-group">;
def end_group : Flag<["--"], "end-group">;
def build_id : Flag<["--"], "build-id">;
def eh_frame_hdr : Flag<["--"], "eh-frame-hdr">;

def help : Flag<["--"], "help">;

def L : Joined<["-"], "L">, 
    MetaVarName<"<dir>">,
    HelpText<"Directory to search for libraries">;
def l : Joined<["-"], "l">, 
    MetaVarName<"<libName>">,
    HelpText<"Root name of library to use">;
def hash_style : Joined <["--"], "hash-style=">;

def noinhibit_exec : Flag<["--"], "noinhibit-exec">,
    HelpText<"Retain the executable output file whenever it is still usable">;

def merge_strings : Flag<["--"], "merge-strings">,
    HelpText<"Merge common strings across mergeable sections">;


// extensions
def emit_yaml : Flag<["-"], "emit-yaml">, 
    HelpText<"Write YAML instead of ELF">;
def force_load : Flag<["--"], "force-load">,
    HelpText<"Force load of all members in all static libraries">;
