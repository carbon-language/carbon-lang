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

def dynamic_linker : Joined<["--"], "dynamic-linker=">,
    HelpText<"Set the path to the dynamic linker">;
def dynamic_linker_alias : Separate<["-"], "dynamic-linker">, Alias<dynamic_linker>;

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


def no_allow_shlib_undefs : Flag<["--"], "no-allow-shlib-undefined">,
    HelpText<"Donot allow undefined symbols from dynamic library when creating executables">;

def allow_shlib_undefs : Flag<["--"], "allow-shlib-undefined">,
    HelpText<"Allow undefined symbols from dynamic library when creating executables">;

def use_shlib_undefs: Flag<["--"], "use-shlib-undefines">,
    HelpText<"Resolve undefined symbols from dynamic libraries">;

def nmagic : Flag<["--"], "nmagic">,
    HelpText<"Turn off page alignment of sections, and disable linking against shared libraries">;
def nmagic_alias : Flag<["-"], "n">, Alias<nmagic>;

def omagic : Flag<["--"], "omagic">,
    HelpText<"Set the text and data sections to be readable and writable.  Also, do not page-align the data segment, and disable linking against shared libraries.">;
def omagic_alias : Flag<["-"], "N">, Alias<omagic>;

def no_omagic : Flag<["--"], "no-omagic">,
    HelpText<"This option negates most of the effects of the -N option. Disable linking with shared libraries">;

// extensions
def emit_yaml : Flag<["-"], "emit-yaml">, 
    HelpText<"Write YAML instead of ELF">;
def force_load : Flag<["--"], "force-load">,
    HelpText<"Force load of all members in all static libraries">;
