include "llvm/Option/OptParser.td"

def target : Separate<["-"], "target">,     
    MetaVarName<"<triple>">,
    HelpText<"Target triple to link for">;
  
def mllvm : Separate<["-"], "mllvm">,
    HelpText<"Options to pass to LLVM">;

// Single and multiple dash options combined
multiclass smDash<string opt1, string opt2, string help> {
  def "" : Separate<["-"], opt1>, HelpText<help>;
  def opt1_eq : Separate<["-"], opt1#"=">,
      Alias<!cast<Option>(opt1)>;
  def opt2_dashdash : Separate<["--"], opt2>,
      Alias<!cast<Option>(opt1)>;
  def opt2_dashdash_eq : Separate<["--"], opt2#"=">,
      Alias<!cast<Option>(opt1)>;
}

defm e : smDash<"e", "entry",
         "Name of entry point symbol">;

def output : Separate<["-"], "o">, 
    MetaVarName<"<path>">,
    HelpText<"Path to file to write output">;
    
def relocatable : Flag<["--"], "relocatable">;
def relocatable_r : Flag<["-"], "r">, Alias<relocatable>;

def dynamic_linker : Joined<["--"], "dynamic-linker=">,
    HelpText<"Set the path to the dynamic linker">;
def dynamic_linker_alias : Separate<["-"], "dynamic-linker">,
    Alias<dynamic_linker>;

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
    HelpText<"Retain the executable output file whenever"
             " it is still usable">;

def merge_strings : Flag<["--"], "merge-strings">,
    HelpText<"Merge common strings across mergeable sections">;


def no_allow_shlib_undefs : Flag<["--"], "no-allow-shlib-undefined">,
    HelpText<"Donot allow undefined symbols from dynamic"
             " library when creating executables">;

def allow_shlib_undefs : Flag<["--"], "allow-shlib-undefined">,
    HelpText<"Allow undefined symbols from dynamic"
             " library when creating executables">;

def use_shlib_undefs: Flag<["--"], "use-shlib-undefines">,
    HelpText<"Resolve undefined symbols from dynamic libraries">;

def nmagic : Flag<["--"], "nmagic">,
    HelpText<"Turn off page alignment of sections,"
             " and disable linking against shared libraries">;
def nmagic_alias : Flag<["-"], "n">, Alias<nmagic>;

def omagic : Flag<["--"], "omagic">,
    HelpText<"Set the text and data sections to be readable and writable."
             " Also, do not page-align the data segment, and"
             " disable linking against shared libraries.">;

def omagic_alias : Flag<["-"], "N">, Alias<omagic>;

def no_omagic : Flag<["--"], "no-omagic">,
    HelpText<"This option negates most of the effects of the -N option."
             "Disable linking with shared libraries">;

defm u : smDash<"u", "undefined",
         "Force symbol to be entered in the output file"
         " as an undefined symbol">;

def as_needed : Flag<["--"], "as-needed">,
    HelpText<"This option affects ELF DT_NEEDED tags for "
             "dynamic libraries mentioned on the command line">;
def no_as_needed : Flag<["--"], "no-as-needed">,
    HelpText<"This option restores the default behavior"
             " of adding DT_NEEDED entries">;

// extensions
def emit_yaml : Flag<["-"], "emit-yaml">, 
    HelpText<"Write YAML instead of ELF">;
def whole_archive: Flag<["--"], "whole-archive">,
    HelpText<"Force load of all members in a static library">;
def no_whole_archive: Flag<["--"], "no-whole-archive">,
    HelpText<"Restores the default behavior of loading archive members">;

def all_load : Flag<["-"], "all_load">,
        HelpText<"Forces all members of all static libraries to be loaded">;
