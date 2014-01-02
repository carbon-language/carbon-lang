include "llvm/Option/OptParser.td"

//===----------------------------------------------------------------------===//
/// Utility Functions
//===----------------------------------------------------------------------===//
// Single and multiple dash options combined
multiclass smDash<string opt1, string opt2, string help> {
  // Option
  def "" : Separate<["-"], opt1>, HelpText<help>;
  def opt1_eq : Separate<["-"], opt1#"=">,
      Alias<!cast<Option>(opt1)>;
  // Compatibility aliases
  def opt2_dashdash : Separate<["--"], opt2>,
      Alias<!cast<Option>(opt1)>;
  def opt2_dashdash_eq : Separate<["--"], opt2#"=">,
      Alias<!cast<Option>(opt1)>;
}

// Support -<option>,-<option>=
multiclass dashEq<string opt1, string opt2, string help> {
  // Option
  def "" : Separate<["-"], opt1>, HelpText<help>;
  // Compatibility aliases
  def opt2_eq : Separate<["-"], opt2#"=">,
      Alias<!cast<Option>(opt1)>;
}

//===----------------------------------------------------------------------===//
/// Output Kinds
//===----------------------------------------------------------------------===//
def grp_kind : OptionGroup<"outs">,
     HelpText<"OUTPUT KIND">;
def relocatable : Flag<["-"], "r">,
     HelpText<"Create relocatable object file">, Group<grp_kind>;
def static : Flag<["-"], "static">,
     HelpText<"Create static executable">, Group<grp_kind>;
def dynamic : Flag<["-"], "dynamic">,
     HelpText<"Create dynamic executable (default)">,Group<grp_kind>;
def shared : Flag<["-"], "shared">,
     HelpText<"Create dynamic library">, Group<grp_kind>;

// output kinds - compatibility aliases
def Bstatic : Flag<["-"], "Bstatic">, Alias<static>;
def Bshareable : Flag<["-"], "Bshareable">, Alias<shared>;

//===----------------------------------------------------------------------===//
/// General Options
//===----------------------------------------------------------------------===//
def grp_general : OptionGroup<"opts">,
     HelpText<"GENERAL OPTIONS">;
def target : Separate<["-"], "target">, MetaVarName<"<triple>">,
     HelpText<"Target triple to link for">,
     Group<grp_general>;
def output : Separate<["-"], "o">, MetaVarName<"<path>">,
     HelpText<"Path to file to write output">,
     Group<grp_general>;
def m : Separate<["-"], "m">, MetaVarName<"<emulation>">,
     HelpText<"Select target emulation">,
     Group<grp_general>;
def build_id : Flag<["--"], "build-id">,
     HelpText<"Request creation of \".note.gnu.build-id\" ELF note section">,
     Group<grp_general>;
def sysroot : Joined<["--"], "sysroot=">,
    HelpText<"Set the system root">,
    Group<grp_general>;


//===----------------------------------------------------------------------===//
/// Executable Options
//===----------------------------------------------------------------------===//
def grp_main : OptionGroup<"opts">,
     HelpText<"EXECUTABLE OPTIONS">;
def L : Joined<["-"], "L">, MetaVarName<"<dir>">,
     HelpText<"Directory to search for libraries">,
     Group<grp_main>;
def l : Joined<["-"], "l">, MetaVarName<"<libName>">,
     HelpText<"Root name of library to use">,
     Group<grp_main>;
def noinhibit_exec : Flag<["--"], "noinhibit-exec">,
     HelpText<"Retain the executable output file whenever"
              " it is still usable">,
     Group<grp_main>;
defm e : smDash<"e", "entry",
      "Name of entry point symbol">,
      Group<grp_main>;
defm init: dashEq<"init", "init",
      "Specify an initializer function">,
      Group<grp_main>;
defm fini: dashEq<"fini", "fini",
      "Specify a finalizer function">,
      Group<grp_main>;
def whole_archive: Flag<["--"], "whole-archive">,
     HelpText<"Force load of all members in a static library">,
     Group<grp_main>;
def no_whole_archive: Flag<["--"], "no-whole-archive">,
     HelpText<"Restores the default behavior of loading archive members">,
     Group<grp_main>;
def nostdlib : Flag<["-"], "nostdlib">,
     HelpText<"Disable default search path for libraries">,
     Group<grp_main>;

//===----------------------------------------------------------------------===//
/// Static Executable Options
//===----------------------------------------------------------------------===//
def grp_staticexec : OptionGroup<"opts">,
     HelpText<"STATIC EXECUTABLE OPTIONS">;
def nmagic : Flag<["--"], "nmagic">,
     HelpText<"Turn off page alignment of sections,"
              " and disable linking against shared libraries">,
     Group<grp_staticexec>;
def omagic : Flag<["--"], "omagic">,
     HelpText<"Set the text and data sections to be readable and writable."
              " Also, do not page-align the data segment, and"
              " disable linking against shared libraries.">,
     Group<grp_staticexec>;
def no_omagic : Flag<["--"], "no-omagic">,
     HelpText<"This option negates most of the effects of the -N option."
              "Disable linking with shared libraries">,
     Group<grp_staticexec>;
// Compatible Aliases
def nmagic_alias : Flag<["-"], "n">,
     Alias<nmagic>;
def omagic_alias : Flag<["-"], "N">,
     Alias<omagic>;

//===----------------------------------------------------------------------===//
/// Dynamic Library/Executable Options
//===----------------------------------------------------------------------===//
def grp_dynlibexec : OptionGroup<"opts">,
     HelpText<"DYNAMIC LIBRARY/EXECUTABLE OPTIONS">;
def dynamic_linker : Joined<["--"], "dynamic-linker=">,
    HelpText<"Set the path to the dynamic linker">, Group<grp_dynlibexec>;
// Executable options - compatibility aliases
def dynamic_linker_alias : Separate<["-"], "dynamic-linker">,
    Alias<dynamic_linker>;
def rpath : Separate<["-"], "rpath">,
     HelpText<"Add a directory to the runtime library search path">,
     Group<grp_dynlibexec>;
def rpath_link : Separate<["-"], "rpath-link">,
     HelpText<"Specifies the first set of directories to search">,
     Group<grp_dynlibexec>;

//===----------------------------------------------------------------------===//
/// Dynamic Library Options
//===----------------------------------------------------------------------===//
def grp_dynlib : OptionGroup<"opts">,
     HelpText<"DYNAMIC LIBRARY OPTIONS">;
def soname : Separate<["-"], "soname">,
     HelpText<"Set the internal DT_SONAME field to the specified name">,
     Group<grp_dynlib>;

//===----------------------------------------------------------------------===//
/// Resolver Options
//===----------------------------------------------------------------------===//
def grp_resolveropt : OptionGroup<"opts">,
     HelpText<"SYMBOL RESOLUTION OPTIONS">;
defm u : smDash<"u", "undefined",
      "Force symbol to be entered in the output file"
      " as an undefined symbol">,
      Group<grp_resolveropt>;
def start_group : Flag<["--"], "start-group">,
     HelpText<"Start a group">,
     Group<grp_resolveropt>;
def alias_start_group: Flag<["-"], "(">,
      Alias<start_group>;
def end_group : Flag<["--"], "end-group">,
     HelpText<"End a group">,
     Group<grp_resolveropt>;
def alias_end_group: Flag<["-"], ")">,
      Alias<end_group>;
def as_needed : Flag<["--"], "as-needed">,
     HelpText<"This option affects ELF DT_NEEDED tags for "
              "dynamic libraries mentioned on the command line">,
     Group<grp_resolveropt>;
def no_as_needed : Flag<["--"], "no-as-needed">,
     HelpText<"This option restores the default behavior"
              " of adding DT_NEEDED entries">,
     Group<grp_resolveropt>;
def no_allow_shlib_undefs : Flag<["--"], "no-allow-shlib-undefined">,
     HelpText<"Do not allow undefined symbols from dynamic"
              " library when creating executables">,
     Group<grp_resolveropt>;
def allow_shlib_undefs : Flag<["--"], "allow-shlib-undefined">,
     HelpText<"Allow undefined symbols from dynamic"
             " library when creating executables">,
     Group<grp_resolveropt>;
def use_shlib_undefs: Flag<["--"], "use-shlib-undefines">,
    HelpText<"Resolve undefined symbols from dynamic libraries">,
    Group<grp_resolveropt>;

//===----------------------------------------------------------------------===//
/// Custom Options 
//===----------------------------------------------------------------------===//
def grp_customopts : OptionGroup<"opts">,
     HelpText<"CUSTOM OPTIONS">;
def z : Separate<["-"], "z">,
     HelpText<"Linker Option extensions">,
     Group<grp_customopts>;

//===----------------------------------------------------------------------===//
/// Optimization Options
//===----------------------------------------------------------------------===//
def grp_opts : OptionGroup<"opts">,
     HelpText<"OPTIMIZATIONS">;
def mllvm : Separate<["-"], "mllvm">,
     HelpText<"Options to pass to LLVM during LTO">, Group<grp_opts>;
def hash_style : Joined <["--"], "hash-style=">,
     HelpText<"Set the type of linker's hash table(s)">,
     Group<grp_opts>;
def merge_strings : Flag<["--"], "merge-strings">,
     HelpText<"Merge common strings across mergeable sections">,
     Group<grp_opts>;
def eh_frame_hdr : Flag<["--"], "eh-frame-hdr">,
     HelpText<"Request creation of .eh_frame_hdr section and ELF "
              " PT_GNU_EH_FRAME segment header">,
     Group<grp_opts>;

//===----------------------------------------------------------------------===//
/// Tracing Options
//===----------------------------------------------------------------------===//
def grp_tracingopts : OptionGroup<"opts">,
     HelpText<"TRACING OPTIONS">;
def t : Flag<["-"], "t">,
     HelpText<"Print the names of the input files as ld processes them">,
     Group<grp_tracingopts>;

//===----------------------------------------------------------------------===//
/// Extensions
//===----------------------------------------------------------------------===//
def grp_extns : OptionGroup<"opts">,
     HelpText<"Extensions">;
def output_filetype: Separate<["--"], "output-filetype">,
      HelpText<"Specify what type of output file that lld creates, YAML/Native">,
      Group<grp_extns>;
def alias_output_filetype: Joined<["--"], "output-filetype=">,
      Alias<output_filetype>;

//===----------------------------------------------------------------------===//
/// Help
//===----------------------------------------------------------------------===//
def help : Flag<["--"], "help">,
     HelpText<"Display this help message">;

