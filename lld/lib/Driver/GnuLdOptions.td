include "llvm/Option/OptParser.td"

//===----------------------------------------------------------------------===//
/// Utility Functions
//===----------------------------------------------------------------------===//
// Single and multiple dash options combined
multiclass smDash<string opt1, string opt2, string help> {
  // Option
  def "" : Separate<["-"], opt1>, HelpText<help>;
  def opt1_eq : Joined<["-"], opt1#"=">,
      Alias<!cast<Option>(opt1)>;
  // Compatibility aliases
  def opt2_dashdash : Separate<["--"], opt2>,
      Alias<!cast<Option>(opt1)>;
  def opt2_dashdash_eq : Joined<["--"], opt2#"=">,
      Alias<!cast<Option>(opt1)>;
}

// Support -<option>,-<option>=
multiclass dashEq<string opt1, string opt2, string help> {
  // Option
  def "" : Separate<["-"], opt1>, HelpText<help>;
  // Compatibility aliases
  def opt2_eq : Joined<["-"], opt2#"=">,
      Alias<!cast<Option>(opt1)>;
}

// Support --<option>,--<option>=
multiclass mDashEq<string opt1, string help> {
  // Option
  def "" : Separate<["--"], opt1>, HelpText<help>;
  // Compatibility aliases
  def opt2_eq : Joined<["--"], opt1#"=">,
      Alias<!cast<Option>(opt1)>;
}

//===----------------------------------------------------------------------===//
/// LLVM and Target options
//===----------------------------------------------------------------------===//
def grp_llvmtarget : OptionGroup<"opts">,
     HelpText<"LLVM and Target Options">;
def mllvm : Separate<["-"], "mllvm">,
     HelpText<"Options to pass to LLVM">, Group<grp_llvmtarget>;
def target : Separate<["-"], "target">, MetaVarName<"<triple>">,
     HelpText<"Target triple to link for">,
     Group<grp_llvmtarget>;

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
def output : Separate<["-"], "o">, MetaVarName<"<path>">,
     HelpText<"Path to file to write output">,
     Group<grp_general>;
def m : JoinedOrSeparate<["-"], "m">, MetaVarName<"<emulation>">,
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
def image_base : Separate<["--"], "image-base">,
     HelpText<"Set the base address">,
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
def dynamic_linker_alias : Separate<["-", "--"], "dynamic-linker">,
    Alias<dynamic_linker>;
defm rpath : dashEq<"rpath", "rpath",
     "Add a directory to the runtime library search path">,
     Group<grp_dynlibexec>;
def rpath_link : Separate<["-"], "rpath-link">,
     HelpText<"Specifies the first set of directories to search">,
     Group<grp_dynlibexec>;
def export_dynamic : Flag<["-", "--"], "export-dynamic">,
     HelpText<"Add all symbols to the dynamic symbol table"
              " when creating executables">,
     Group<grp_main>;
def alias_export_dynamic: Flag<["-"], "E">,
      Alias<export_dynamic>;
def no_export_dynamic : Flag<["--"], "no-export-dynamic">,
     Group<grp_main>;

//===----------------------------------------------------------------------===//
/// Dynamic Library Options
//===----------------------------------------------------------------------===//
def grp_dynlib : OptionGroup<"opts">,
     HelpText<"DYNAMIC LIBRARY OPTIONS">;
def soname : Joined<["-", "--"], "soname=">,
     HelpText<"Set the internal DT_SONAME field to the specified name">,
     Group<grp_dynlib>;
def soname_separate : Separate<["-", "--"], "soname">, Alias<soname>;
def soname_h : Separate<["-"], "h">, Alias<soname>;

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
def allow_shlib_undefs : Flag<["-", "--"], "allow-shlib-undefined">,
     HelpText<"Allow undefined symbols from dynamic"
             " library when creating executables">,
     Group<grp_resolveropt>;
def use_shlib_undefs: Flag<["--"], "use-shlib-undefines">,
    HelpText<"Resolve undefined symbols from dynamic libraries">,
    Group<grp_resolveropt>;
def allow_multiple_definition: Flag<["--"], "allow-multiple-definition">,
    HelpText<"Allow multiple definitions">,
    Group<grp_resolveropt>;
defm defsym : mDashEq<"defsym",
           "Create a global symbol in the output file "
           "containing the absolute address given by expression">,
           MetaVarName<"symbol=<expression>">,
           Group<grp_resolveropt>;

//===----------------------------------------------------------------------===//
/// Custom Options
//===----------------------------------------------------------------------===//
def grp_customopts : OptionGroup<"opts">,
     HelpText<"CUSTOM OPTIONS">;
def disable_newdtags: Flag<["--"], "disable-new-dtags">,
     HelpText<"Disable new dynamic tags">,
     Group<grp_customopts>;
def enable_newdtags: Flag<["--"], "enable-new-dtags">,
     HelpText<"Enable new dynamic tags">,
     Group<grp_customopts>;
def rosegment: Flag<["--"], "rosegment">,
     HelpText<"Put read-only non-executable sections in their own segment">,
     Group<grp_customopts>;
def z : Separate<["-"], "z">,
     HelpText<"Linker Option extensions">,
     Group<grp_customopts>;
def no_align_segments: Flag<["--"], "no-align-segments">,
     HelpText<"Don't align ELF segments(virtualaddress/fileoffset) to page boundaries">,
     Group<grp_customopts>;

//===----------------------------------------------------------------------===//
/// Symbol options
//===----------------------------------------------------------------------===//
def grp_symbolopts : OptionGroup<"opts">,
     HelpText<"SYMBOL OPTIONS">;
def demangle : Flag<["--"], "demangle">,
     HelpText<"Demangle C++ symbols">,
     Group<grp_symbolopts>;
def discard_loc : Flag<["--"], "discard-all">,
     HelpText<"Discard all local symbols">,
     Group<grp_symbolopts>;
def alias_discard_loc: Flag<["-"], "x">,
     Alias<discard_loc>;
def discard_temp_loc : Flag<["--"], "discard-locals">,
     HelpText<"Discard temporary local symbols">,
     Group<grp_symbolopts>;
def alias_discard_temp_loc : Flag<["-"], "X">,
     Alias<discard_temp_loc>;
def no_demangle : Flag<["--"], "no-demangle">,
     HelpText<"Dont demangle C++ symbols">,
     Group<grp_symbolopts>;
def strip_all : Flag<["--"], "strip-all">,
     HelpText<"Omit all symbol informations from output">,
     Group<grp_symbolopts>;
def alias_strip_all : Flag<["-"], "s">,
     Alias<strip_all>;
defm wrap : smDash<"wrap", "wrap",
            "Use a wrapper function for symbol. Any "
            " undefined reference to symbol will be resolved to "
            "\"__wrap_symbol\". Any undefined reference to \"__real_symbol\""
            " will be resolved to symbol.">,
            MetaVarName<"<symbol>">,
            Group<grp_symbolopts>;

//===----------------------------------------------------------------------===//
/// Script Options
//===----------------------------------------------------------------------===//
def grp_scriptopts : OptionGroup<"opts">,
     HelpText<"SCRIPT OPTIONS">;
defm T : smDash<"T", "script",
      "Use the given linker script in place of the default script.">,
      Group<grp_scriptopts>;

//===----------------------------------------------------------------------===//
/// Optimization Options
//===----------------------------------------------------------------------===//
def grp_opts : OptionGroup<"opts">,
     HelpText<"OPTIMIZATIONS">;
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
def stats : Flag<["--"], "stats">,
     HelpText<"Print time and memory usage stats">, Group<grp_tracingopts>;

//===----------------------------------------------------------------------===//
/// Extensions
//===----------------------------------------------------------------------===//
def grp_extns : OptionGroup<"opts">,
     HelpText<"Extensions">;
def output_filetype: Separate<["--"], "output-filetype">,
      HelpText<"Specify yaml to create an output in YAML format">,
      Group<grp_extns>;
def alias_output_filetype: Joined<["--"], "output-filetype=">,
      Alias<output_filetype>;

//===----------------------------------------------------------------------===//
/// Target Specific Options
//===----------------------------------------------------------------------===//
def grp_targetopts : OptionGroup<"opts">,
     HelpText<"ARCH SPECIFIC OPTIONS">;

//===----------------------------------------------------------------------===//
/// ARM Target Specific Options
//===----------------------------------------------------------------------===//
def grp_arm_targetopts : OptionGroup<"ARM SPECIFIC OPTIONS">,
     Group<grp_targetopts>;
def target1_rel : Flag<["--"], "target1-rel">,
     Group<grp_arm_targetopts>, HelpText<"Interpret R_ARM_TARGET1 as R_ARM_REL32">;
def target1_abs : Flag<["--"], "target1-abs">,
     Group<grp_arm_targetopts>, HelpText<"Interpret R_ARM_TARGET1 as R_ARM_ABS32">;

//===----------------------------------------------------------------------===//
/// MIPS Target Specific Options
//===----------------------------------------------------------------------===//
def grp_mips_targetopts : OptionGroup<"MIPS SPECIFIC OPTIONS">,
     Group<grp_targetopts>;
def pcrel_eh_reloc : Flag<["-", "--"], "pcrel-eh-reloc">,
     Group<grp_mips_targetopts>,
     HelpText<"Interpret R_MIPS_EH as R_MIPS_PC32">;

//===----------------------------------------------------------------------===//
/// Ignored options
//===----------------------------------------------------------------------===//
def grp_ignored: OptionGroup<"ignored">,
     HelpText<"GNU Options ignored for Compatibility ">;
def dashg : Flag<["-"], "g">,
     HelpText<"Ignored.">,
     Group<grp_ignored>;
def Qy : Flag<["-"], "Qy">,
     HelpText<"Ignored for SVR4 Compatibility">,
     Group<grp_ignored>;
def qmagic : Flag<["-"], "qmagic">,
     HelpText<"Ignored for Linux Compatibility">,
     Group<grp_ignored>;
def G : Separate<["-"], "G">,
     HelpText<"Ignored for MIPS GNU Linker Compatibility">,
     Group<grp_ignored>;

//===----------------------------------------------------------------------===//
/// Help
//===----------------------------------------------------------------------===//
def help : Flag<["--"], "help">,
     HelpText<"Display this help message">;
