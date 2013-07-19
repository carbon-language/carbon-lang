include "llvm/Option/OptParser.td"


// output kinds
def grp_kind : OptionGroup<"outs">, HelpText<"OUTPUT KIND">;
def relocatable : Flag<["-"], "r">, 
                HelpText<"Create relocatable object file">, Group<grp_kind>;
def static : Flag<["-"], "static">, 
                HelpText<"Create static executable">, Group<grp_kind>;
def dynamic : Flag<["-"], "dynamic">, 
                HelpText<"Create dynamic executable (default)">,Group<grp_kind>;
def dylib : Flag<["-"], "dylib">, 
                HelpText<"Create dynamic library">, Group<grp_kind>;
def bundle : Flag<["-"], "bundle">, 
                HelpText<"Create dynamic bundle">, Group<grp_kind>;
def execute : Flag<["-"], "execute">, 
                HelpText<"Create main executable (default)">, Group<grp_kind>;
def preload : Flag<["-"], "preload">, 
                HelpText<"Create preload">, Group<grp_kind>;

// optimizations
def grp_opts : OptionGroup<"opts">, HelpText<"OPTIMIZATIONS">;
def dead_strip : Flag<["-"], "dead_strip">,
                HelpText<"Remove unreference code and data">, Group<grp_opts>;
def macosx_version_min : Separate<["-"], "macosx_version_min">, 
                HelpText<"Minimum Mac OS X version">, Group<grp_opts>;
def ios_version_min : Separate<["-"], "ios_version_min">, 
                HelpText<"Minimum iOS version">, Group<grp_opts>;
def ios_simulator_version_min : Separate<["-"], "ios_simulator_version_min">, 
                HelpText<"Minimum iOS simulator version">, Group<grp_opts>;
def mllvm : Separate<["-"], "mllvm">, 
                HelpText<"Options to pass to LLVM during LTO">, Group<grp_opts>;

// main executable options
def grp_main : OptionGroup<"opts">, HelpText<"MAIN EXECUTABLE OPTIONS">;
def entry : Separate<["-"], "e">, HelpText<"entry symbol name">,Group<grp_main>;

// dylib executable options
def grp_dylib : OptionGroup<"opts">, HelpText<"DYLIB EXECUTABLE OPTIONS">;

// library options
def grp_libs : OptionGroup<"libs">, HelpText<"LIBRARY OPTIONS">;
def L : Joined<["-"], "L">,
        HelpText<"Add directory to library search path">, Group<grp_libs>;
def all_load : Flag<["-"], "all_load">,
        HelpText<"Forces all members of all static libraries to be loaded">, 
        Group<grp_libs>;
  

// general options
def output : Separate<["-"], "o">, HelpText<"Output file path">;
def arch : Separate<["-"], "arch">, HelpText<"Architecture to link">;

// extras
def help : Flag<["-"], "help">;
