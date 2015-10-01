include "llvm/Option/OptParser.td"

def L : Joined<["-"], "L">, MetaVarName<"<dir>">,
  HelpText<"Directory to search for libraries">;

def allow_multiple_definition: Flag<["--"], "allow-multiple-definition">,
  HelpText<"Allow multiple definitions">;

def discard_all : Flag<["-"], "discard-all">,
  HelpText<"Delete all local symbols">;

def discard_locals : Flag<["-"], "discard-locals">,
  HelpText<"Delete temporary local symbols">;

def discard_none : Flag<["-"], "discard-none">,
  HelpText<"Keep all symbols in the symbol table">;

def dynamic_linker : Separate<["-"], "dynamic-linker">,
  HelpText<"Which dynamic linker to use">;

def entry : Separate<["--", "-"], "entry">, MetaVarName<"<entry>">,
  HelpText<"Name of entry point symbol">;

def export_dynamic : Flag<["--"], "export-dynamic">,
  HelpText<"Put symbols in the dynamic symbol table">;

def l : Joined<["-"], "l">, MetaVarName<"<libName>">,
  HelpText<"Root name of library to use">;

def noinhibit_exec : Flag<["--"], "noinhibit-exec">,
  HelpText<"Retain the executable output file whenever it is still usable">;

def output : Separate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;

def rpath : Separate<["-"], "rpath">,
  HelpText<"Add a DT_RUNPATH to the output">;

def shared : Flag<["-"], "shared">,
  HelpText<"Build a shared object">;

def sysroot : Joined<["--"], "sysroot=">,
  HelpText<"Set the system root">;

// Aliases
def alias_L : Joined<["--"], "library-path=">, Alias<L>;
def alias_discard_all: Flag<["-"], "x">, Alias<discard_all>;
def alias_discard_locals: Flag<["-"], "X">, Alias<discard_locals>;
def alias_entry : Separate<["-"], "e">, Alias<entry>;
def alias_l : Joined<["--"], "library=">, Alias<l>;

// Options listed below are silently ignored now.
def as_needed : Flag<["--"], "as-needed">;
def build_id : Flag<["--"], "build-id">;
def eh_frame_hdr : Flag<["--"], "eh-frame-hdr">;
def hash_style : Joined<["--"], "hash-style=">;
def m : Separate<["-"], "m">;
def no_as_needed : Flag<["--"], "no-as-needed">;
def z : Separate<["-"], "z">;
