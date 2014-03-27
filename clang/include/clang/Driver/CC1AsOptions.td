//===--- CC1AsOptions.td - Options for clang -cc1as -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by clang -cc1as.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "llvm/Option/OptParser.td"

//===----------------------------------------------------------------------===//
// Target Options
//===----------------------------------------------------------------------===//

def triple : Separate<["-"], "triple">,
  HelpText<"Specify target triple (e.g. x86_64-pc-linux-gnu)">;
def target_cpu : Separate<["-"], "target-cpu">,
  HelpText<"Target a specific cpu type">;
def target_feature : Separate<["-"], "target-feature">,
  HelpText<"Target specific attributes">;

//===----------------------------------------------------------------------===//
// Language Options
//===----------------------------------------------------------------------===//

def I : JoinedOrSeparate<["-"], "I">, MetaVarName<"<directory>">,
  HelpText<"Add directory to include search path">;
def n : Flag<["-"], "n">,
  HelpText<"Don't automatically start assembly file with a text section">;
def msave_temp_labels : Flag<["-"], "msave-temp-labels">,
  HelpText<"Save temporary labels in the symbol table. "
           "Note this may change .s semantics, it should almost never be used "
           "on compiler generated code!">;
def main_file_name : Separate<["-"], "main-file-name">,
  HelpText<"Main file name to use for debug info">;

//===----------------------------------------------------------------------===//
// Frontend Options
//===----------------------------------------------------------------------===//

def o : Separate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Specify output file">;

def filetype : Separate<["-"], "filetype">,
    HelpText<"Specify the output file type ('asm', 'null', or 'obj')">;

def help : Flag<["-", "--"], "help">,
  HelpText<"Print this help text">;

def version : Flag<["-", "--"], "version">,
  HelpText<"Print the assembler version">;
def v : Flag<["-"], "v">, Alias<version>;

// Generic forwarding to LLVM options. This should only be used for debugging
// and experimental features.
def mllvm : Separate<["-"], "mllvm">,
  HelpText<"Additional arguments to forward to LLVM's option processing">;

//===----------------------------------------------------------------------===//
// Transliterate Options
//===----------------------------------------------------------------------===//

def output_asm_variant : Separate<["-"], "output-asm-variant">,
    HelpText<"Select the asm variant index to use for output">;
def show_encoding : Flag<["-"], "show-encoding">,
    HelpText<"Show instruction encoding information in transliterate mode">;
def show_inst : Flag<["-"], "show-inst">,
    HelpText<"Show internal instruction representation in transliterate mode">;

//===----------------------------------------------------------------------===//
// Assemble Options
//===----------------------------------------------------------------------===//

def mrelax_all : Flag<["-"], "mrelax-all">,
    HelpText<"Relax all fixups (for performance testing)">;

def mno_exec_stack : Flag<["-"], "mnoexecstack">,
    HelpText<"Mark the file as not needing an executable stack">;

def compress_debug_sections : Flag<["-"], "compress-debug-sections">,
    HelpText<"Compress DWARF debug sections using zlib">;

def g : Flag<["-"], "g">, HelpText<"Generate source level debug information">;

def fdebug_compilation_dir : Separate<["-"], "fdebug-compilation-dir">,
  HelpText<"The compilation directory to embed in the debug info.">;

def dwarf_debug_flags : Separate<["-"], "dwarf-debug-flags">,
  HelpText<"The string to embed in the Dwarf debug flags record.">;

def dwarf_debug_producer : Separate<["-"], "dwarf-debug-producer">,
  HelpText<"The string to embed in the Dwarf debug AT_producer record.">;
