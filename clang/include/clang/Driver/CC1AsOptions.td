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
include "OptParser.td"

//===----------------------------------------------------------------------===//
// Target Options
//===----------------------------------------------------------------------===//

def triple : Separate<"-triple">,
  HelpText<"Specify target triple (e.g. x86_64-pc-linux-gnu)">;

//===----------------------------------------------------------------------===//
// Language Options
//===----------------------------------------------------------------------===//

def I : JoinedOrSeparate<"-I">, MetaVarName<"<directory>">,
  HelpText<"Add directory to include search path">;
def n : Flag<"-n">,
  HelpText<"Don't automatically start assembly file with a text section">;

//===----------------------------------------------------------------------===//
// Frontend Options
//===----------------------------------------------------------------------===//

def o : Separate<"-o">, MetaVarName<"<path>">, HelpText<"Specify output file">;

def filetype : Separate<"-filetype">,
    HelpText<"Specify the output file type ('asm', 'null', or 'obj')">;

def help : Flag<"-help">,
  HelpText<"Print this help text">;
def _help : Flag<"--help">, Alias<help>;

def version : Flag<"-version">,
  HelpText<"Print the assembler version">;
def _version : Flag<"--version">, Alias<version>;
def v : Flag<"-v">, Alias<version>;

// Generic forwarding to LLVM options. This should only be used for debugging
// and experimental features.
def mllvm : Separate<"-mllvm">,
  HelpText<"Additional arguments to forward to LLVM's option processing">;

//===----------------------------------------------------------------------===//
// Transliterate Options
//===----------------------------------------------------------------------===//

def output_asm_variant : Separate<"-output-asm-variant">,
    HelpText<"Select the asm variant index to use for output">;
def show_encoding : Flag<"-show-encoding">,
    HelpText<"Show instruction encoding information in transliterate mode">;
def show_inst : Flag<"-show-inst">,
    HelpText<"Show internal instruction representation in transliterate mode">;

//===----------------------------------------------------------------------===//
// Assemble Options
//===----------------------------------------------------------------------===//

def relax_all : Flag<"-relax-all">,
    HelpText<"Relax all fixups (for performance testing)">;

def no_exec_stack : Flag<"--noexecstack">,
    HelpText<"Mark the file as not needing an executable stack">;