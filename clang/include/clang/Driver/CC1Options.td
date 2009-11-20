//===--- CC1Options.td - Options for clang -cc1 ---------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by clang -cc1.
//
//===----------------------------------------------------------------------===//

// Include the common option parsing interfaces.
include "OptParser.td"

// Target Options

def target_abi : Separate<"-target-abi">,
  HelpText<"Target a particular ABI type">;
def mcpu : Separate<"-mcpu">,
  HelpText<"Target a specific cpu type (-mcpu=help for details)">;
def target_feature : Separate<"-target-feature">,
  HelpText<"Target specific attributes">;
def triple : Separate<"-triple">,
  HelpText<"Specify target triple (e.g. i686-apple-darwin9)">;

// CodeGen Options

def disable_llvm_optzns : Flag<"-disable-llvm-optzns">,
  HelpText<"Don't run LLVM optimization passes">;
def disable_red_zone : Flag<"-disable-red-zone">,
  HelpText<"Do not emit code that uses the red zone.">;
def g : Flag<"-g">, HelpText<"Generate source level debug information">;
def fno_common : Flag<"-fno-common">,
  HelpText<"Compile common globals like normal definitions">;
def no_implicit_float : Flag<"-no-implicit-float">,
  HelpText<"Don't generate implicit floating point instructions (x86-only)">;
def fno_merge_all_constants : Flag<"-fno-merge-all-constants">,
  HelpText<"Disallow merging of constants.">;
def O : Joined<"-O">, HelpText<"Optimization level">;
def Os : Flag<"-Os">, HelpText<"Optimize for size">;
