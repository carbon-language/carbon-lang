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
def target_cpu : Separate<"-mcpu">,
  HelpText<"Target a specific cpu type (-mcpu=help for details)">;
def target_features : Separate<"-target-feature">,
  HelpText<"Target specific attributes">;
def target_triple : Separate<"-triple">,
  HelpText<"Specify target triple (e.g. i686-apple-darwin9)">;
