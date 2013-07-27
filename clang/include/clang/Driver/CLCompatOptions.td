//===--- CLCompatOptions.td - Options for clang-cl ------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the options accepted by clang-cl.
//
//===----------------------------------------------------------------------===//

def cl_Group : OptionGroup<"<clang-cl options>">,
    HelpText<"CL.EXE COMPATIBILITY OPTIONS">;

class CLFlag<string name> : Option<["/", "-"], name, KIND_FLAG>,
    Group<cl_Group>, Flags<[CLOption]>;

def _QUESTION : CLFlag<"?">, Alias<help>, HelpText<"Display available options">;
def cl_help : CLFlag<"help">, Alias<help>, HelpText<"Display available options">;
