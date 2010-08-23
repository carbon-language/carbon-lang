//===- Simple.td - A simple LLVMC-based driver ----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A simple LLVMC-based gcc wrapper.
//
// To compile, use this command:
//
//      $ cd $LLVM_OBJ_DIR/tools/llvmc
//      $ make BUILD_EXAMPLES=1
//
// Run as:
//
//      $ $LLVM_OBJ_DIR/$(BuildMode)/bin/Simple
//
// For instructions on how to build your own LLVMC-based driver, see
// the 'examples/Skeleton' directory.
//===----------------------------------------------------------------------===//

include "llvm/CompilerDriver/Common.td"

def gcc : Tool<
[(in_language "c"),
 (out_language "executable"),
 (output_suffix "out"),
 (command "gcc"),
 (sink),

 // -o is what is used by default, out_file_option here is included for
 // instructive purposes.
 (out_file_option "-o")
]>;

def LanguageMap : LanguageMap<[(lang_to_suffixes "c", "c")]>;

def CompilationGraph : CompilationGraph<[(edge "root", "gcc")]>;
