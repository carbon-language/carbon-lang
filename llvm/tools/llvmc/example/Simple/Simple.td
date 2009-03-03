//===- Simple.td - A simple plugin for LLVMC ------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A simple LLVMC-based gcc wrapper that shows how to write LLVMC plugins.
//
// To compile, use this command:
//
//      $ cd $LLVMC_DIR/example/Simple
//      $ make
//
// Run as:
//
//      $ llvmc -load $LLVM_DIR/Release/lib/plugin_llvmc_Simple.so
//
// For instructions on how to build your own LLVMC-based driver, see
// the 'example/Skeleton' directory.
//===----------------------------------------------------------------------===//

include "llvm/CompilerDriver/Common.td"

def gcc : Tool<
[(in_language "c"),
 (out_language "executable"),
 (output_suffix "out"),
 (cmd_line "gcc $INFILE -o $OUTFILE"),
 (sink)
]>;

def LanguageMap : LanguageMap<[LangToSuffixes<"c", ["c"]>]>;

def CompilationGraph : CompilationGraph<[Edge<"root", "gcc">]>;
