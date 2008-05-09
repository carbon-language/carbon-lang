//===- Example.td - LLVMCC toolchain descriptions  ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains compilation graph description used by llvmcc.
//
//===----------------------------------------------------------------------===//

include "Common.td"
include "Tools.td"

// Toolchains

def CompilationGraph : CompilationGraph<[
    Edge<root, llvm_gcc_c>,
    Edge<root, llvm_gcc_assembler>,
    Edge<root, llvm_gcc_cpp>,
    Edge<root, llvm_as>,

    Edge<llvm_gcc_c, llc>,
    Edge<llvm_gcc_cpp, llc>,
    Edge<llvm_as, llc>,

    OptionalEdge<llvm_gcc_c, opt, [(switch_on "opt")]>,
    OptionalEdge<llvm_gcc_cpp, opt, [(switch_on "opt")]>,
    OptionalEdge<llvm_as, opt, [(switch_on "opt")]>,
    Edge<opt, llc>,

    Edge<llc, llvm_gcc_assembler>,
    Edge<llvm_gcc_assembler, llvm_gcc_linker>,
    OptionalEdge<llvm_gcc_assembler, llvm_gcc_cpp_linker,
                 [(if_input_languages_contain "c++"),
                  (or (parameter_equals "linker", "g++"),
                      (parameter_equals "linker", "c++"))]>,


    Edge<root, llvm_gcc_linker>,
    OptionalEdge<root, llvm_gcc_cpp_linker,
                 [(if_input_languages_contain "c++"),
                  (or (parameter_equals "linker", "g++"),
                      (parameter_equals "linker", "c++"))]>
    ]>;
