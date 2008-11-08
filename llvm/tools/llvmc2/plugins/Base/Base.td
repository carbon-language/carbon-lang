//===- Base.td - LLVMC2 toolchain descriptions -------------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains compilation graph description used by llvmc2.
//
//===----------------------------------------------------------------------===//

include "llvm/CompilerDriver/Common.td"
include "llvm/CompilerDriver/Tools.td"

// Toolchains

def CompilationGraph : CompilationGraph<[
    Edge<root, llvm_gcc_c>,
    Edge<root, llvm_gcc_assembler>,
    Edge<root, llvm_gcc_cpp>,
    Edge<root, llvm_gcc_m>,
    Edge<root, llvm_gcc_mxx>,
    Edge<root, llvm_as>,

    Edge<llvm_gcc_c, llc>,
    Edge<llvm_gcc_cpp, llc>,
    Edge<llvm_gcc_m, llc>,
    Edge<llvm_gcc_mxx, llc>,
    Edge<llvm_as, llc>,

    OptionalEdge<llvm_gcc_c, opt, (case (switch_on "opt"), (inc_weight))>,
    OptionalEdge<llvm_gcc_cpp, opt, (case (switch_on "opt"), (inc_weight))>,
    OptionalEdge<llvm_gcc_m, opt, (case (switch_on "opt"), (inc_weight))>,
    OptionalEdge<llvm_gcc_mxx, opt, (case (switch_on "opt"), (inc_weight))>,
    OptionalEdge<llvm_as, opt, (case (switch_on "opt"), (inc_weight))>,
    Edge<opt, llc>,

    Edge<llc, llvm_gcc_assembler>,
    Edge<llvm_gcc_assembler, llvm_gcc_linker>,
    OptionalEdge<llvm_gcc_assembler, llvm_gcc_cpp_linker,
                 (case
                     (or (input_languages_contain "c++"),
                         (input_languages_contain "objective-c++")), 
                     (inc_weight),
                     (or (parameter_equals "linker", "g++"),
                         (parameter_equals "linker", "c++")), (inc_weight))>,


    Edge<root, llvm_gcc_linker>,
    OptionalEdge<root, llvm_gcc_cpp_linker,
                 (case
                     (or (input_languages_contain "c++"),
                         (input_languages_contain "objective-c++")), 
                     (inc_weight),
                     (or (parameter_equals "linker", "g++"),
                         (parameter_equals "linker", "c++")), (inc_weight))>
    ]>;
