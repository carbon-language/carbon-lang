//===- Example.td - LLVMCC toolchain descriptions  ---------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains toolchain descriptions used by llvmcc.
//
//===----------------------------------------------------------------------===//

include "Common.td"
include "Tools.td"

// Toolchains

def ToolChains : ToolChains<[
    ToolChain<[llvm_gcc_c, llc, llvm_gcc_assembler, llvm_gcc_linker]>,
    ToolChain<[llvm_gcc_cpp, llc, llvm_gcc_assembler, llvm_gcc_linker]>,
    ToolChain<[llvm_as, llc, llvm_gcc_assembler, llvm_gcc_linker]>,
    ToolChain<[llvm_gcc_assembler, llvm_gcc_linker]>
    ]>;
