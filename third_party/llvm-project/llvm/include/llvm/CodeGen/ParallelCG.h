//===-- llvm/CodeGen/ParallelCG.h - Parallel code generation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares functions that can be used for parallel code generation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_PARALLELCG_H
#define LLVM_CODEGEN_PARALLELCG_H

#include "llvm/Support/CodeGen.h"
#include <functional>
#include <memory>

namespace llvm {

template <typename T> class ArrayRef;
class Module;
class TargetMachine;
class raw_pwrite_stream;

/// Split M into OSs.size() partitions, and generate code for each. Takes a
/// factory function for the TargetMachine TMFactory. Writes OSs.size() output
/// files to the output streams in OSs. The resulting output files if linked
/// together are intended to be equivalent to the single output file that would
/// have been code generated from M.
///
/// Writes bitcode for individual partitions into output streams in BCOSs, if
/// BCOSs is not empty.
void splitCodeGen(
    Module &M, ArrayRef<raw_pwrite_stream *> OSs,
    ArrayRef<llvm::raw_pwrite_stream *> BCOSs,
    const std::function<std::unique_ptr<TargetMachine>()> &TMFactory,
    CodeGenFileType FileType = CGFT_ObjectFile, bool PreserveLocals = false);

} // namespace llvm

#endif
