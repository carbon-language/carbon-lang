//===- llvm/InitializeAllTargets.h - Initialize All Targets -----*- C++ -*-===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This header initializes all configured LLVM targets, ensuring that they
// are registered.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_INITIALIZE_ALL_TARGETS_H
#define LLVM_INITIALIZE_ALL_TARGETS_H

namespace llvm {

  // Declare all of the target-initialization functions.
#define LLVM_TARGET(TargetName) void Initialize##TargetName##Target();
#include "llvm/Config/Targets.def"

  namespace {
    struct InitializeAllTargets {
      InitializeAllTargets() {
        // Call all of the target-initialization functions.
#define LLVM_TARGET(TargetName) llvm::Initialize##TargetName##Target();
#include "llvm/Config/Targets.def"
      }
    } DoInitializeAllTargets;
  }
} // end namespace llvm

#endif // LLVM_INITIALIZE_ALL_TARGETS_H
