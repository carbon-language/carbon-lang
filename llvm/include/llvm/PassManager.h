//===- llvm/PassManager.h - Container for Passes ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is a legacy redirect header for the old PassManager. It is intended to
// be used by clients that have not been converted to be aware of the new pass
// management infrastructure being built for LLVM, which is every client
// initially. Eventually this header (and the legacy management layer) will go
// away, but we want to minimize changes to out-of-tree users of LLVM in the
// interim.
//
// Note that this header *must not* be included into the same file as the new
// pass management infrastructure is included. Things will break spectacularly.
// If you are starting that conversion, you should switch to explicitly
// including LegacyPassManager.h and using the legacy namespace.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_PASSMANAGER_H
#define LLVM_PASSMANAGER_H

#include "llvm/IR/LegacyPassManager.h"

namespace llvm {

// Pull these into the llvm namespace so that existing code that expects it
// there can find it.
using legacy::PassManagerBase;
using legacy::PassManager;
using legacy::FunctionPassManager;

}

#endif
