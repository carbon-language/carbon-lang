//===-- WebAssemblyTargetInfo.cpp - WebAssembly Target Implementation -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file registers the WebAssembly target.
///
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-target-info"

Target llvm::TheWebAssemblyTarget;

extern "C" void LLVMInitializeWebAssemblyTargetInfo() {
  RegisterTarget<Triple::wasm32> X(TheWebAssemblyTarget, "wasm32",
                                   "WebAssembly 32-bit");
  RegisterTarget<Triple::wasm64> Y(TheWebAssemblyTarget, "wasm64",
                                   "WebAssembly 64-bit");
}
