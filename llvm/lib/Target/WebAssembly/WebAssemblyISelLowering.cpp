//=- WebAssemblyISelLowering.cpp - WebAssembly DAG Lowering Implementation -==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements the WebAssemblyTargetLowering class.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyISelLowering.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblyMachineFunctionInfo.h"
#include "WebAssemblySubtarget.h"
#include "WebAssemblyTargetMachine.h"
#include "WebAssemblyTargetObjectFile.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetOptions.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-lower"

WebAssemblyTargetLowering::WebAssemblyTargetLowering(
    const TargetMachine &TM, const WebAssemblySubtarget &STI)
    : TargetLowering(TM), Subtarget(&STI) {
  // WebAssembly does not produce floating-point exceptions on normal floating
  // point operations.
  setHasFloatingPointExceptions(false);
}

//===----------------------------------------------------------------------===//
// WebAssembly Lowering private implementation.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Lowering Code
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//  Other Lowering Code
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                          WebAssembly Optimization Hooks
//===----------------------------------------------------------------------===//

MCSection *WebAssemblyTargetObjectFile::SelectSectionForGlobal(
    const GlobalValue *GV, SectionKind Kind, Mangler &Mang,
    const TargetMachine &TM) const {
  return getDataSection();
}
