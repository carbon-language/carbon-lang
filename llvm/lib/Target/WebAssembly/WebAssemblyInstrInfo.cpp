//===-- WebAssemblyInstrInfo.cpp - WebAssembly Instruction Information ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the WebAssembly implementation of the
/// TargetInstrInfo class.
///
//===----------------------------------------------------------------------===//

#include "WebAssemblyInstrInfo.h"
#include "MCTargetDesc/WebAssemblyMCTargetDesc.h"
#include "WebAssemblySubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "wasm-instr-info"

#define GET_INSTRINFO_CTOR_DTOR
#include "WebAssemblyGenInstrInfo.inc"

WebAssemblyInstrInfo::WebAssemblyInstrInfo(const WebAssemblySubtarget &STI)
    : RI(STI.getTargetTriple()) {}
