//===-- SimpleBBISel.cpp - Implement the SimpleBBISel class ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements simple basic block instruction selection. If the given
// BasicBlock is considered "simple", i.e. all operations are supported by
// the target and their types are legal, it does instruction directly from
// LLVM BasicBlock to MachineInstr's.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simple-isel"
#include "SimpleBBISel.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
using namespace llvm;

/// SelectBasicBlock - Try to convert a LLVM basic block into a
/// MachineBasicBlock using simple instruction selection. Returns false if it
/// is not able to do so.
bool SimpleBBISel::SelectBasicBlock(BasicBlock *BB, MachineBasicBlock *MBB) {
  return false;
}
