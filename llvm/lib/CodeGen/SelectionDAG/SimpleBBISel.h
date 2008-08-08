//===-- SimpleBBISel.cpp - Definition of the SimpleBBISel class -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SimpleBBISel class which handles simple basic block
// instruction selection. If the given BasicBlock is considered "simple", i.e.
// all operations are supported by the target and their types are legal, it
// does instruction directly from LLVM BasicBlock to MachineInstr's.
//
//===----------------------------------------------------------------------===//

#ifndef SELECTIONDAG_SIMPLEBBISEL_H
#define SELECTIONDAG_SIMPLEBBISEL_H

#include "llvm/Support/Compiler.h"

namespace llvm {

class BasicBlock;
class MachineBasicBlock;
class MachineFunction;
class TargetLowering;

class VISIBILITY_HIDDEN SimpleBBISel {
  MachineFunction &MF;
  TargetLowering &TLI;
  
 public:
  explicit SimpleBBISel(MachineFunction &mf, TargetLowering &tli)
    : MF(mf), TLI(tli) {};

  /// SelectBasicBlock - Try to convert a LLVM basic block into a
  /// MachineBasicBlock using simple instruction selection. Returns false if it
  /// is not able to do so.
  bool SelectBasicBlock(BasicBlock *BB, MachineBasicBlock *MBB);
};
  
} // end namespace llvm.

#endif
