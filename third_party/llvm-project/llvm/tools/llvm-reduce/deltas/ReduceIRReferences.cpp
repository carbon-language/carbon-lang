//===- ReduceIRReferences.cpp - Specialized Delta Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a function which calls the Generic Delta pass in order
// to remove backreferences to the IR from MIR. In particular, this will remove
// the Value references in MachineMemOperands.
//
//===----------------------------------------------------------------------===//

#include "ReduceIRReferences.h"
#include "Delta.h"
#include "llvm/CodeGen/MachineFrameInfo.h"

using namespace llvm;

static void dropIRReferencesFromInstructions(Oracle &O, MachineFunction &MF) {
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!O.shouldKeep()) {
        for (MachineMemOperand *MMO : MI.memoperands()) {
          // Leave behind pseudo source values.
          // TODO: Removing all MemOperand values is a further reduction step.
          if (MMO->getPointerInfo().V.is<const Value *>())
            MMO->setValue(static_cast<const Value *>(nullptr));
        }

        // TODO: Try to remove GlobalValue references and metadata
      }
    }
  }
}

static void stripIRFromInstructions(Oracle &O, ReducerWorkItem &WorkItem) {
  for (const Function &F : WorkItem.getModule()) {
    if (auto *MF = WorkItem.MMI->getMachineFunction(F))
      dropIRReferencesFromInstructions(O, *MF);
  }
}

static void stripIRFromBlocks(Oracle &O, ReducerWorkItem &WorkItem) {
  for (const Function &F : WorkItem.getModule()) {
    if (auto *MF = WorkItem.MMI->getMachineFunction(F)) {
      for (MachineBasicBlock &MBB : *MF) {
        if (!O.shouldKeep())
          MBB.clearBasicBlock();
      }
    }
  }
}

static void stripIRFromFunctions(Oracle &O, ReducerWorkItem &WorkItem) {
  for (const Function &F : WorkItem.getModule()) {
    if (!O.shouldKeep()) {
      if (auto *MF = WorkItem.MMI->getMachineFunction(F)) {
        MachineFrameInfo &MFI = MF->getFrameInfo();
        for (int I = MFI.getObjectIndexBegin(), E = MFI.getObjectIndexEnd();
             I != E; ++I)
          MFI.clearObjectAllocation(I);
      }
    }
  }
}

void llvm::reduceIRInstructionReferencesDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing IR references from instructions...\n";
  runDeltaPass(Test, stripIRFromInstructions);
}

void llvm::reduceIRBlockReferencesDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing IR references from blocks...\n";
  runDeltaPass(Test, stripIRFromBlocks);
}

void llvm::reduceIRFunctionReferencesDeltaPass(TestRunner &Test) {
  outs() << "*** Reducing IR references from functions...\n";
  runDeltaPass(Test, stripIRFromFunctions);
}
