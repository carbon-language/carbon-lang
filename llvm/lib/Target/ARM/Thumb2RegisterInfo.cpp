//===- Thumb2RegisterInfo.cpp - Thumb-2 Register Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Thumb-2 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMAddressingModes.h"
#include "ARMBaseInstrInfo.h"
#include "ARMMachineFunctionInfo.h"
#include "ARMSubtarget.h"
#include "Thumb2InstrInfo.h"
#include "Thumb2RegisterInfo.h"
#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/LLVMContext.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

Thumb2RegisterInfo::Thumb2RegisterInfo(const ARMBaseInstrInfo &tii,
                                       const ARMSubtarget &sti)
  : ARMBaseRegisterInfo(tii, sti) {
}

/// emitLoadConstPool - Emits a load from constpool to materialize the
/// specified immediate.
void Thumb2RegisterInfo::emitLoadConstPool(MachineBasicBlock &MBB,
                                           MachineBasicBlock::iterator &MBBI,
                                           DebugLoc dl,
                                           unsigned DestReg, unsigned SubIdx,
                                           int Val,
                                           ARMCC::CondCodes Pred,
                                           unsigned PredReg) const {
  MachineFunction &MF = *MBB.getParent();
  MachineConstantPool *ConstantPool = MF.getConstantPool();
  Constant *C = ConstantInt::get(
           Type::getInt32Ty(MBB.getParent()->getFunction()->getContext()), Val);
  unsigned Idx = ConstantPool->getConstantPoolIndex(C, 4);

  BuildMI(MBB, MBBI, dl, TII.get(ARM::t2LDRpci))
    .addReg(DestReg, getDefRegState(true), SubIdx)
    .addConstantPoolIndex(Idx).addImm((int64_t)ARMCC::AL).addReg(0);
}

bool Thumb2RegisterInfo::
requiresRegisterScavenging(const MachineFunction &MF) const {
  return true;
}
