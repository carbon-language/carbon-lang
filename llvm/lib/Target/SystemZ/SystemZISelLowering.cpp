//===-- SystemZISelLowering.cpp - SystemZ DAG Lowering Implementation  -----==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemZTargetLowering class.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "systemz-lower"

#include "SystemZISelLowering.h"
#include "SystemZ.h"
#include "SystemZTargetMachine.h"
#include "SystemZSubtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/CallingConv.h"
#include "llvm/GlobalVariable.h"
#include "llvm/GlobalAlias.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/VectorExtras.h"
using namespace llvm;

SystemZTargetLowering::SystemZTargetLowering(SystemZTargetMachine &tm) :
  TargetLowering(tm), Subtarget(*tm.getSubtargetImpl()), TM(tm) {

  // Set up the register classes.
  addRegisterClass(MVT::i64, SystemZ::GR64RegisterClass);

  // Compute derived properties from the register classes
  computeRegisterProperties();

  // Provide all sorts of operation actions

  setStackPointerRegisterToSaveRestore(SystemZ::R15);
  setSchedulingPreference(SchedulingForLatency);
}

SDValue SystemZTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default:
    assert(0 && "unimplemented operand");
    return SDValue();
  }
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "SystemZGenCallingConv.inc"

const char *SystemZTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return NULL;
  }
}

