//===-- ARMISelDAGToDAG.cpp - A dag to dag inst selector for ARM ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the ARM target.
//
//===----------------------------------------------------------------------===//

#include "ARM.h"
#include "ARMTargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/Debug.h"
#include <iostream>
#include <set>
using namespace llvm;

namespace {
  class ARMTargetLowering : public TargetLowering {
  public:
    ARMTargetLowering(TargetMachine &TM);
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
  };

}

ARMTargetLowering::ARMTargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  setOperationAction(ISD::RET, MVT::Other, Custom);
}

static SDOperand LowerCALL(SDOperand Op, SelectionDAG &DAG) {
  assert(0 && "Not implemented");
  abort();
}

static SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Copy;
  SDOperand Chain = Op.getOperand(0);
  switch(Op.getNumOperands()) {
  default:
    assert(0 && "Do not know how to return this many arguments!");
    abort();
  case 1: {
    SDOperand LR = DAG.getRegister(ARM::R14, MVT::i32);
    return DAG.getNode(ISD::BRIND, MVT::Other, Chain, LR);
  }
  case 3:
    Copy = DAG.getCopyToReg(Chain, ARM::R0, Op.getOperand(1), SDOperand());
    if (DAG.getMachineFunction().liveout_empty())
      DAG.getMachineFunction().addLiveOut(ARM::R0);
    break;
  }

  SDOperand LR = DAG.getRegister(ARM::R14, MVT::i32);

  //bug: the copy and branch should be linked with a flag so that the
  //scheduller can't move an instruction that destroys R0 in between them
  //return DAG.getNode(ISD::BRIND, MVT::Other, Copy, LR, Copy.getValue(1));

  return DAG.getNode(ISD::BRIND, MVT::Other, Copy, LR);
}

static SDOperand LowerFORMAL_ARGUMENT(SDOperand Op, SelectionDAG &DAG,
				      unsigned ArgNo) {
  MachineFunction &MF = DAG.getMachineFunction();
  MVT::ValueType ObjectVT = Op.getValue(ArgNo).getValueType();
  assert (ObjectVT == MVT::i32);
  SDOperand Root = Op.getOperand(0);
  SSARegMap *RegMap = MF.getSSARegMap();

  unsigned num_regs = 4;
  static const unsigned REGS[] = {
    ARM::R0, ARM::R1, ARM::R2, ARM::R3
  };

  if(ArgNo < num_regs) {
    unsigned VReg = RegMap->createVirtualRegister(&ARM::IntRegsRegClass);
    MF.addLiveIn(REGS[ArgNo], VReg);
    return DAG.getCopyFromReg(Root, VReg, MVT::i32);
  } else {
    // If the argument is actually used, emit a load from the right stack
      // slot.
    if (!Op.Val->hasNUsesOfValue(0, ArgNo)) {
      //hack
      unsigned ArgOffset = 0;

      MachineFrameInfo *MFI = MF.getFrameInfo();
      unsigned ObjSize = MVT::getSizeInBits(ObjectVT)/8;
      int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);
      return DAG.getLoad(ObjectVT, Root, FIN,
			 DAG.getSrcValue(NULL));
    } else {
      // Don't emit a dead load.
      return DAG.getNode(ISD::UNDEF, ObjectVT);
    }
  }
}

static SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG) {
  std::vector<SDOperand> ArgValues;
  SDOperand Root = Op.getOperand(0);

  for (unsigned ArgNo = 0, e = Op.Val->getNumValues()-1; ArgNo != e; ++ArgNo) {
    SDOperand ArgVal = LowerFORMAL_ARGUMENT(Op, DAG, ArgNo);

    ArgValues.push_back(ArgVal);
  }

  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  assert(!isVarArg);

  ArgValues.push_back(Root);

  // Return the new list of results.
  std::vector<MVT::ValueType> RetVT(Op.Val->value_begin(),
                                    Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVT, ArgValues);
}

SDOperand ARMTargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default:
    assert(0 && "Should not custom lower this!");
    abort();
  case ISD::FORMAL_ARGUMENTS:
    return LowerFORMAL_ARGUMENTS(Op, DAG);
  case ISD::CALL:
    return LowerCALL(Op, DAG);
  case ISD::RET:
    return LowerRET(Op, DAG);
  }
}

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
/// ARMDAGToDAGISel - ARM specific code to select ARM machine
/// instructions for SelectionDAG operations.
///
namespace {
class ARMDAGToDAGISel : public SelectionDAGISel {
  ARMTargetLowering Lowering;

public:
  ARMDAGToDAGISel(TargetMachine &TM)
    : SelectionDAGISel(Lowering), Lowering(TM) {
  }

  void Select(SDOperand &Result, SDOperand Op);
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);

  // Include the pieces autogenerated from the target description.
#include "ARMGenDAGISel.inc"
};

void ARMDAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());

  DAG.setRoot(SelectRoot(DAG.getRoot()));
  assert(InFlightSet.empty() && "ISel InFlightSet has not been emptied!");
  CodeGenMap.clear();
  HandleMap.clear();
  ReplaceMap.clear();
  DAG.RemoveDeadNodes();

  ScheduleAndEmitDAG(DAG);
}

static void SelectFrameIndex(SelectionDAG *CurDAG, SDOperand &Result, SDNode *N, SDOperand Op) {
  int FI = cast<FrameIndexSDNode>(N)->getIndex();

  SDOperand TFI = CurDAG->getTargetFrameIndex(FI, Op.getValueType());

  Result = CurDAG->SelectNodeTo(N, ARM::movri, Op.getValueType(), TFI);
}

void ARMDAGToDAGISel::Select(SDOperand &Result, SDOperand Op) {
  SDNode *N = Op.Val;

  switch (N->getOpcode()) {
  default:
    SelectCode(Result, Op);
    break;

  case ISD::FrameIndex:
    SelectFrameIndex(CurDAG, Result, N, Op);
    break;
  }
}

}  // end anonymous namespace

/// createARMISelDag - This pass converts a legalized DAG into a
/// ARM-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createARMISelDag(TargetMachine &TM) {
  return new ARMDAGToDAGISel(TM);
}
