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
#include "llvm/CallingConv.h"
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
#include <queue>
#include <set>
using namespace llvm;

namespace {
  class ARMTargetLowering : public TargetLowering {
  public:
    ARMTargetLowering(TargetMachine &TM);
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    virtual const char *getTargetNodeName(unsigned Opcode) const;
  };

}

ARMTargetLowering::ARMTargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  setOperationAction(ISD::RET,           MVT::Other, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i32,   Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i32,   Custom);

  setSchedulingPreference(SchedulingForRegPressure);
}

namespace llvm {
  namespace ARMISD {
    enum NodeType {
      // Start the numbering where the builting ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+ARM::INSTRUCTION_LIST_END,
      /// CALL - A direct function call.
      CALL,

      /// Return with a flag operand.
      RET_FLAG
    };
  }
}

const char *ARMTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case ARMISD::CALL:          return "ARMISD::CALL";
  case ARMISD::RET_FLAG:      return "ARMISD::RET_FLAG";
  }
}

// This transforms a ISD::CALL node into a
// callseq_star <- ARMISD:CALL <- callseq_end
// chain
static SDOperand LowerCALL(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Chain    = Op.getOperand(0);
  unsigned CallConv  = cast<ConstantSDNode>(Op.getOperand(1))->getValue();
  assert(CallConv == CallingConv::C && "unknown calling convention");
  bool isVarArg      = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  bool isTailCall    = cast<ConstantSDNode>(Op.getOperand(3))->getValue() != 0;
  assert(isTailCall == false && "tail call not supported");
  SDOperand Callee   = Op.getOperand(4);
  unsigned NumOps    = (Op.getNumOperands() - 5) / 2;

  // Count how many bytes are to be pushed on the stack. Initially
  // only the link register.
  unsigned NumBytes = 4;

  // Add up all the space actually used.
  for (unsigned i = 4; i < NumOps; ++i)
    NumBytes += MVT::getSizeInBits(Op.getOperand(5+2*i).getValueType())/8;

  // Adjust the stack pointer for the new arguments...
  // These operations are automatically eliminated by the prolog/epilog pass
  Chain = DAG.getCALLSEQ_START(Chain,
                               DAG.getConstant(NumBytes, MVT::i32));

  SDOperand StackPtr = DAG.getRegister(ARM::R13, MVT::i32);

  static const unsigned int num_regs = 4;
  static const unsigned regs[num_regs] = {
    ARM::R0, ARM::R1, ARM::R2, ARM::R3
  };

  std::vector<std::pair<unsigned, SDOperand> > RegsToPass;
  std::vector<SDOperand> MemOpChains;

  for (unsigned i = 0; i != NumOps; ++i) {
    SDOperand Arg = Op.getOperand(5+2*i);
    assert(Arg.getValueType() == MVT::i32);
    if (i < num_regs)
      RegsToPass.push_back(std::make_pair(regs[i], Arg));
    else {
      unsigned ArgOffset = (i - num_regs) * 4;
      SDOperand PtrOff = DAG.getConstant(ArgOffset, StackPtr.getValueType());
      PtrOff = DAG.getNode(ISD::ADD, MVT::i32, StackPtr, PtrOff);
      MemOpChains.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                          Arg, PtrOff, DAG.getSrcValue(NULL)));
    }
  }
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, MemOpChains);

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing args into the appropriate regs.
  SDOperand InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, RegsToPass[i].first, RegsToPass[i].second,
                             InFlag);
    InFlag = Chain.getValue(1);
  }

  std::vector<MVT::ValueType> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.

  // If the callee is a GlobalAddress/ExternalSymbol node (quite common, every
  // direct call is) turn it into a TargetGlobalAddress/TargetExternalSymbol
  // node so that legalize doesn't hack it.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), Callee.getValueType());

  // If this is a direct call, pass the chain and the callee.
  assert (Callee.Val);
  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are known live
  // into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  unsigned CallOpc = ARMISD::CALL;
  if (InFlag.Val)
    Ops.push_back(InFlag);
  Chain = DAG.getNode(CallOpc, NodeTys, Ops);
  InFlag = Chain.getValue(1);

  std::vector<SDOperand> ResultVals;
  NodeTys.clear();

  // If the call has results, copy the values out of the ret val registers.
  switch (Op.Val->getValueType(0)) {
  default: assert(0 && "Unexpected ret value!");
  case MVT::Other:
    break;
  case MVT::i32:
    Chain = DAG.getCopyFromReg(Chain, ARM::R0, MVT::i32, InFlag).getValue(1);
    ResultVals.push_back(Chain.getValue(0));
    NodeTys.push_back(MVT::i32);
  }

  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, MVT::i32));
  NodeTys.push_back(MVT::Other);

  if (ResultVals.empty())
    return Chain;

  ResultVals.push_back(Chain);
  SDOperand Res = DAG.getNode(ISD::MERGE_VALUES, NodeTys, ResultVals);
  return Res.getValue(Op.ResNo);
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
    return DAG.getNode(ARMISD::RET_FLAG, MVT::Other, Chain);
  }
  case 3:
    Copy = DAG.getCopyToReg(Chain, ARM::R0, Op.getOperand(1), SDOperand());
    if (DAG.getMachineFunction().liveout_empty())
      DAG.getMachineFunction().addLiveOut(ARM::R0);
    break;
  }

  //We must use RET_FLAG instead of BRIND because BRIND doesn't have a flag
  return DAG.getNode(ARMISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
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
      unsigned ArgOffset = (ArgNo - num_regs) * 4;

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

static SDOperand LowerConstantPool(SDOperand Op, SelectionDAG &DAG) {
  MVT::ValueType PtrVT = Op.getValueType();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
  Constant *C = CP->get();
  SDOperand CPI = DAG.getTargetConstantPool(C, PtrVT, CP->getAlignment());

  return CPI;
}

static SDOperand LowerGlobalAddress(SDOperand Op,
				    SelectionDAG &DAG) {
  GlobalValue  *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  SDOperand CPAddr = DAG.getConstantPool(GV, MVT::i32, 2);
  return DAG.getLoad(MVT::i32, DAG.getEntryNode(), CPAddr,
		     DAG.getSrcValue(NULL));
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
  case ISD::ConstantPool:
    return LowerConstantPool(Op, DAG);
  case ISD::GlobalAddress:
    return LowerGlobalAddress(Op, DAG);
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

  SDNode *Select(SDOperand &Result, SDOperand Op);
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);
  bool SelectAddrRegImm(SDOperand N, SDOperand &Offset, SDOperand &Base);

  // Include the pieces autogenerated from the target description.
#include "ARMGenDAGISel.inc"
};

void ARMDAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());

  DAG.setRoot(SelectRoot(DAG.getRoot()));
  DAG.RemoveDeadNodes();

  ScheduleAndEmitDAG(DAG);
}

//register plus/minus 12 bit offset
bool ARMDAGToDAGISel::SelectAddrRegImm(SDOperand N, SDOperand &Offset,
				    SDOperand &Base) {
  Offset = CurDAG->getTargetConstant(0, MVT::i32);
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N)) {
    Base = CurDAG->getTargetFrameIndex(FI->getIndex(), N.getValueType());
  }
  else
    Base = N;
  return true;      //any address fits in a register
}

SDNode *ARMDAGToDAGISel::Select(SDOperand &Result, SDOperand Op) {
  SDNode *N = Op.Val;

  switch (N->getOpcode()) {
  default:
    return SelectCode(Result, Op);
    break;
  }
  return NULL;
}

}  // end anonymous namespace

/// createARMISelDag - This pass converts a legalized DAG into a
/// ARM-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createARMISelDag(TargetMachine &TM) {
  return new ARMDAGToDAGISel(TM);
}
