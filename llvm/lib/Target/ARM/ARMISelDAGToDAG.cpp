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
#include "llvm/Constants.h"
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
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
  public:
    ARMTargetLowering(TargetMachine &TM);
    virtual SDOperand LowerOperation(SDOperand Op, SelectionDAG &DAG);
    virtual const char *getTargetNodeName(unsigned Opcode) const;
  };

}

ARMTargetLowering::ARMTargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  addRegisterClass(MVT::i32, ARM::IntRegsRegisterClass);
  addRegisterClass(MVT::f32, ARM::FPRegsRegisterClass);
  addRegisterClass(MVT::f64, ARM::DFPRegsRegisterClass);

  setOperationAction(ISD::SINT_TO_FP, MVT::i32, Custom);

  setOperationAction(ISD::RET,           MVT::Other, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i32,   Custom);
  setOperationAction(ISD::ConstantPool,  MVT::i32,   Custom);

  setOperationAction(ISD::SETCC, MVT::i32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i32, Custom);
  setOperationAction(ISD::BR_CC, MVT::i32, Custom);

  setOperationAction(ISD::VASTART,       MVT::Other, Custom);
  setOperationAction(ISD::VAEND,         MVT::Other, Expand);

  setSchedulingPreference(SchedulingForRegPressure);
  computeRegisterProperties();
}

namespace llvm {
  namespace ARMISD {
    enum NodeType {
      // Start the numbering where the builting ops and target ops leave off.
      FIRST_NUMBER = ISD::BUILTIN_OP_END+ARM::INSTRUCTION_LIST_END,
      /// CALL - A direct function call.
      CALL,

      /// Return with a flag operand.
      RET_FLAG,

      CMP,

      SELECT,

      BR,

      FSITOS,

      FSITOD,

      FMRRD
    };
  }
}

/// DAGCCToARMCC - Convert a DAG integer condition code to an ARM CC
static ARMCC::CondCodes DAGCCToARMCC(ISD::CondCode CC) {
  switch (CC) {
  default:
    std::cerr << "CC = " << CC << "\n";
    assert(0 && "Unknown condition code!");
  case ISD::SETUGT: return ARMCC::HI;
  case ISD::SETULE: return ARMCC::LS;
  case ISD::SETLE:  return ARMCC::LE;
  case ISD::SETLT:  return ARMCC::LT;
  case ISD::SETGT:  return ARMCC::GT;
  case ISD::SETNE:  return ARMCC::NE;
  case ISD::SETEQ:  return ARMCC::EQ;
  case ISD::SETGE:  return ARMCC::GE;
  case ISD::SETUGE: return ARMCC::CS;
  case ISD::SETULT: return ARMCC::CC;
  }
}

const char *ARMTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case ARMISD::CALL:          return "ARMISD::CALL";
  case ARMISD::RET_FLAG:      return "ARMISD::RET_FLAG";
  case ARMISD::SELECT:        return "ARMISD::SELECT";
  case ARMISD::CMP:           return "ARMISD::CMP";
  case ARMISD::BR:            return "ARMISD::BR";
  case ARMISD::FSITOS:        return "ARMISD::FSITOS";
  case ARMISD::FSITOD:        return "ARMISD::FSITOD";
  case ARMISD::FMRRD:         return "ARMISD::FMRRD";
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

  // Count how many bytes are to be pushed on the stack.
  unsigned NumBytes = 0;

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
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

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
  Chain = DAG.getNode(CallOpc, NodeTys, &Ops[0], Ops.size());
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
  SDOperand Res = DAG.getNode(ISD::MERGE_VALUES, NodeTys, &ResultVals[0],
                              ResultVals.size());
  return Res.getValue(Op.ResNo);
}

static SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG) {
  SDOperand Copy;
  SDOperand Chain = Op.getOperand(0);
  SDOperand    R0 = DAG.getRegister(ARM::R0, MVT::i32);
  SDOperand    R1 = DAG.getRegister(ARM::R1, MVT::i32);

  switch(Op.getNumOperands()) {
  default:
    assert(0 && "Do not know how to return this many arguments!");
    abort();
  case 1: {
    SDOperand LR = DAG.getRegister(ARM::R14, MVT::i32);
    return DAG.getNode(ARMISD::RET_FLAG, MVT::Other, Chain);
  }
  case 3: {
    SDOperand Val = Op.getOperand(1);
    assert(Val.getValueType() == MVT::i32 ||
	   Val.getValueType() == MVT::f32 ||
	   Val.getValueType() == MVT::f64);

    if (Val.getValueType() == MVT::f64) {
      SDVTList    VTs = DAG.getVTList(MVT::Other, MVT::Flag);
      SDOperand Ops[] = {Chain, R0, R1, Val};
      Copy  = DAG.getNode(ARMISD::FMRRD, VTs, Ops, 4);
    } else {
      if (Val.getValueType() == MVT::f32)
	Val = DAG.getNode(ISD::BIT_CONVERT, MVT::i32, Val);
      Copy = DAG.getCopyToReg(Chain, R0, Val, SDOperand());
    }

    if (DAG.getMachineFunction().liveout_empty()) {
      DAG.getMachineFunction().addLiveOut(ARM::R0);
      if (Val.getValueType() == MVT::f64)
        DAG.getMachineFunction().addLiveOut(ARM::R1);
    }
    break;
  }
  case 5:
    Copy = DAG.getCopyToReg(Chain, ARM::R1, Op.getOperand(3), SDOperand());
    Copy = DAG.getCopyToReg(Copy, ARM::R0, Op.getOperand(1), Copy.getValue(1));
    // If we haven't noted the R0+R1 are live out, do so now.
    if (DAG.getMachineFunction().liveout_empty()) {
      DAG.getMachineFunction().addLiveOut(ARM::R0);
      DAG.getMachineFunction().addLiveOut(ARM::R1);
    }
    break;
  }

  //We must use RET_FLAG instead of BRIND because BRIND doesn't have a flag
  return DAG.getNode(ARMISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
}

static SDOperand LowerFORMAL_ARGUMENT(SDOperand Op, SelectionDAG &DAG,
				      unsigned *vRegs,
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
    vRegs[ArgNo] = VReg;
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
  Constant *C = CP->getConstVal();
  SDOperand CPI = DAG.getTargetConstantPool(C, PtrVT, CP->getAlignment());

  return CPI;
}

static SDOperand LowerGlobalAddress(SDOperand Op,
				    SelectionDAG &DAG) {
  GlobalValue  *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  int alignment = 2;
  SDOperand CPAddr = DAG.getConstantPool(GV, MVT::i32, alignment);
  return DAG.getLoad(MVT::i32, DAG.getEntryNode(), CPAddr,
		     DAG.getSrcValue(NULL));
}

static SDOperand LowerVASTART(SDOperand Op, SelectionDAG &DAG,
                              unsigned VarArgsFrameIndex) {
  // vastart just stores the address of the VarArgsFrameIndex slot into the
  // memory location argument.
  MVT::ValueType PtrVT = DAG.getTargetLoweringInfo().getPointerTy();
  SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, PtrVT);
  return DAG.getNode(ISD::STORE, MVT::Other, Op.getOperand(0), FR, 
                     Op.getOperand(1), Op.getOperand(2));
}

static SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG,
				       int &VarArgsFrameIndex) {
  std::vector<SDOperand> ArgValues;
  SDOperand Root = Op.getOperand(0);
  unsigned VRegs[4];

  unsigned NumArgs = Op.Val->getNumValues()-1;
  for (unsigned ArgNo = 0; ArgNo < NumArgs; ++ArgNo) {
    SDOperand ArgVal = LowerFORMAL_ARGUMENT(Op, DAG, VRegs, ArgNo);

    ArgValues.push_back(ArgVal);
  }

  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  if (isVarArg) {
    MachineFunction &MF = DAG.getMachineFunction();
    SSARegMap *RegMap = MF.getSSARegMap();
    MachineFrameInfo *MFI = MF.getFrameInfo();
    VarArgsFrameIndex = MFI->CreateFixedObject(MVT::getSizeInBits(MVT::i32)/8,
                                               -16 + NumArgs * 4);


    static const unsigned REGS[] = {
      ARM::R0, ARM::R1, ARM::R2, ARM::R3
    };
    // If this function is vararg, store r0-r3 to their spots on the stack
    // so that they may be loaded by deferencing the result of va_next.
    SmallVector<SDOperand, 4> MemOps;
    for (unsigned ArgNo = 0; ArgNo < 4; ++ArgNo) {
      int ArgOffset = - (4 - ArgNo) * 4;
      int FI = MFI->CreateFixedObject(MVT::getSizeInBits(MVT::i32)/8,
				      ArgOffset);
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i32);

      unsigned VReg;
      if (ArgNo < NumArgs)
	VReg = VRegs[ArgNo];
      else
	VReg = RegMap->createVirtualRegister(&ARM::IntRegsRegClass);
      if (ArgNo >= NumArgs)
	MF.addLiveIn(REGS[ArgNo], VReg);

      SDOperand Val = DAG.getCopyFromReg(Root, VReg, MVT::i32);
      SDOperand Store = DAG.getNode(ISD::STORE, MVT::Other, Val.getValue(1),
                                    Val, FIN, DAG.getSrcValue(NULL));
      MemOps.push_back(Store);
    }
    Root = DAG.getNode(ISD::TokenFactor, MVT::Other,&MemOps[0],MemOps.size());
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  std::vector<MVT::ValueType> RetVT(Op.Val->value_begin(),
                                    Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVT, &ArgValues[0], ArgValues.size());
}

static SDOperand LowerSELECT_CC(SDOperand Op, SelectionDAG &DAG) {
  SDOperand LHS = Op.getOperand(0);
  SDOperand RHS = Op.getOperand(1);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  SDOperand TrueVal = Op.getOperand(2);
  SDOperand FalseVal = Op.getOperand(3);
  SDOperand    ARMCC = DAG.getConstant(DAGCCToARMCC(CC), MVT::i32);

  SDOperand Cmp = DAG.getNode(ARMISD::CMP, MVT::Flag, LHS, RHS);
  return DAG.getNode(ARMISD::SELECT, MVT::i32, TrueVal, FalseVal, ARMCC, Cmp);
}

static SDOperand LowerBR_CC(SDOperand Op, SelectionDAG &DAG) {
  SDOperand  Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDOperand    LHS = Op.getOperand(2);
  SDOperand    RHS = Op.getOperand(3);
  SDOperand   Dest = Op.getOperand(4);
  SDOperand  ARMCC = DAG.getConstant(DAGCCToARMCC(CC), MVT::i32);

  SDOperand Cmp = DAG.getNode(ARMISD::CMP, MVT::Flag, LHS, RHS);
  return DAG.getNode(ARMISD::BR, MVT::Other, Chain, Dest, ARMCC, Cmp);
}

static SDOperand LowerSINT_TO_FP(SDOperand Op, SelectionDAG &DAG) {
  SDOperand IntVal  = Op.getOperand(0);
  assert(IntVal.getValueType() == MVT::i32);
  MVT::ValueType vt = Op.getValueType();
  assert(vt == MVT::f32 ||
         vt == MVT::f64);

  SDOperand Tmp = DAG.getNode(ISD::BIT_CONVERT, MVT::f32, IntVal);
  ARMISD::NodeType op = vt == MVT::f32 ? ARMISD::FSITOS : ARMISD::FSITOD;
  return DAG.getNode(op, vt, Tmp);
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
  case ISD::SINT_TO_FP:
    return LowerSINT_TO_FP(Op, DAG);
  case ISD::FORMAL_ARGUMENTS:
    return LowerFORMAL_ARGUMENTS(Op, DAG, VarArgsFrameIndex);
  case ISD::CALL:
    return LowerCALL(Op, DAG);
  case ISD::RET:
    return LowerRET(Op, DAG);
  case ISD::SELECT_CC:
    return LowerSELECT_CC(Op, DAG);
  case ISD::BR_CC:
    return LowerBR_CC(Op, DAG);
  case ISD::VASTART:
    return LowerVASTART(Op, DAG, VarArgsFrameIndex);
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

  SDNode *Select(SDOperand Op);
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);
  bool SelectAddrRegImm(SDOperand N, SDOperand &Offset, SDOperand &Base);
  bool SelectAddrMode1(SDOperand N, SDOperand &Arg, SDOperand &Shift,
		       SDOperand &ShiftType);

  // Include the pieces autogenerated from the target description.
#include "ARMGenDAGISel.inc"
};

void ARMDAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());

  DAG.setRoot(SelectRoot(DAG.getRoot()));
  DAG.RemoveDeadNodes();

  ScheduleAndEmitDAG(DAG);
}

static bool isInt12Immediate(SDNode *N, short &Imm) {
  if (N->getOpcode() != ISD::Constant)
    return false;

  int32_t t = cast<ConstantSDNode>(N)->getValue();
  int max = 1<<12;
  int min = -max;
  if (t > min && t < max) {
    Imm = t;
    return true;
  }
  else
    return false;
}

static bool isInt12Immediate(SDOperand Op, short &Imm) {
  return isInt12Immediate(Op.Val, Imm);
}

static uint32_t rotateL(uint32_t x) {
  uint32_t bit31 = (x & (1 << 31)) >> 31;
  uint32_t     t = x << 1;
  return t | bit31;
}

static bool isUInt8Immediate(uint32_t x) {
  return x < (1 << 8);
}

static bool isRotInt8Immediate(uint32_t x) {
  int r;
  for (r = 0; r < 16; r++) {
    if (isUInt8Immediate(x))
      return true;
    x = rotateL(rotateL(x));
  }
  return false;
}

bool ARMDAGToDAGISel::SelectAddrMode1(SDOperand N,
				      SDOperand &Arg,
				      SDOperand &Shift,
				      SDOperand &ShiftType) {
  switch(N.getOpcode()) {
  case ISD::Constant: {
    uint32_t val = cast<ConstantSDNode>(N)->getValue();
    if(!isRotInt8Immediate(val)) {
      const Type  *t =  MVT::getTypeForValueType(MVT::i32);
      Constant    *C = ConstantUInt::get(t, val);
      int  alignment = 2;
      SDOperand Addr = CurDAG->getTargetConstantPool(C, MVT::i32, alignment);
      SDOperand    Z = CurDAG->getTargetConstant(0,     MVT::i32);
      SDNode      *n = CurDAG->getTargetNode(ARM::ldr,  MVT::i32, Z, Addr);
      Arg            = SDOperand(n, 0);
    } else
      Arg            = CurDAG->getTargetConstant(val,    MVT::i32);

    Shift     = CurDAG->getTargetConstant(0,             MVT::i32);
    ShiftType = CurDAG->getTargetConstant(ARMShift::LSL, MVT::i32);
    return true;
  }
  case ISD::SRA:
    Arg       = N.getOperand(0);
    Shift     = N.getOperand(1);
    ShiftType = CurDAG->getTargetConstant(ARMShift::ASR, MVT::i32);
    return true;
  case ISD::SRL:
    Arg       = N.getOperand(0);
    Shift     = N.getOperand(1);
    ShiftType = CurDAG->getTargetConstant(ARMShift::LSR, MVT::i32);
    return true;
  case ISD::SHL:
    Arg       = N.getOperand(0);
    Shift     = N.getOperand(1);
    ShiftType = CurDAG->getTargetConstant(ARMShift::LSL, MVT::i32);
    return true;
  }

  Arg       = N;
  Shift     = CurDAG->getTargetConstant(0, MVT::i32);
  ShiftType = CurDAG->getTargetConstant(ARMShift::LSL, MVT::i32);
  return true;
}

//register plus/minus 12 bit offset
bool ARMDAGToDAGISel::SelectAddrRegImm(SDOperand N, SDOperand &Offset,
				    SDOperand &Base) {
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(N)) {
    Base = CurDAG->getTargetFrameIndex(FIN->getIndex(), MVT::i32);
    Offset = CurDAG->getTargetConstant(0, MVT::i32);
    return true;
  }
  if (N.getOpcode() == ISD::ADD) {
    short imm = 0;
    if (isInt12Immediate(N.getOperand(1), imm)) {
      Offset = CurDAG->getTargetConstant(imm, MVT::i32);
      if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N.getOperand(0))) {
	Base = CurDAG->getTargetFrameIndex(FI->getIndex(), N.getValueType());
      } else {
	Base = N.getOperand(0);
      }
      return true; // [r+i]
    }
  }

  Offset = CurDAG->getTargetConstant(0, MVT::i32);
  if (FrameIndexSDNode *FI = dyn_cast<FrameIndexSDNode>(N)) {
    Base = CurDAG->getTargetFrameIndex(FI->getIndex(), N.getValueType());
  }
  else
    Base = N;
  return true;      //any address fits in a register
}

SDNode *ARMDAGToDAGISel::Select(SDOperand Op) {
  SDNode *N = Op.Val;

  switch (N->getOpcode()) {
  default:
    return SelectCode(Op);
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
