//===- BlackfinISelLowering.cpp - Blackfin DAG Lowering Implementation ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the interfaces that Blackfin uses to lower LLVM code
// into a selection DAG.
//
//===----------------------------------------------------------------------===//

#include "BlackfinISelLowering.h"
#include "BlackfinTargetMachine.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "BlackfinGenCallingConv.inc"

//===----------------------------------------------------------------------===//
// TargetLowering Implementation
//===----------------------------------------------------------------------===//

BlackfinTargetLowering::BlackfinTargetLowering(TargetMachine &TM)
  : TargetLowering(TM, new TargetLoweringObjectFileELF()) {
  setShiftAmountType(MVT::i16);
  setBooleanContents(ZeroOrOneBooleanContent);
  setStackPointerRegisterToSaveRestore(BF::SP);
  setIntDivIsCheap(false);

  // Set up the legal register classes.
  addRegisterClass(MVT::i32, BF::DRegisterClass);
  addRegisterClass(MVT::i16, BF::D16RegisterClass);

  computeRegisterProperties();

  // Blackfin doesn't have i1 loads or stores
  setLoadExtAction(ISD::EXTLOAD,  MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::JumpTable,     MVT::i32, Custom);

  setOperationAction(ISD::SELECT_CC, MVT::Other, Expand);
  setOperationAction(ISD::BR_JT,     MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,     MVT::Other, Expand);

  // i16 registers don't do much
  setOperationAction(ISD::AND,   MVT::i16, Promote);
  setOperationAction(ISD::OR,    MVT::i16, Promote);
  setOperationAction(ISD::XOR,   MVT::i16, Promote);
  setOperationAction(ISD::CTPOP, MVT::i16, Promote);
  // The expansion of CTLZ/CTTZ uses AND/OR, so we might as well promote
  // immediately.
  setOperationAction(ISD::CTLZ,  MVT::i16, Promote);
  setOperationAction(ISD::CTTZ,  MVT::i16, Promote);
  setOperationAction(ISD::SETCC, MVT::i16, Promote);

  // Blackfin has no division
  setOperationAction(ISD::SDIV,    MVT::i16, Expand);
  setOperationAction(ISD::SDIV,    MVT::i32, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i16, Expand);
  setOperationAction(ISD::SDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::SREM,    MVT::i16, Expand);
  setOperationAction(ISD::SREM,    MVT::i32, Expand);
  setOperationAction(ISD::UDIV,    MVT::i16, Expand);
  setOperationAction(ISD::UDIV,    MVT::i32, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i16, Expand);
  setOperationAction(ISD::UDIVREM, MVT::i32, Expand);
  setOperationAction(ISD::UREM,    MVT::i16, Expand);
  setOperationAction(ISD::UREM,    MVT::i32, Expand);

  setOperationAction(ISD::SMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::UMUL_LOHI, MVT::i32, Expand);
  setOperationAction(ISD::MULHU,     MVT::i32, Expand);
  setOperationAction(ISD::MULHS,     MVT::i32, Expand);

  // No carry-in operations.
  setOperationAction(ISD::ADDE, MVT::i32, Custom);
  setOperationAction(ISD::SUBE, MVT::i32, Custom);

  // Blackfin has no intrinsics for these particular operations.
  setOperationAction(ISD::MEMBARRIER, MVT::Other, Expand);
  setOperationAction(ISD::BSWAP, MVT::i32, Expand);

  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  // i32 has native CTPOP, but not CTLZ/CTTZ
  setOperationAction(ISD::CTLZ, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);

  // We don't have line number support yet.
  setOperationAction(ISD::DBG_STOPPOINT, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  setOperationAction(ISD::DBG_LABEL, MVT::Other, Expand);
  setOperationAction(ISD::EH_LABEL, MVT::Other, Expand);
  setOperationAction(ISD::DECLARE, MVT::Other, Expand);

  // Use the default implementation.
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);

  // RET must be custom lowered, to meet ABI requirements
  setOperationAction(ISD::RET, MVT::Other, Custom);
}

const char *BlackfinTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case BFISD::CALL:     return "BFISD::CALL";
  case BFISD::RET_FLAG: return "BFISD::RET_FLAG";
  case BFISD::Wrapper:  return "BFISD::Wrapper";
  }
}

MVT BlackfinTargetLowering::getSetCCResultType(MVT VT) const {
  // SETCC always sets the CC register. Technically that is an i1 register, but
  // that type is not legal, so we treat it as an i32 register.
  return MVT::i32;
}

SDValue BlackfinTargetLowering::LowerGlobalAddress(SDValue Op,
                                                   SelectionDAG &DAG) {
  DebugLoc DL = Op.getDebugLoc();
  GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();

  Op = DAG.getTargetGlobalAddress(GV, MVT::i32);
  return DAG.getNode(BFISD::Wrapper, DL, MVT::i32, Op);
}

SDValue BlackfinTargetLowering::LowerJumpTable(SDValue Op, SelectionDAG &DAG) {
  DebugLoc DL = Op.getDebugLoc();
  int JTI = cast<JumpTableSDNode>(Op)->getIndex();

  Op = DAG.getTargetJumpTable(JTI, MVT::i32);
  return DAG.getNode(BFISD::Wrapper, DL, MVT::i32, Op);
}

// FORMAL_ARGUMENTS(CHAIN, CC#, ISVARARG, FLAG0, ..., FLAGn) - This node
// represents the formal arguments for a function.  CC# is a Constant value
// indicating the calling convention of the function, and ISVARARG is a
// flag that indicates whether the function is varargs or not. This node
// has one result value for each incoming argument, plus one for the output
// chain.
SDValue BlackfinTargetLowering::LowerFORMAL_ARGUMENTS(SDValue Op,
                                                      SelectionDAG &DAG) {
  DebugLoc dl = Op.getDebugLoc();
  SDValue Root = Op.getOperand(0);
  unsigned CC = Op.getConstantOperandVal(1);
  bool isVarArg = Op.getConstantOperandVal(2);
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs, *DAG.getContext());
  CCInfo.AllocateStack(12, 4);	// ABI requires 12 bytes stack space
  CCInfo.AnalyzeFormalArguments(Op.getNode(), CC_Blackfin);

  SmallVector<SDValue, 8> ArgValues;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    if (VA.isRegLoc()) {
      MVT RegVT = VA.getLocVT();
      TargetRegisterClass *RC = VA.getLocReg() == BF::P0 ?
        BF::PRegisterClass : BF::DRegisterClass;
      assert(RC->contains(VA.getLocReg()));
      assert(RC->hasType(RegVT));

      unsigned Reg = MF.getRegInfo().createVirtualRegister(RC);
      MF.getRegInfo().addLiveIn(VA.getLocReg(), Reg);
      SDValue ArgValue = DAG.getCopyFromReg(Root, dl, Reg, RegVT);

      // If this is an 8 or 16-bit value, it is really passed promoted to 32
      // bits.  Insert an assert[sz]ext to capture this, then truncate to the
      // right size.
      if (VA.getLocInfo() == CCValAssign::SExt)
        ArgValue = DAG.getNode(ISD::AssertSext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));
      else if (VA.getLocInfo() == CCValAssign::ZExt)
        ArgValue = DAG.getNode(ISD::AssertZext, dl, RegVT, ArgValue,
                               DAG.getValueType(VA.getValVT()));

      if (VA.getLocInfo() != CCValAssign::Full)
        ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);

      ArgValues.push_back(ArgValue);
    } else {
      assert(VA.isMemLoc());
      unsigned ObjSize = VA.getLocVT().getStoreSizeInBits()/8;
      int FI = MFI->CreateFixedObject(ObjSize, VA.getLocMemOffset());
      SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
      ArgValues.push_back(DAG.getLoad(VA.getValVT(), dl, Root, FIN, NULL, 0));
    }
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  return DAG.getNode(ISD::MERGE_VALUES, dl, Op.getNode()->getVTList(),
                     &ArgValues[0], ArgValues.size()).getValue(Op.getResNo());
}

SDValue BlackfinTargetLowering::LowerRET(SDValue Op, SelectionDAG &DAG) {
  // CCValAssign - represent the assignment of the return value to locations.
  SmallVector<CCValAssign, 16> RVLocs;
  unsigned CC = DAG.getMachineFunction().getFunction()->getCallingConv();
  bool isVarArg = DAG.getMachineFunction().getFunction()->isVarArg();
  DebugLoc dl = Op.getDebugLoc();

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CC, isVarArg, DAG.getTarget(), RVLocs, *DAG.getContext());

  // Analize return values of ISD::RET
  CCInfo.AnalyzeReturn(Op.getNode(), RetCC_Blackfin);

  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  SDValue Chain = Op.getOperand(0);
  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    SDValue Opi = Op.getOperand(i*2+1);

    // Expand to i32 if necessary
    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::SExt:
      Opi = DAG.getNode(ISD::SIGN_EXTEND, dl, VA.getLocVT(), Opi);
      break;
    case CCValAssign::ZExt:
      Opi = DAG.getNode(ISD::ZERO_EXTEND, dl, VA.getLocVT(), Opi);
      break;
    case CCValAssign::AExt:
      Opi = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), Opi);
      break;
    }
    // ISD::RET => ret chain, (regnum1,val1), ...
    // So i*2+1 index only the regnums.
    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), Opi, SDValue());
    // Guarantee that all emitted copies are stuck together with flags.
    Flag = Chain.getValue(1);
  }

  if (Flag.getNode()) {
    return DAG.getNode(BFISD::RET_FLAG, dl, MVT::Other, Chain, Flag);
  } else {
    return DAG.getNode(BFISD::RET_FLAG, dl, MVT::Other, Chain);
  }
}

SDValue BlackfinTargetLowering::LowerCALL(SDValue Op, SelectionDAG &DAG) {
  CallSDNode *TheCall = cast<CallSDNode>(Op.getNode());
  unsigned CallingConv = TheCall->getCallingConv();
  SDValue Chain = TheCall->getChain();
  SDValue Callee = TheCall->getCallee();
  bool isVarArg = TheCall->isVarArg();
  DebugLoc dl = TheCall->getDebugLoc();

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallingConv, isVarArg, DAG.getTarget(), ArgLocs,
                 *DAG.getContext());
  CCInfo.AllocateStack(12, 4);	// ABI requires 12 bytes stack space
  CCInfo.AnalyzeCallOperands(TheCall, CC_Blackfin);

  // Get the size of the outgoing arguments stack space requirement.
  unsigned ArgsSize = CCInfo.getNextStackOffset();

  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(ArgsSize, true));
  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    // Arguments start after the 5 first operands of ISD::CALL
    SDValue Arg = TheCall->getArg(i);

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
    default: llvm_unreachable("Unknown loc info!");
    case CCValAssign::Full: break;
    case CCValAssign::SExt:
      Arg = DAG.getNode(ISD::SIGN_EXTEND, dl, VA.getLocVT(), Arg);
      break;
    case CCValAssign::ZExt:
      Arg = DAG.getNode(ISD::ZERO_EXTEND, dl, VA.getLocVT(), Arg);
      break;
    case CCValAssign::AExt:
      Arg = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), Arg);
      break;
    }

    // Arguments that can be passed on register must be kept at
    // RegsToPass vector
    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      assert(VA.isMemLoc());
      int Offset = VA.getLocMemOffset();
      assert(Offset%4 == 0);
      assert(VA.getLocVT()==MVT::i32);
      SDValue SPN = DAG.getCopyFromReg(Chain, dl, BF::SP, MVT::i32);
      SDValue OffsetN = DAG.getIntPtrConstant(Offset);
      OffsetN = DAG.getNode(ISD::ADD, dl, MVT::i32, SPN, OffsetN);
      MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, OffsetN,
                                         PseudoSourceValue::getStack(),
                                         Offset));
    }
  }

  // Transform all store nodes into one single node because
  // all store nodes are independent of each other.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token
  // chain and flag operands which copy the outgoing args into registers.
  // The InFlag in necessary since all emited instructions must be
  // stuck together.
  SDValue InFlag;
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain, dl, RegsToPass[i].first,
                             RegsToPass[i].second, InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
  // Likewise ExternalSymbol -> TargetExternalSymbol.
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee))
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i32);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), MVT::i32);

  std::vector<MVT> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  SDValue Ops[] = { Chain, Callee, InFlag };
  Chain = DAG.getNode(BFISD::CALL, dl, NodeTys, Ops,
                      InFlag.getNode() ? 3 : 2);
  InFlag = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(ArgsSize, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  InFlag = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState RVInfo(CallingConv, isVarArg, DAG.getTarget(), RVLocs,
                 *DAG.getContext());

  RVInfo.AnalyzeCallResult(TheCall, RetCC_Blackfin);
  SmallVector<SDValue, 8> ResultVals;

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &RV = RVLocs[i];
    unsigned Reg = RV.getLocReg();

    Chain = DAG.getCopyFromReg(Chain, dl, Reg,
                               RVLocs[i].getLocVT(), InFlag);
    SDValue Val = Chain.getValue(0);
    InFlag = Chain.getValue(2);
    Chain = Chain.getValue(1);

    // Callee is responsible for extending any i16 return values.
    switch (RV.getLocInfo()) {
    case CCValAssign::SExt:
      Val = DAG.getNode(ISD::AssertSext, dl, RV.getLocVT(), Val,
                        DAG.getValueType(RV.getValVT()));
      break;
    case CCValAssign::ZExt:
      Val = DAG.getNode(ISD::AssertZext, dl, RV.getLocVT(), Val,
                        DAG.getValueType(RV.getValVT()));
      break;
    default:
      break;
    }

    // Truncate to valtype
    if (RV.getLocInfo() != CCValAssign::Full)
      Val = DAG.getNode(ISD::TRUNCATE, dl, RV.getValVT(), Val);
    ResultVals.push_back(Val);
  }

  ResultVals.push_back(Chain);

  // Merge everything together with a MERGE_VALUES node.
  SDValue merge = DAG.getNode(ISD::MERGE_VALUES, dl,
                              TheCall->getVTList(), &ResultVals[0],
                              ResultVals.size());
  return merge;
}

// Expansion of ADDE / SUBE. This is a bit involved since blackfin doesn't have
// add-with-carry instructions.
SDValue BlackfinTargetLowering::LowerADDE(SDValue Op, SelectionDAG &DAG) {
  // Operands: lhs, rhs, carry-in (AC0 flag)
  // Results: sum, carry-out (AC0 flag)
  DebugLoc dl = Op.getDebugLoc();

  unsigned Opcode = Op.getOpcode()==ISD::ADDE ? BF::ADD : BF::SUB;

  // zext incoming carry flag in AC0 to 32 bits
  SDNode* CarryIn = DAG.getTargetNode(BF::MOVE_cc_ac0, dl, MVT::i32,
                                      /* flag= */ Op.getOperand(2));
  CarryIn = DAG.getTargetNode(BF::MOVECC_zext, dl, MVT::i32,
                              SDValue(CarryIn, 0));

  // Add operands, produce sum and carry flag
  SDNode *Sum = DAG.getTargetNode(Opcode, dl, MVT::i32, MVT::Flag,
                                  Op.getOperand(0), Op.getOperand(1));

  // Store intermediate carry from Sum
  SDNode* Carry1 = DAG.getTargetNode(BF::MOVE_cc_ac0, dl, MVT::i32,
                                     /* flag= */ SDValue(Sum, 1));

  // Add incoming carry, again producing an output flag
  Sum = DAG.getTargetNode(Opcode, dl, MVT::i32, MVT::Flag,
                          SDValue(Sum, 0), SDValue(CarryIn, 0));

  // Update AC0 with the intermediate carry, producing a flag.
  SDNode *CarryOut = DAG.getTargetNode(BF::OR_ac0_cc, dl, MVT::Flag,
                                       SDValue(Carry1, 0));

  // Compose (i32, flag) pair
  SDValue ops[2] = { SDValue(Sum, 0), SDValue(CarryOut, 0) };
  return DAG.getMergeValues(ops, 2, dl);
}

SDValue BlackfinTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default:
    Op.getNode()->dump();
    llvm_unreachable("Should not custom lower this!");
  case ISD::GlobalAddress:      return LowerGlobalAddress(Op, DAG);
  case ISD::GlobalTLSAddress:
    llvm_unreachable("TLS not implemented for Blackfin.");
  case ISD::JumpTable:          return LowerJumpTable(Op, DAG);
    // Frame & Return address.  Currently unimplemented
  case ISD::FRAMEADDR:          return SDValue();
  case ISD::RETURNADDR:         return SDValue();
  case ISD::FORMAL_ARGUMENTS:   return LowerFORMAL_ARGUMENTS(Op, DAG);
  case ISD::CALL:               return LowerCALL(Op, DAG);
  case ISD::RET:                return LowerRET(Op, DAG);
  case ISD::ADDE:
  case ISD::SUBE:               return LowerADDE(Op, DAG);
  }
}

/// getFunctionAlignment - Return the Log2 alignment of this function.
unsigned BlackfinTargetLowering::getFunctionAlignment(const Function *F) const {
  return 2;
}

//===----------------------------------------------------------------------===//
//                         Blackfin Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
BlackfinTargetLowering::ConstraintType
BlackfinTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default:  break;
    case 'r': return C_RegisterClass;
    }
  }

  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass*> BlackfinTargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint, MVT VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return std::make_pair(0U, BF::DRegisterClass);
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

std::vector<unsigned> BlackfinTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT VT) const {
  if (Constraint.size() != 1)
    return std::vector<unsigned>();

  return std::vector<unsigned>();
}

bool BlackfinTargetLowering::
isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // The Blackfin target isn't yet aware of offsets.
  return false;
}
