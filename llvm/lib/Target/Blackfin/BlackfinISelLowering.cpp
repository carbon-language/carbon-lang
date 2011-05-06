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
#include "llvm/Type.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/ADT/VectorExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
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

  // READCYCLECOUNTER needs special type legalization.
  setOperationAction(ISD::READCYCLECOUNTER, MVT::i64, Custom);

  setOperationAction(ISD::EH_LABEL, MVT::Other, Expand);

  // Use the default implementation.
  setOperationAction(ISD::VACOPY, MVT::Other, Expand);
  setOperationAction(ISD::VAEND, MVT::Other, Expand);
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);

  setMinFunctionAlignment(2);
}

const char *BlackfinTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case BFISD::CALL:     return "BFISD::CALL";
  case BFISD::RET_FLAG: return "BFISD::RET_FLAG";
  case BFISD::Wrapper:  return "BFISD::Wrapper";
  }
}

MVT::SimpleValueType BlackfinTargetLowering::getSetCCResultType(EVT VT) const {
  // SETCC always sets the CC register. Technically that is an i1 register, but
  // that type is not legal, so we treat it as an i32 register.
  return MVT::i32;
}

SDValue BlackfinTargetLowering::LowerGlobalAddress(SDValue Op,
                                                   SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();

  Op = DAG.getTargetGlobalAddress(GV, DL, MVT::i32);
  return DAG.getNode(BFISD::Wrapper, DL, MVT::i32, Op);
}

SDValue BlackfinTargetLowering::LowerJumpTable(SDValue Op,
                                               SelectionDAG &DAG) const {
  DebugLoc DL = Op.getDebugLoc();
  int JTI = cast<JumpTableSDNode>(Op)->getIndex();

  Op = DAG.getTargetJumpTable(JTI, MVT::i32);
  return DAG.getNode(BFISD::Wrapper, DL, MVT::i32, Op);
}

SDValue
BlackfinTargetLowering::LowerFormalArguments(SDValue Chain,
                                             CallingConv::ID CallConv, bool isVarArg,
                                            const SmallVectorImpl<ISD::InputArg>
                                               &Ins,
                                             DebugLoc dl, SelectionDAG &DAG,
                                             SmallVectorImpl<SDValue> &InVals)
                                               const {

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(),
                 ArgLocs, *DAG.getContext());
  CCInfo.AllocateStack(12, 4);  // ABI requires 12 bytes stack space
  CCInfo.AnalyzeFormalArguments(Ins, CC_Blackfin);

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    if (VA.isRegLoc()) {
      EVT RegVT = VA.getLocVT();
      TargetRegisterClass *RC = VA.getLocReg() == BF::P0 ?
        BF::PRegisterClass : BF::DRegisterClass;
      assert(RC->contains(VA.getLocReg()) && "Unexpected regclass in CCState");
      assert(RC->hasType(RegVT) && "Unexpected regclass in CCState");

      unsigned Reg = MF.getRegInfo().createVirtualRegister(RC);
      MF.getRegInfo().addLiveIn(VA.getLocReg(), Reg);
      SDValue ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);

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

      InVals.push_back(ArgValue);
    } else {
      assert(VA.isMemLoc() && "CCValAssign must be RegLoc or MemLoc");
      unsigned ObjSize = VA.getLocVT().getStoreSize();
      int FI = MFI->CreateFixedObject(ObjSize, VA.getLocMemOffset(), true);
      SDValue FIN = DAG.getFrameIndex(FI, MVT::i32);
      InVals.push_back(DAG.getLoad(VA.getValVT(), dl, Chain, FIN,
                                   MachinePointerInfo(),
                                   false, false, 0));
    }
  }

  return Chain;
}

SDValue
BlackfinTargetLowering::LowerReturn(SDValue Chain,
                                    CallingConv::ID CallConv, bool isVarArg,
                                    const SmallVectorImpl<ISD::OutputArg> &Outs,
                                    const SmallVectorImpl<SDValue> &OutVals,
                                    DebugLoc dl, SelectionDAG &DAG) const {

  // CCValAssign - represent the assignment of the return value to locations.
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, isVarArg, DAG.getTarget(),
                 RVLocs, *DAG.getContext());

  // Analize return values.
  CCInfo.AnalyzeReturn(Outs, RetCC_Blackfin);

  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    assert(VA.isRegLoc() && "Can only return in registers!");
    SDValue Opi = OutVals[i];

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

SDValue
BlackfinTargetLowering::LowerCall(SDValue Chain, SDValue Callee,
                                  CallingConv::ID CallConv, bool isVarArg,
                                  bool &isTailCall,
                                  const SmallVectorImpl<ISD::OutputArg> &Outs,
                                  const SmallVectorImpl<SDValue> &OutVals,
                                  const SmallVectorImpl<ISD::InputArg> &Ins,
                                  DebugLoc dl, SelectionDAG &DAG,
                                  SmallVectorImpl<SDValue> &InVals) const {
  // Blackfin target does not yet support tail call optimization.
  isTailCall = false;

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getTarget(), ArgLocs,
                 *DAG.getContext());
  CCInfo.AllocateStack(12, 4);  // ABI requires 12 bytes stack space
  CCInfo.AnalyzeCallOperands(Outs, CC_Blackfin);

  // Get the size of the outgoing arguments stack space requirement.
  unsigned ArgsSize = CCInfo.getNextStackOffset();

  Chain = DAG.getCALLSEQ_START(Chain, DAG.getIntPtrConstant(ArgsSize, true));
  SmallVector<std::pair<unsigned, SDValue>, 8> RegsToPass;
  SmallVector<SDValue, 8> MemOpChains;

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    SDValue Arg = OutVals[i];

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
      assert(VA.isMemLoc() && "CCValAssign must be RegLoc or MemLoc");
      int Offset = VA.getLocMemOffset();
      assert(Offset%4 == 0 && "Unaligned LocMemOffset");
      assert(VA.getLocVT()==MVT::i32 && "Illegal CCValAssign type");
      SDValue SPN = DAG.getCopyFromReg(Chain, dl, BF::SP, MVT::i32);
      SDValue OffsetN = DAG.getIntPtrConstant(Offset);
      OffsetN = DAG.getNode(ISD::ADD, dl, MVT::i32, SPN, OffsetN);
      MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, OffsetN,
                                         MachinePointerInfo(),false, false, 0));
    }
  }

  // Transform all store nodes into one single node because
  // all store nodes are independent of each other.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token
  // chain and flag operands which copy the outgoing args into registers.
  // The InFlag in necessary since all emitted instructions must be
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
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), dl, MVT::i32);
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), MVT::i32);

  std::vector<EVT> NodeTys;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Glue);    // Returns a flag for retval copy to use.
  SDValue Ops[] = { Chain, Callee, InFlag };
  Chain = DAG.getNode(BFISD::CALL, dl, NodeTys, Ops,
                      InFlag.getNode() ? 3 : 2);
  InFlag = Chain.getValue(1);

  Chain = DAG.getCALLSEQ_END(Chain, DAG.getIntPtrConstant(ArgsSize, true),
                             DAG.getIntPtrConstant(0, true), InFlag);
  InFlag = Chain.getValue(1);

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState RVInfo(CallConv, isVarArg, DAG.getTarget(), RVLocs,
                 *DAG.getContext());

  RVInfo.AnalyzeCallResult(Ins, RetCC_Blackfin);

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
    InVals.push_back(Val);
  }

  return Chain;
}

// Expansion of ADDE / SUBE. This is a bit involved since blackfin doesn't have
// add-with-carry instructions.
SDValue BlackfinTargetLowering::LowerADDE(SDValue Op, SelectionDAG &DAG) const {
  // Operands: lhs, rhs, carry-in (AC0 flag)
  // Results: sum, carry-out (AC0 flag)
  DebugLoc dl = Op.getDebugLoc();

  unsigned Opcode = Op.getOpcode()==ISD::ADDE ? BF::ADD : BF::SUB;

  // zext incoming carry flag in AC0 to 32 bits
  SDNode* CarryIn = DAG.getMachineNode(BF::MOVE_cc_ac0, dl, MVT::i32,
                                       /* flag= */ Op.getOperand(2));
  CarryIn = DAG.getMachineNode(BF::MOVECC_zext, dl, MVT::i32,
                               SDValue(CarryIn, 0));

  // Add operands, produce sum and carry flag
  SDNode *Sum = DAG.getMachineNode(Opcode, dl, MVT::i32, MVT::Glue,
                                   Op.getOperand(0), Op.getOperand(1));

  // Store intermediate carry from Sum
  SDNode* Carry1 = DAG.getMachineNode(BF::MOVE_cc_ac0, dl, MVT::i32,
                                      /* flag= */ SDValue(Sum, 1));

  // Add incoming carry, again producing an output flag
  Sum = DAG.getMachineNode(Opcode, dl, MVT::i32, MVT::Glue,
                           SDValue(Sum, 0), SDValue(CarryIn, 0));

  // Update AC0 with the intermediate carry, producing a flag.
  SDNode *CarryOut = DAG.getMachineNode(BF::OR_ac0_cc, dl, MVT::Glue,
                                        SDValue(Carry1, 0));

  // Compose (i32, flag) pair
  SDValue ops[2] = { SDValue(Sum, 0), SDValue(CarryOut, 0) };
  return DAG.getMergeValues(ops, 2, dl);
}

SDValue BlackfinTargetLowering::LowerOperation(SDValue Op,
                                               SelectionDAG &DAG) const {
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
  case ISD::ADDE:
  case ISD::SUBE:               return LowerADDE(Op, DAG);
  }
}

void
BlackfinTargetLowering::ReplaceNodeResults(SDNode *N,
                                           SmallVectorImpl<SDValue> &Results,
                                           SelectionDAG &DAG) const {
  DebugLoc dl = N->getDebugLoc();
  switch (N->getOpcode()) {
  default:
    llvm_unreachable("Do not know how to custom type legalize this operation!");
    return;
  case ISD::READCYCLECOUNTER: {
    // The low part of the cycle counter is in CYCLES, the high part in
    // CYCLES2. Reading CYCLES will latch the value of CYCLES2, so we must read
    // CYCLES2 last.
    SDValue TheChain = N->getOperand(0);
    SDValue lo = DAG.getCopyFromReg(TheChain, dl, BF::CYCLES, MVT::i32);
    SDValue hi = DAG.getCopyFromReg(lo.getValue(1), dl, BF::CYCLES2, MVT::i32);
    // Use a buildpair to merge the two 32-bit values into a 64-bit one.
    Results.push_back(DAG.getNode(ISD::BUILD_PAIR, dl, MVT::i64, lo, hi));
    // Outgoing chain. If we were to use the chain from lo instead, it would be
    // possible to entirely eliminate the CYCLES2 read in (i32 (trunc
    // readcyclecounter)). Unfortunately this could possibly delay the CYCLES2
    // read beyond the next CYCLES read, leading to invalid results.
    Results.push_back(hi.getValue(1));
    return;
  }
  }
}

//===----------------------------------------------------------------------===//
//                         Blackfin Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
BlackfinTargetLowering::ConstraintType
BlackfinTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() != 1)
    return TargetLowering::getConstraintType(Constraint);

  switch (Constraint[0]) {
    // Standard constraints
  case 'r':
    return C_RegisterClass;

    // Blackfin-specific constraints
  case 'a':
  case 'd':
  case 'z':
  case 'D':
  case 'W':
  case 'e':
  case 'b':
  case 'v':
  case 'f':
  case 'c':
  case 't':
  case 'u':
  case 'k':
  case 'x':
  case 'y':
  case 'w':
    return C_RegisterClass;
  case 'A':
  case 'B':
  case 'C':
  case 'Z':
  case 'Y':
    return C_Register;
  }

  // Not implemented: q0-q7, qA. Use {R2} etc instead

  return TargetLowering::getConstraintType(Constraint);
}

/// Examine constraint type and operand type and determine a weight value.
/// This object must already have been set up with the operand type
/// and the current alternative constraint selected.
TargetLowering::ConstraintWeight
BlackfinTargetLowering::getSingleConstraintMatchWeight(
    AsmOperandInfo &info, const char *constraint) const {
  ConstraintWeight weight = CW_Invalid;
  Value *CallOperandVal = info.CallOperandVal;
    // If we don't have a value, we can't do a match,
    // but allow it at the lowest weight.
  if (CallOperandVal == NULL)
    return CW_Default;
  // Look at the constraint type.
  switch (*constraint) {
  default:
    weight = TargetLowering::getSingleConstraintMatchWeight(info, constraint);
    break;

    // Blackfin-specific constraints
  case 'a':
  case 'd':
  case 'z':
  case 'D':
  case 'W':
  case 'e':
  case 'b':
  case 'v':
  case 'f':
  case 'c':
  case 't':
  case 'u':
  case 'k':
  case 'x':
  case 'y':
  case 'w':
    return CW_Register;
  case 'A':
  case 'B':
  case 'C':
  case 'Z':
  case 'Y':
    return CW_SpecificReg;
  }
  return weight;
}

/// getRegForInlineAsmConstraint - Return register no and class for a C_Register
/// constraint.
std::pair<unsigned, const TargetRegisterClass*> BlackfinTargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint, EVT VT) const {
  typedef std::pair<unsigned, const TargetRegisterClass*> Pair;
  using namespace BF;

  if (Constraint.size() != 1)
    return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);

  switch (Constraint[0]) {
    // Standard constraints
  case 'r':
    return Pair(0U, VT == MVT::i16 ? D16RegisterClass : DPRegisterClass);

    // Blackfin-specific constraints
  case 'a': return Pair(0U, PRegisterClass);
  case 'd': return Pair(0U, DRegisterClass);
  case 'e': return Pair(0U, AccuRegisterClass);
  case 'A': return Pair(A0, AccuRegisterClass);
  case 'B': return Pair(A1, AccuRegisterClass);
  case 'b': return Pair(0U, IRegisterClass);
  case 'v': return Pair(0U, BRegisterClass);
  case 'f': return Pair(0U, MRegisterClass);
  case 'C': return Pair(CC, JustCCRegisterClass);
  case 'x': return Pair(0U, GRRegisterClass);
  case 'w': return Pair(0U, ALLRegisterClass);
  case 'Z': return Pair(P3, PRegisterClass);
  case 'Y': return Pair(P1, PRegisterClass);
  }

  // Not implemented: q0-q7, qA. Use {R2} etc instead.
  // Constraints z, D, W, c, t, u, k, and y use non-existing classes, defer to
  // getRegClassForInlineAsmConstraint()

  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

std::vector<unsigned> BlackfinTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint, EVT VT) const {
  using namespace BF;

  if (Constraint.size() != 1)
    return std::vector<unsigned>();

  switch (Constraint[0]) {
  case 'z': return make_vector<unsigned>(P0, P1, P2, 0);
  case 'D': return make_vector<unsigned>(R0, R2, R4, R6, 0);
  case 'W': return make_vector<unsigned>(R1, R3, R5, R7, 0);
  case 'c': return make_vector<unsigned>(I0, I1, I2, I3,
                                         B0, B1, B2, B3,
                                         L0, L1, L2, L3, 0);
  case 't': return make_vector<unsigned>(LT0, LT1, 0);
  case 'u': return make_vector<unsigned>(LB0, LB1, 0);
  case 'k': return make_vector<unsigned>(LC0, LC1, 0);
  case 'y': return make_vector<unsigned>(RETS, RETN, RETI, RETX, RETE,
                                         ASTAT, SEQSTAT, USP, 0);
  }

  return std::vector<unsigned>();
}

bool BlackfinTargetLowering::
isOffsetFoldingLegal(const GlobalAddressSDNode *GA) const {
  // The Blackfin target isn't yet aware of offsets.
  return false;
}
