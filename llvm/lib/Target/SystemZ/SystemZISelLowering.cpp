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
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/VectorExtras.h"
using namespace llvm;

SystemZTargetLowering::SystemZTargetLowering(SystemZTargetMachine &tm) :
  TargetLowering(tm, new TargetLoweringObjectFileELF()),
  Subtarget(*tm.getSubtargetImpl()), TM(tm) {

  RegInfo = TM.getRegisterInfo();

  // Set up the register classes.
  addRegisterClass(MVT::i32,  SystemZ::GR32RegisterClass);
  addRegisterClass(MVT::i64,  SystemZ::GR64RegisterClass);
  addRegisterClass(MVT::v2i32,SystemZ::GR64PRegisterClass);
  addRegisterClass(MVT::v2i64,SystemZ::GR128RegisterClass);

  if (!UseSoftFloat) {
    addRegisterClass(MVT::f32, SystemZ::FP32RegisterClass);
    addRegisterClass(MVT::f64, SystemZ::FP64RegisterClass);
  }

  // Compute derived properties from the register classes
  computeRegisterProperties();

  // Set shifts properties
  setShiftAmountType(MVT::i64);

  // Provide all sorts of operation actions
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::EXTLOAD,  MVT::i1, Promote);

  setLoadExtAction(ISD::SEXTLOAD, MVT::f32, Expand);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::f32, Expand);
  setLoadExtAction(ISD::EXTLOAD,  MVT::f32, Expand);

  setLoadExtAction(ISD::SEXTLOAD, MVT::f64, Expand);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::f64, Expand);
  setLoadExtAction(ISD::EXTLOAD,  MVT::f64, Expand);

  setStackPointerRegisterToSaveRestore(SystemZ::R15D);

  // TODO: It may be better to default to latency-oriented scheduling, however
  // LLVM's current latency-oriented scheduler can't handle physreg definitions
  // such as SystemZ has with PSW, so set this to the register-pressure
  // scheduler, because it can.
  setSchedulingPreference(Sched::RegPressure);

  setBooleanContents(ZeroOrOneBooleanContent);

  setOperationAction(ISD::BR_JT,            MVT::Other, Expand);
  setOperationAction(ISD::BRCOND,           MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,            MVT::i32, Custom);
  setOperationAction(ISD::BR_CC,            MVT::i64, Custom);
  setOperationAction(ISD::BR_CC,            MVT::f32, Custom);
  setOperationAction(ISD::BR_CC,            MVT::f64, Custom);
  setOperationAction(ISD::ConstantPool,     MVT::i32, Custom);
  setOperationAction(ISD::ConstantPool,     MVT::i64, Custom);
  setOperationAction(ISD::GlobalAddress,    MVT::i64, Custom);
  setOperationAction(ISD::JumpTable,        MVT::i64, Custom);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Expand);

  setOperationAction(ISD::SDIV,             MVT::i32, Expand);
  setOperationAction(ISD::UDIV,             MVT::i32, Expand);
  setOperationAction(ISD::SDIV,             MVT::i64, Expand);
  setOperationAction(ISD::UDIV,             MVT::i64, Expand);
  setOperationAction(ISD::SREM,             MVT::i32, Expand);
  setOperationAction(ISD::UREM,             MVT::i32, Expand);
  setOperationAction(ISD::SREM,             MVT::i64, Expand);
  setOperationAction(ISD::UREM,             MVT::i64, Expand);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  setOperationAction(ISD::CTPOP,            MVT::i32, Expand);
  setOperationAction(ISD::CTPOP,            MVT::i64, Expand);
  setOperationAction(ISD::CTTZ,             MVT::i32, Expand);
  setOperationAction(ISD::CTTZ,             MVT::i64, Expand);
  setOperationAction(ISD::CTLZ,             MVT::i32, Promote);
  setOperationAction(ISD::CTLZ,             MVT::i64, Legal);

  // FIXME: Can we lower these 2 efficiently?
  setOperationAction(ISD::SETCC,            MVT::i32, Expand);
  setOperationAction(ISD::SETCC,            MVT::i64, Expand);
  setOperationAction(ISD::SETCC,            MVT::f32, Expand);
  setOperationAction(ISD::SETCC,            MVT::f64, Expand);
  setOperationAction(ISD::SELECT,           MVT::i32, Expand);
  setOperationAction(ISD::SELECT,           MVT::i64, Expand);
  setOperationAction(ISD::SELECT,           MVT::f32, Expand);
  setOperationAction(ISD::SELECT,           MVT::f64, Expand);
  setOperationAction(ISD::SELECT_CC,        MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC,        MVT::i64, Custom);
  setOperationAction(ISD::SELECT_CC,        MVT::f32, Custom);
  setOperationAction(ISD::SELECT_CC,        MVT::f64, Custom);

  setOperationAction(ISD::MULHS,            MVT::i64, Expand);
  setOperationAction(ISD::SMUL_LOHI,        MVT::i64, Expand);

  // FIXME: Can we support these natively?
  setOperationAction(ISD::UMUL_LOHI,        MVT::i64, Expand);
  setOperationAction(ISD::SRL_PARTS,        MVT::i64, Expand);
  setOperationAction(ISD::SHL_PARTS,        MVT::i64, Expand);
  setOperationAction(ISD::SRA_PARTS,        MVT::i64, Expand);

  // Lower some FP stuff
  setOperationAction(ISD::FSIN,             MVT::f32, Expand);
  setOperationAction(ISD::FSIN,             MVT::f64, Expand);
  setOperationAction(ISD::FCOS,             MVT::f32, Expand);
  setOperationAction(ISD::FCOS,             MVT::f64, Expand);
  setOperationAction(ISD::FREM,             MVT::f32, Expand);
  setOperationAction(ISD::FREM,             MVT::f64, Expand);

  // We have only 64-bit bitconverts
  setOperationAction(ISD::BITCAST,          MVT::f32, Expand);
  setOperationAction(ISD::BITCAST,          MVT::i32, Expand);

  setOperationAction(ISD::UINT_TO_FP,       MVT::i32, Expand);
  setOperationAction(ISD::UINT_TO_FP,       MVT::i64, Expand);
  setOperationAction(ISD::FP_TO_UINT,       MVT::i32, Expand);
  setOperationAction(ISD::FP_TO_UINT,       MVT::i64, Expand);

  setTruncStoreAction(MVT::f64, MVT::f32, Expand);
}

SDValue SystemZTargetLowering::LowerOperation(SDValue Op,
                                              SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
  case ISD::BR_CC:            return LowerBR_CC(Op, DAG);
  case ISD::SELECT_CC:        return LowerSELECT_CC(Op, DAG);
  case ISD::GlobalAddress:    return LowerGlobalAddress(Op, DAG);
  case ISD::JumpTable:        return LowerJumpTable(Op, DAG);
  case ISD::ConstantPool:     return LowerConstantPool(Op, DAG);
  default:
    llvm_unreachable("Should not custom lower this!");
    return SDValue();
  }
}

bool SystemZTargetLowering::isFPImmLegal(const APFloat &Imm, EVT VT) const {
  if (UseSoftFloat || (VT != MVT::f32 && VT != MVT::f64))
    return false;

  // +0.0  lzer
  // +0.0f lzdr
  // -0.0  lzer + lner
  // -0.0f lzdr + lndr
  return Imm.isZero() || Imm.isNegZero();
}

//===----------------------------------------------------------------------===//
//                       SystemZ Inline Assembly Support
//===----------------------------------------------------------------------===//

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
TargetLowering::ConstraintType
SystemZTargetLowering::getConstraintType(const std::string &Constraint) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    case 'r':
      return C_RegisterClass;
    default:
      break;
    }
  }
  return TargetLowering::getConstraintType(Constraint);
}

std::pair<unsigned, const TargetRegisterClass*>
SystemZTargetLowering::
getRegForInlineAsmConstraint(const std::string &Constraint,
                             EVT VT) const {
  if (Constraint.size() == 1) {
    // GCC Constraint Letters
    switch (Constraint[0]) {
    default: break;
    case 'r':   // GENERAL_REGS
      if (VT == MVT::i32)
        return std::make_pair(0U, SystemZ::GR32RegisterClass);
      else if (VT == MVT::i128)
        return std::make_pair(0U, SystemZ::GR128RegisterClass);

      return std::make_pair(0U, SystemZ::GR64RegisterClass);
    }
  }

  return TargetLowering::getRegForInlineAsmConstraint(Constraint, VT);
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "SystemZGenCallingConv.inc"

SDValue
SystemZTargetLowering::LowerFormalArguments(SDValue Chain,
                                            CallingConv::ID CallConv,
                                            bool isVarArg,
                                            const SmallVectorImpl<ISD::InputArg>
                                              &Ins,
                                            DebugLoc dl,
                                            SelectionDAG &DAG,
                                            SmallVectorImpl<SDValue> &InVals)
                                              const {

  switch (CallConv) {
  default:
    llvm_unreachable("Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    return LowerCCCArguments(Chain, CallConv, isVarArg, Ins, dl, DAG, InVals);
  }
}

SDValue
SystemZTargetLowering::LowerCall(SDValue Chain, SDValue Callee,
                                 CallingConv::ID CallConv, bool isVarArg,
                                 bool &isTailCall,
                                 const SmallVectorImpl<ISD::OutputArg> &Outs,
                                 const SmallVectorImpl<SDValue> &OutVals,
                                 const SmallVectorImpl<ISD::InputArg> &Ins,
                                 DebugLoc dl, SelectionDAG &DAG,
                                 SmallVectorImpl<SDValue> &InVals) const {
  // SystemZ target does not yet support tail call optimization.
  isTailCall = false;

  switch (CallConv) {
  default:
    llvm_unreachable("Unsupported calling convention");
  case CallingConv::Fast:
  case CallingConv::C:
    return LowerCCCCallTo(Chain, Callee, CallConv, isVarArg, isTailCall,
                          Outs, OutVals, Ins, dl, DAG, InVals);
  }
}

/// LowerCCCArguments - transform physical registers into virtual registers and
/// generate load operations for arguments places on the stack.
// FIXME: struct return stuff
// FIXME: varargs
SDValue
SystemZTargetLowering::LowerCCCArguments(SDValue Chain,
                                         CallingConv::ID CallConv,
                                         bool isVarArg,
                                         const SmallVectorImpl<ISD::InputArg>
                                           &Ins,
                                         DebugLoc dl,
                                         SelectionDAG &DAG,
                                         SmallVectorImpl<SDValue> &InVals)
                                           const {

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(),
                 ArgLocs, *DAG.getContext());
  CCInfo.AnalyzeFormalArguments(Ins, CC_SystemZ);

  if (isVarArg)
    report_fatal_error("Varargs not supported yet");

  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    SDValue ArgValue;
    CCValAssign &VA = ArgLocs[i];
    EVT LocVT = VA.getLocVT();
    if (VA.isRegLoc()) {
      // Arguments passed in registers
      TargetRegisterClass *RC;
      switch (LocVT.getSimpleVT().SimpleTy) {
      default:
#ifndef NDEBUG
        errs() << "LowerFormalArguments Unhandled argument type: "
             << LocVT.getSimpleVT().SimpleTy
             << "\n";
#endif
        llvm_unreachable(0);
      case MVT::i64:
        RC = SystemZ::GR64RegisterClass;
        break;
      case MVT::f32:
        RC = SystemZ::FP32RegisterClass;
        break;
      case MVT::f64:
        RC = SystemZ::FP64RegisterClass;
        break;
      }

      unsigned VReg = RegInfo.createVirtualRegister(RC);
      RegInfo.addLiveIn(VA.getLocReg(), VReg);
      ArgValue = DAG.getCopyFromReg(Chain, dl, VReg, LocVT);
    } else {
      // Sanity check
      assert(VA.isMemLoc());

      // Create the nodes corresponding to a load from this parameter slot.
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(LocVT.getSizeInBits()/8,
                                      VA.getLocMemOffset(), true);

      // Create the SelectionDAG nodes corresponding to a load
      // from this parameter
      SDValue FIN = DAG.getFrameIndex(FI, getPointerTy());
      ArgValue = DAG.getLoad(LocVT, dl, Chain, FIN,
                             MachinePointerInfo::getFixedStack(FI),
                             false, false, 0);
    }

    // If this is an 8/16/32-bit value, it is really passed promoted to 64
    // bits. Insert an assert[sz]ext to capture this, then truncate to the
    // right size.
    if (VA.getLocInfo() == CCValAssign::SExt)
      ArgValue = DAG.getNode(ISD::AssertSext, dl, LocVT, ArgValue,
                             DAG.getValueType(VA.getValVT()));
    else if (VA.getLocInfo() == CCValAssign::ZExt)
      ArgValue = DAG.getNode(ISD::AssertZext, dl, LocVT, ArgValue,
                             DAG.getValueType(VA.getValVT()));

    if (VA.getLocInfo() != CCValAssign::Full)
      ArgValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), ArgValue);

    InVals.push_back(ArgValue);
  }

  return Chain;
}

/// LowerCCCCallTo - functions arguments are copied from virtual regs to
/// (physical regs)/(stack frame), CALLSEQ_START and CALLSEQ_END are emitted.
/// TODO: sret.
SDValue
SystemZTargetLowering::LowerCCCCallTo(SDValue Chain, SDValue Callee,
                                      CallingConv::ID CallConv, bool isVarArg,
                                      bool isTailCall,
                                      const SmallVectorImpl<ISD::OutputArg>
                                        &Outs,
                                      const SmallVectorImpl<SDValue> &OutVals,
                                      const SmallVectorImpl<ISD::InputArg> &Ins,
                                      DebugLoc dl, SelectionDAG &DAG,
                                      SmallVectorImpl<SDValue> &InVals) const {
  MachineFunction &MF = DAG.getMachineFunction();
  const TargetFrameLowering *TFI = TM.getFrameLowering();

  // Offset to first argument stack slot.
  const unsigned FirstArgOffset = 160;

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(),
                 ArgLocs, *DAG.getContext());

  CCInfo.AnalyzeCallOperands(Outs, CC_SystemZ);

  // Get a count of how many bytes are to be pushed on the stack.
  unsigned NumBytes = CCInfo.getNextStackOffset();

  Chain = DAG.getCALLSEQ_START(Chain ,DAG.getConstant(NumBytes,
                                                      getPointerTy(), true));

  SmallVector<std::pair<unsigned, SDValue>, 4> RegsToPass;
  SmallVector<SDValue, 12> MemOpChains;
  SDValue StackPtr;

  // Walk the register/memloc assignments, inserting copies/loads.
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];

    SDValue Arg = OutVals[i];

    // Promote the value if needed.
    switch (VA.getLocInfo()) {
      default: assert(0 && "Unknown loc info!");
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

    // Arguments that can be passed on register must be kept at RegsToPass
    // vector
    if (VA.isRegLoc()) {
      RegsToPass.push_back(std::make_pair(VA.getLocReg(), Arg));
    } else {
      assert(VA.isMemLoc());

      if (StackPtr.getNode() == 0)
        StackPtr =
          DAG.getCopyFromReg(Chain, dl,
                             (TFI->hasFP(MF) ?
                              SystemZ::R11D : SystemZ::R15D),
                             getPointerTy());

      unsigned Offset = FirstArgOffset + VA.getLocMemOffset();
      SDValue PtrOff = DAG.getNode(ISD::ADD, dl, getPointerTy(),
                                   StackPtr,
                                   DAG.getIntPtrConstant(Offset));

      MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, PtrOff,
                                         MachinePointerInfo(),
                                         false, false, 0));
    }
  }

  // Transform all store nodes into one single node because all store nodes are
  // independent of each other.
  if (!MemOpChains.empty())
    Chain = DAG.getNode(ISD::TokenFactor, dl, MVT::Other,
                        &MemOpChains[0], MemOpChains.size());

  // Build a sequence of copy-to-reg nodes chained together with token chain and
  // flag operands which copy the outgoing args into registers.  The InFlag in
  // necessary since all emited instructions must be stuck together.
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
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), dl, getPointerTy());
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), getPointerTy());

  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Glue);
  SmallVector<SDValue, 8> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);

  // Add argument registers to the end of the list so that they are
  // known live into the call.
  for (unsigned i = 0, e = RegsToPass.size(); i != e; ++i)
    Ops.push_back(DAG.getRegister(RegsToPass[i].first,
                                  RegsToPass[i].second.getValueType()));

  if (InFlag.getNode())
    Ops.push_back(InFlag);

  Chain = DAG.getNode(SystemZISD::CALL, dl, NodeTys, &Ops[0], Ops.size());
  InFlag = Chain.getValue(1);

  // Create the CALLSEQ_END node.
  Chain = DAG.getCALLSEQ_END(Chain,
                             DAG.getConstant(NumBytes, getPointerTy(), true),
                             DAG.getConstant(0, getPointerTy(), true),
                             InFlag);
  InFlag = Chain.getValue(1);

  // Handle result values, copying them out of physregs into vregs that we
  // return.
  return LowerCallResult(Chain, InFlag, CallConv, isVarArg, Ins, dl,
                         DAG, InVals);
}

/// LowerCallResult - Lower the result values of a call into the
/// appropriate copies out of appropriate physical registers.
///
SDValue
SystemZTargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                       CallingConv::ID CallConv, bool isVarArg,
                                       const SmallVectorImpl<ISD::InputArg>
                                         &Ins,
                                       DebugLoc dl, SelectionDAG &DAG,
                                       SmallVectorImpl<SDValue> &InVals) const {

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(), RVLocs,
                 *DAG.getContext());

  CCInfo.AnalyzeCallResult(Ins, RetCC_SystemZ);

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];

    Chain = DAG.getCopyFromReg(Chain, dl, VA.getLocReg(),
                               VA.getLocVT(), InFlag).getValue(1);
    SDValue RetValue = Chain.getValue(0);
    InFlag = Chain.getValue(2);

    // If this is an 8/16/32-bit value, it is really passed promoted to 64
    // bits. Insert an assert[sz]ext to capture this, then truncate to the
    // right size.
    if (VA.getLocInfo() == CCValAssign::SExt)
      RetValue = DAG.getNode(ISD::AssertSext, dl, VA.getLocVT(), RetValue,
                             DAG.getValueType(VA.getValVT()));
    else if (VA.getLocInfo() == CCValAssign::ZExt)
      RetValue = DAG.getNode(ISD::AssertZext, dl, VA.getLocVT(), RetValue,
                             DAG.getValueType(VA.getValVT()));

    if (VA.getLocInfo() != CCValAssign::Full)
      RetValue = DAG.getNode(ISD::TRUNCATE, dl, VA.getValVT(), RetValue);

    InVals.push_back(RetValue);
  }

  return Chain;
}


SDValue
SystemZTargetLowering::LowerReturn(SDValue Chain,
                                   CallingConv::ID CallConv, bool isVarArg,
                                   const SmallVectorImpl<ISD::OutputArg> &Outs,
                                   const SmallVectorImpl<SDValue> &OutVals,
                                   DebugLoc dl, SelectionDAG &DAG) const {

  // CCValAssign - represent the assignment of the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CallConv, isVarArg, getTargetMachine(),
                 RVLocs, *DAG.getContext());

  // Analize return values.
  CCInfo.AnalyzeReturn(Outs, RetCC_SystemZ);

  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    SDValue ResValue = OutVals[i];
    assert(VA.isRegLoc() && "Can only return in registers!");

    // If this is an 8/16/32-bit value, it is really should be passed promoted
    // to 64 bits.
    if (VA.getLocInfo() == CCValAssign::SExt)
      ResValue = DAG.getNode(ISD::SIGN_EXTEND, dl, VA.getLocVT(), ResValue);
    else if (VA.getLocInfo() == CCValAssign::ZExt)
      ResValue = DAG.getNode(ISD::ZERO_EXTEND, dl, VA.getLocVT(), ResValue);
    else if (VA.getLocInfo() == CCValAssign::AExt)
      ResValue = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), ResValue);

    Chain = DAG.getCopyToReg(Chain, dl, VA.getLocReg(), ResValue, Flag);

    // Guarantee that all emitted copies are stuck together,
    // avoiding something bad.
    Flag = Chain.getValue(1);
  }

  if (Flag.getNode())
    return DAG.getNode(SystemZISD::RET_FLAG, dl, MVT::Other, Chain, Flag);

  // Return Void
  return DAG.getNode(SystemZISD::RET_FLAG, dl, MVT::Other, Chain);
}

SDValue SystemZTargetLowering::EmitCmp(SDValue LHS, SDValue RHS,
                                       ISD::CondCode CC, SDValue &SystemZCC,
                                       SelectionDAG &DAG) const {
  // FIXME: Emit a test if RHS is zero

  bool isUnsigned = false;
  SystemZCC::CondCodes TCC;
  switch (CC) {
  default:
    llvm_unreachable("Invalid integer condition!");
  case ISD::SETEQ:
  case ISD::SETOEQ:
    TCC = SystemZCC::E;
    break;
  case ISD::SETUEQ:
    TCC = SystemZCC::NLH;
    break;
  case ISD::SETNE:
  case ISD::SETONE:
    TCC = SystemZCC::NE;
    break;
  case ISD::SETUNE:
    TCC = SystemZCC::LH;
    break;
  case ISD::SETO:
    TCC = SystemZCC::O;
    break;
  case ISD::SETUO:
    TCC = SystemZCC::NO;
    break;
  case ISD::SETULE:
    if (LHS.getValueType().isFloatingPoint()) {
      TCC = SystemZCC::NH;
      break;
    }
    isUnsigned = true;   // FALLTHROUGH
  case ISD::SETLE:
  case ISD::SETOLE:
    TCC = SystemZCC::LE;
    break;
  case ISD::SETUGE:
    if (LHS.getValueType().isFloatingPoint()) {
      TCC = SystemZCC::NL;
      break;
    }
    isUnsigned = true;   // FALLTHROUGH
  case ISD::SETGE:
  case ISD::SETOGE:
    TCC = SystemZCC::HE;
    break;
  case ISD::SETUGT:
    if (LHS.getValueType().isFloatingPoint()) {
      TCC = SystemZCC::NLE;
      break;
    }
    isUnsigned = true;  // FALLTHROUGH
  case ISD::SETGT:
  case ISD::SETOGT:
    TCC = SystemZCC::H;
    break;
  case ISD::SETULT:
    if (LHS.getValueType().isFloatingPoint()) {
      TCC = SystemZCC::NHE;
      break;
    }
    isUnsigned = true;  // FALLTHROUGH
  case ISD::SETLT:
  case ISD::SETOLT:
    TCC = SystemZCC::L;
    break;
  }

  SystemZCC = DAG.getConstant(TCC, MVT::i32);

  DebugLoc dl = LHS.getDebugLoc();
  return DAG.getNode((isUnsigned ? SystemZISD::UCMP : SystemZISD::CMP),
                     dl, MVT::i64, LHS, RHS);
}


SDValue SystemZTargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) const {
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS   = Op.getOperand(2);
  SDValue RHS   = Op.getOperand(3);
  SDValue Dest  = Op.getOperand(4);
  DebugLoc dl   = Op.getDebugLoc();

  SDValue SystemZCC;
  SDValue Flag = EmitCmp(LHS, RHS, CC, SystemZCC, DAG);
  return DAG.getNode(SystemZISD::BRCOND, dl, Op.getValueType(),
                     Chain, Dest, SystemZCC, Flag);
}

SDValue SystemZTargetLowering::LowerSELECT_CC(SDValue Op,
                                              SelectionDAG &DAG) const {
  SDValue LHS    = Op.getOperand(0);
  SDValue RHS    = Op.getOperand(1);
  SDValue TrueV  = Op.getOperand(2);
  SDValue FalseV = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  DebugLoc dl   = Op.getDebugLoc();

  SDValue SystemZCC;
  SDValue Flag = EmitCmp(LHS, RHS, CC, SystemZCC, DAG);

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Glue);
  SmallVector<SDValue, 4> Ops;
  Ops.push_back(TrueV);
  Ops.push_back(FalseV);
  Ops.push_back(SystemZCC);
  Ops.push_back(Flag);

  return DAG.getNode(SystemZISD::SELECT, dl, VTs, &Ops[0], Ops.size());
}

SDValue SystemZTargetLowering::LowerGlobalAddress(SDValue Op,
                                                  SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();
  int64_t Offset = cast<GlobalAddressSDNode>(Op)->getOffset();

  bool IsPic = getTargetMachine().getRelocationModel() == Reloc::PIC_;
  bool ExtraLoadRequired =
    Subtarget.GVRequiresExtraLoad(GV, getTargetMachine(), false);

  SDValue Result;
  if (!IsPic && !ExtraLoadRequired) {
    Result = DAG.getTargetGlobalAddress(GV, dl, getPointerTy(), Offset);
    Offset = 0;
  } else {
    unsigned char OpFlags = 0;
    if (ExtraLoadRequired)
      OpFlags = SystemZII::MO_GOTENT;

    Result = DAG.getTargetGlobalAddress(GV, dl, getPointerTy(), 0, OpFlags);
  }

  Result = DAG.getNode(SystemZISD::PCRelativeWrapper, dl,
                       getPointerTy(), Result);

  if (ExtraLoadRequired)
    Result = DAG.getLoad(getPointerTy(), dl, DAG.getEntryNode(), Result,
                         MachinePointerInfo::getGOT(), false, false, 0);

  // If there was a non-zero offset that we didn't fold, create an explicit
  // addition for it.
  if (Offset != 0)
    Result = DAG.getNode(ISD::ADD, dl, getPointerTy(), Result,
                         DAG.getConstant(Offset, getPointerTy()));

  return Result;
}

// FIXME: PIC here
SDValue SystemZTargetLowering::LowerJumpTable(SDValue Op,
                                              SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  JumpTableSDNode *JT = cast<JumpTableSDNode>(Op);
  SDValue Result = DAG.getTargetJumpTable(JT->getIndex(), getPointerTy());

  return DAG.getNode(SystemZISD::PCRelativeWrapper, dl, getPointerTy(), Result);
}


// FIXME: PIC here
// FIXME: This is just dirty hack. We need to lower cpool properly
SDValue SystemZTargetLowering::LowerConstantPool(SDValue Op,
                                                 SelectionDAG &DAG) const {
  DebugLoc dl = Op.getDebugLoc();
  ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);

  SDValue Result = DAG.getTargetConstantPool(CP->getConstVal(), getPointerTy(),
                                             CP->getAlignment(),
                                             CP->getOffset());

  return DAG.getNode(SystemZISD::PCRelativeWrapper, dl, getPointerTy(), Result);
}

const char *SystemZTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case SystemZISD::RET_FLAG:           return "SystemZISD::RET_FLAG";
  case SystemZISD::CALL:               return "SystemZISD::CALL";
  case SystemZISD::BRCOND:             return "SystemZISD::BRCOND";
  case SystemZISD::CMP:                return "SystemZISD::CMP";
  case SystemZISD::UCMP:               return "SystemZISD::UCMP";
  case SystemZISD::SELECT:             return "SystemZISD::SELECT";
  case SystemZISD::PCRelativeWrapper:  return "SystemZISD::PCRelativeWrapper";
  default: return NULL;
  }
}

//===----------------------------------------------------------------------===//
//  Other Lowering Code
//===----------------------------------------------------------------------===//

MachineBasicBlock*
SystemZTargetLowering::EmitInstrWithCustomInserter(MachineInstr *MI,
                                                   MachineBasicBlock *BB) const {
  const SystemZInstrInfo &TII = *TM.getInstrInfo();
  DebugLoc dl = MI->getDebugLoc();
  assert((MI->getOpcode() == SystemZ::Select32  ||
          MI->getOpcode() == SystemZ::SelectF32 ||
          MI->getOpcode() == SystemZ::Select64  ||
          MI->getOpcode() == SystemZ::SelectF64) &&
         "Unexpected instr type to insert");

  // To "insert" a SELECT instruction, we actually have to insert the diamond
  // control-flow pattern.  The incoming instruction knows the destination vreg
  // to set, the condition code register to branch on, the true/false values to
  // select between, and a branch opcode to use.
  const BasicBlock *LLVM_BB = BB->getBasicBlock();
  MachineFunction::iterator I = BB;
  ++I;

  //  thisMBB:
  //  ...
  //   TrueVal = ...
  //   cmpTY ccX, r1, r2
  //   jCC copy1MBB
  //   fallthrough --> copy0MBB
  MachineBasicBlock *thisMBB = BB;
  MachineFunction *F = BB->getParent();
  MachineBasicBlock *copy0MBB = F->CreateMachineBasicBlock(LLVM_BB);
  MachineBasicBlock *copy1MBB = F->CreateMachineBasicBlock(LLVM_BB);
  SystemZCC::CondCodes CC = (SystemZCC::CondCodes)MI->getOperand(3).getImm();
  F->insert(I, copy0MBB);
  F->insert(I, copy1MBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  copy1MBB->splice(copy1MBB->begin(), BB,
                   llvm::next(MachineBasicBlock::iterator(MI)),
                   BB->end());
  copy1MBB->transferSuccessorsAndUpdatePHIs(BB);
  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(copy1MBB);

  BuildMI(BB, dl, TII.getBrCond(CC)).addMBB(copy1MBB);

  //  copy0MBB:
  //   %FalseValue = ...
  //   # fallthrough to copy1MBB
  BB = copy0MBB;

  // Update machine-CFG edges
  BB->addSuccessor(copy1MBB);

  //  copy1MBB:
  //   %Result = phi [ %FalseValue, copy0MBB ], [ %TrueValue, thisMBB ]
  //  ...
  BB = copy1MBB;
  BuildMI(*BB, BB->begin(), dl, TII.get(SystemZ::PHI),
          MI->getOperand(0).getReg())
    .addReg(MI->getOperand(2).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(1).getReg()).addMBB(thisMBB);

  MI->eraseFromParent();   // The pseudo instruction is gone now.
  return BB;
}
