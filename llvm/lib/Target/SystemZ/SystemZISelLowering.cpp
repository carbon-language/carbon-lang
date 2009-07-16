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

  RegInfo = TM.getRegisterInfo();

  // Set up the register classes.
  addRegisterClass(MVT::i32, SystemZ::GR32RegisterClass);
  addRegisterClass(MVT::i64, SystemZ::GR64RegisterClass);

  // Compute derived properties from the register classes
  computeRegisterProperties();

  // Set shifts properties
  setShiftAmountFlavor(Extend);
  setShiftAmountType(MVT::i32);

  // Provide all sorts of operation actions
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1, Promote);
  setLoadExtAction(ISD::EXTLOAD,  MVT::i1, Promote);

  setStackPointerRegisterToSaveRestore(SystemZ::R15D);
  setSchedulingPreference(SchedulingForLatency);

  setOperationAction(ISD::RET,              MVT::Other, Custom);

  setOperationAction(ISD::BRCOND,           MVT::Other, Expand);
  setOperationAction(ISD::BR_CC,            MVT::i32, Custom);
  setOperationAction(ISD::BR_CC,            MVT::i64, Custom);

  // FIXME: Can we lower these 2 efficiently?
  setOperationAction(ISD::SETCC,            MVT::i32, Expand);
  setOperationAction(ISD::SETCC,            MVT::i64, Expand);
  setOperationAction(ISD::SELECT,           MVT::i32, Expand);
  setOperationAction(ISD::SELECT,           MVT::i64, Expand);
  setOperationAction(ISD::SELECT_CC,        MVT::i32, Custom);
  setOperationAction(ISD::SELECT_CC,        MVT::i64, Custom);
}

SDValue SystemZTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  case ISD::FORMAL_ARGUMENTS: return LowerFORMAL_ARGUMENTS(Op, DAG);
  case ISD::RET:              return LowerRET(Op, DAG);
  case ISD::CALL:             return LowerCALL(Op, DAG);
  case ISD::BR_CC:            return LowerBR_CC(Op, DAG);
  case ISD::SELECT_CC:        return LowerSELECT_CC(Op, DAG);
  default:
    assert(0 && "unimplemented operand");
    return SDValue();
  }
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "SystemZGenCallingConv.inc"

SDValue SystemZTargetLowering::LowerFORMAL_ARGUMENTS(SDValue Op,
                                                     SelectionDAG &DAG) {
  unsigned CC = cast<ConstantSDNode>(Op.getOperand(1))->getZExtValue();
  switch (CC) {
  default:
    assert(0 && "Unsupported calling convention");
  case CallingConv::C:
  case CallingConv::Fast:
    return LowerCCCArguments(Op, DAG);
  }
}

SDValue SystemZTargetLowering::LowerCALL(SDValue Op, SelectionDAG &DAG) {
  CallSDNode *TheCall = cast<CallSDNode>(Op.getNode());
  unsigned CallingConv = TheCall->getCallingConv();
  switch (CallingConv) {
  default:
    assert(0 && "Unsupported calling convention");
  case CallingConv::Fast:
  case CallingConv::C:
    return LowerCCCCallTo(Op, DAG, CallingConv);
  }
}

/// LowerCCCArguments - transform physical registers into virtual registers and
/// generate load operations for arguments places on the stack.
// FIXME: struct return stuff
// FIXME: varargs
SDValue SystemZTargetLowering::LowerCCCArguments(SDValue Op,
                                                 SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  MachineRegisterInfo &RegInfo = MF.getRegInfo();
  SDValue Root = Op.getOperand(0);
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getZExtValue() != 0;
  unsigned CC = MF.getFunction()->getCallingConv();
  DebugLoc dl = Op.getDebugLoc();

  // Assign locations to all of the incoming arguments.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);
  CCInfo.AnalyzeFormalArguments(Op.getNode(), CC_SystemZ);

  assert(!isVarArg && "Varargs not supported yet");

  SmallVector<SDValue, 16> ArgValues;
  for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {
    CCValAssign &VA = ArgLocs[i];
    if (VA.isRegLoc()) {
      // Arguments passed in registers
      MVT RegVT = VA.getLocVT();
      switch (RegVT.getSimpleVT()) {
      default:
        cerr << "LowerFORMAL_ARGUMENTS Unhandled argument type: "
             << RegVT.getSimpleVT()
             << "\n";
        abort();
      case MVT::i64:
        unsigned VReg =
          RegInfo.createVirtualRegister(SystemZ::GR64RegisterClass);
        RegInfo.addLiveIn(VA.getLocReg(), VReg);
        SDValue ArgValue = DAG.getCopyFromReg(Root, dl, VReg, RegVT);

        // If this is an 8/16/32-bit value, it is really passed promoted to 64
        // bits. Insert an assert[sz]ext to capture this, then truncate to the
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
      }
    } else {
      // Sanity check
      assert(VA.isMemLoc());
      // Load the argument to a virtual register
      unsigned ObjSize = VA.getLocVT().getSizeInBits()/8;
      if (ObjSize > 8) {
        cerr << "LowerFORMAL_ARGUMENTS Unhandled argument type: "
             << VA.getLocVT().getSimpleVT()
             << "\n";
      }
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(ObjSize, VA.getLocMemOffset());

      // Create the SelectionDAG nodes corresponding to a load
      //from this parameter
      SDValue FIN = DAG.getFrameIndex(FI, MVT::i64);
      ArgValues.push_back(DAG.getLoad(VA.getLocVT(), dl, Root, FIN,
                                      PseudoSourceValue::getFixedStack(FI), 0));
    }
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  return DAG.getNode(ISD::MERGE_VALUES, dl, Op.getNode()->getVTList(),
                     &ArgValues[0], ArgValues.size()).getValue(Op.getResNo());
}

/// LowerCCCCallTo - functions arguments are copied from virtual regs to
/// (physical regs)/(stack frame), CALLSEQ_START and CALLSEQ_END are emitted.
/// TODO: sret.
SDValue SystemZTargetLowering::LowerCCCCallTo(SDValue Op, SelectionDAG &DAG,
                                              unsigned CC) {
  CallSDNode *TheCall = cast<CallSDNode>(Op.getNode());
  SDValue Chain  = TheCall->getChain();
  SDValue Callee = TheCall->getCallee();
  bool isVarArg  = TheCall->isVarArg();
  DebugLoc dl = Op.getDebugLoc();
  MachineFunction &MF = DAG.getMachineFunction();

  // Offset to first argument stack slot.
  const unsigned FirstArgOffset = 160;

  // Analyze operands of the call, assigning locations to each operand.
  SmallVector<CCValAssign, 16> ArgLocs;
  CCState CCInfo(CC, isVarArg, getTargetMachine(), ArgLocs);

  CCInfo.AnalyzeCallOperands(TheCall, CC_SystemZ);

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

    // Arguments start after the 5 first operands of ISD::CALL
    SDValue Arg = TheCall->getArg(i);

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
                             (RegInfo->hasFP(MF) ?
                              SystemZ::R11D : SystemZ::R15D),
                             getPointerTy());

      unsigned Offset = FirstArgOffset + VA.getLocMemOffset();
      SDValue PtrOff = DAG.getNode(ISD::ADD, dl, getPointerTy(),
                                   StackPtr,
                                   DAG.getIntPtrConstant(Offset));

      MemOpChains.push_back(DAG.getStore(Chain, dl, Arg, PtrOff,
                                         PseudoSourceValue::getStack(), Offset));
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
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), getPointerTy());
  else if (ExternalSymbolSDNode *E = dyn_cast<ExternalSymbolSDNode>(Callee))
    Callee = DAG.getTargetExternalSymbol(E->getSymbol(), getPointerTy());

  // Returns a chain & a flag for retval copy to use.
  SDVTList NodeTys = DAG.getVTList(MVT::Other, MVT::Flag);
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
  return SDValue(LowerCallResult(Chain, InFlag, TheCall, CC, DAG),
                 Op.getResNo());
}

/// LowerCallResult - Lower the result values of an ISD::CALL into the
/// appropriate copies out of appropriate physical registers.  This assumes that
/// Chain/InFlag are the input chain/flag to use, and that TheCall is the call
/// being lowered. Returns a SDNode with the same number of values as the
/// ISD::CALL.
SDNode*
SystemZTargetLowering::LowerCallResult(SDValue Chain, SDValue InFlag,
                                       CallSDNode *TheCall,
                                       unsigned CallingConv,
                                       SelectionDAG &DAG) {
  bool isVarArg = TheCall->isVarArg();
  DebugLoc dl = TheCall->getDebugLoc();

  // Assign locations to each value returned by this call.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallingConv, isVarArg, getTargetMachine(), RVLocs);

  CCInfo.AnalyzeCallResult(TheCall, RetCC_SystemZ);
  SmallVector<SDValue, 8> ResultVals;

  // Copy all of the result registers out of their specified physreg.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    Chain = DAG.getCopyFromReg(Chain, dl, RVLocs[i].getLocReg(),
                               RVLocs[i].getValVT(), InFlag).getValue(1);
    InFlag = Chain.getValue(2);
    ResultVals.push_back(Chain.getValue(0));
  }

  ResultVals.push_back(Chain);

  // Merge everything together with a MERGE_VALUES node.
  return DAG.getNode(ISD::MERGE_VALUES, dl, TheCall->getVTList(),
                     &ResultVals[0], ResultVals.size()).getNode();
}


SDValue SystemZTargetLowering::LowerRET(SDValue Op, SelectionDAG &DAG) {
  // CCValAssign - represent the assignment of the return value to a location
  SmallVector<CCValAssign, 16> RVLocs;
  unsigned CC   = DAG.getMachineFunction().getFunction()->getCallingConv();
  bool isVarArg = DAG.getMachineFunction().getFunction()->isVarArg();
  DebugLoc dl = Op.getDebugLoc();

  // CCState - Info about the registers and stack slot.
  CCState CCInfo(CC, isVarArg, getTargetMachine(), RVLocs);

  // Analize return values of ISD::RET
  CCInfo.AnalyzeReturn(Op.getNode(), RetCC_SystemZ);

  // If this is the first return lowered for this function, add the regs to the
  // liveout set for the function.
  if (DAG.getMachineFunction().getRegInfo().liveout_empty()) {
    for (unsigned i = 0; i != RVLocs.size(); ++i)
      if (RVLocs[i].isRegLoc())
        DAG.getMachineFunction().getRegInfo().addLiveOut(RVLocs[i].getLocReg());
  }

  // The chain is always operand #0
  SDValue Chain = Op.getOperand(0);
  SDValue Flag;

  // Copy the result values into the output registers.
  for (unsigned i = 0; i != RVLocs.size(); ++i) {
    CCValAssign &VA = RVLocs[i];
    SDValue ResValue = Op.getOperand(i*2+1);
    assert(VA.isRegLoc() && "Can only return in registers!");

    // If this is an 8/16/32-bit value, it is really should be passed promoted
    // to 64 bits.
    if (VA.getLocInfo() == CCValAssign::SExt)
      ResValue = DAG.getNode(ISD::SIGN_EXTEND, dl, VA.getLocVT(), ResValue);
    else if (VA.getLocInfo() == CCValAssign::ZExt)
      ResValue = DAG.getNode(ISD::ZERO_EXTEND, dl, VA.getLocVT(), ResValue);
    else if (VA.getLocInfo() == CCValAssign::AExt)
      ResValue = DAG.getNode(ISD::ANY_EXTEND, dl, VA.getLocVT(), ResValue);

    // ISD::RET => ret chain, (regnum1,val1), ...
    // So i*2+1 index only the regnums
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
                                       SelectionDAG &DAG) {
  assert(!LHS.getValueType().isFloatingPoint() && "We don't handle FP yet");

  // FIXME: Emit a test if RHS is zero

  bool isUnsigned = false;
  SystemZCC::CondCodes TCC;
  switch (CC) {
  default: assert(0 && "Invalid integer condition!");
  case ISD::SETEQ:
    TCC = SystemZCC::E;
    break;
  case ISD::SETNE:
    TCC = SystemZCC::NE;
    break;
  case ISD::SETULE:
    isUnsigned = true;   // FALLTHROUGH
  case ISD::SETLE:
    TCC = SystemZCC::LE;
    break;
  case ISD::SETUGE:
    isUnsigned = true;   // FALLTHROUGH
  case ISD::SETGE:
    TCC = SystemZCC::HE;
    break;
  case ISD::SETUGT:
    isUnsigned = true;
  case ISD::SETGT:
    TCC = SystemZCC::H; // FALLTHROUGH
    break;
  case ISD::SETULT:
    isUnsigned = true;
  case ISD::SETLT:      // FALLTHROUGH
    TCC = SystemZCC::L;
    break;
  }

  SystemZCC = DAG.getConstant(TCC, MVT::i32);

  DebugLoc dl = LHS.getDebugLoc();
  return DAG.getNode((isUnsigned ? SystemZISD::UCMP : SystemZISD::CMP),
                     dl, MVT::Flag, LHS, RHS);
}


SDValue SystemZTargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) {
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

SDValue SystemZTargetLowering::LowerSELECT_CC(SDValue Op, SelectionDAG &DAG) {
  SDValue LHS    = Op.getOperand(0);
  SDValue RHS    = Op.getOperand(1);
  SDValue TrueV  = Op.getOperand(2);
  SDValue FalseV = Op.getOperand(3);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(4))->get();
  DebugLoc dl   = Op.getDebugLoc();

  SDValue SystemZCC;
  SDValue Flag = EmitCmp(LHS, RHS, CC, SystemZCC, DAG);

  SDVTList VTs = DAG.getVTList(Op.getValueType(), MVT::Flag);
  SmallVector<SDValue, 4> Ops;
  Ops.push_back(TrueV);
  Ops.push_back(FalseV);
  Ops.push_back(SystemZCC);
  Ops.push_back(Flag);

  return DAG.getNode(SystemZISD::SELECT, dl, VTs, &Ops[0], Ops.size());
}


const char *SystemZTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  case SystemZISD::RET_FLAG:           return "SystemZISD::RET_FLAG";
  case SystemZISD::CALL:               return "SystemZISD::CALL";
  case SystemZISD::BRCOND:             return "SystemZISD::BRCOND";
  case SystemZISD::CMP:                return "SystemZISD::CMP";
  case SystemZISD::UCMP:               return "SystemZISD::UCMP";
  case SystemZISD::SELECT:             return "SystemZISD::SELECT";
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
  assert((MI->getOpcode() == SystemZ::Select32 ||
          MI->getOpcode() == SystemZ::Select64) &&
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
  BuildMI(BB, dl, TII.getBrCond(CC)).addMBB(copy1MBB);
  F->insert(I, copy0MBB);
  F->insert(I, copy1MBB);
  // Update machine-CFG edges by transferring all successors of the current
  // block to the new block which will contain the Phi node for the select.
  copy1MBB->transferSuccessors(BB);
  // Next, add the true and fallthrough blocks as its successors.
  BB->addSuccessor(copy0MBB);
  BB->addSuccessor(copy1MBB);

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
  BuildMI(BB, dl, TII.get(SystemZ::PHI),
          MI->getOperand(0).getReg())
    .addReg(MI->getOperand(2).getReg()).addMBB(copy0MBB)
    .addReg(MI->getOperand(1).getReg()).addMBB(thisMBB);

  F->DeleteMachineInstr(MI);   // The pseudo instruction is gone now.
  return BB;
}
