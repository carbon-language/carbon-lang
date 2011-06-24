//===-- PTXISelLowering.cpp - PTX DAG Lowering Implementation -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PTXTargetLowering class.
//
//===----------------------------------------------------------------------===//

#include "PTX.h"
#include "PTXISelLowering.h"
#include "PTXMachineFunctionInfo.h"
#include "PTXRegisterInfo.h"
#include "PTXSubtarget.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
// Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "PTXGenCallingConv.inc"

//===----------------------------------------------------------------------===//
// TargetLowering Implementation
//===----------------------------------------------------------------------===//

PTXTargetLowering::PTXTargetLowering(TargetMachine &TM)
  : TargetLowering(TM, new TargetLoweringObjectFileELF()) {
  // Set up the register classes.
  addRegisterClass(MVT::i1,  PTX::RegPredRegisterClass);
  addRegisterClass(MVT::i16, PTX::RegI16RegisterClass);
  addRegisterClass(MVT::i32, PTX::RegI32RegisterClass);
  addRegisterClass(MVT::i64, PTX::RegI64RegisterClass);
  addRegisterClass(MVT::f32, PTX::RegF32RegisterClass);
  addRegisterClass(MVT::f64, PTX::RegF64RegisterClass);

  setBooleanContents(ZeroOrOneBooleanContent);

  setOperationAction(ISD::EXCEPTIONADDR, MVT::i32, Expand);

  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f64, Legal);

  // Turn i16 (z)extload into load + (z)extend
  setLoadExtAction(ISD::EXTLOAD, MVT::i16, Expand);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i16, Expand);

  // Turn f32 extload into load + fextend
  setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);

  // Turn f64 truncstore into trunc + store.
  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  // Customize translation of memory addresses
  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);

  // Expand BR_CC into BRCOND
  setOperationAction(ISD::BR_CC, MVT::Other, Expand);

  // Expand SELECT_CC into SETCC
  setOperationAction(ISD::SELECT_CC, MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);

  // need to lower SETCC of RegPred into bitwise logic
  setOperationAction(ISD::SETCC, MVT::i1, Custom);

  setMinFunctionAlignment(2);

  // Compute derived properties from the register classes
  computeRegisterProperties();
}

MVT::SimpleValueType PTXTargetLowering::getSetCCResultType(EVT VT) const {
  return MVT::i1;
}

SDValue PTXTargetLowering::LowerOperation(SDValue Op, SelectionDAG &DAG) const {
  switch (Op.getOpcode()) {
    default:
      llvm_unreachable("Unimplemented operand");
    case ISD::SETCC:
      return LowerSETCC(Op, DAG);
    case ISD::GlobalAddress:
      return LowerGlobalAddress(Op, DAG);
  }
}

const char *PTXTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
    default:
      llvm_unreachable("Unknown opcode");
    case PTXISD::COPY_ADDRESS:
      return "PTXISD::COPY_ADDRESS";
    case PTXISD::LOAD_PARAM:
      return "PTXISD::LOAD_PARAM";
    case PTXISD::STORE_PARAM:
      return "PTXISD::STORE_PARAM";
    case PTXISD::EXIT:
      return "PTXISD::EXIT";
    case PTXISD::RET:
      return "PTXISD::RET";
  }
}

//===----------------------------------------------------------------------===//
//                      Custom Lower Operation
//===----------------------------------------------------------------------===//

SDValue PTXTargetLowering::LowerSETCC(SDValue Op, SelectionDAG &DAG) const {
  assert(Op.getValueType() == MVT::i1 && "SetCC type must be 1-bit integer");
  SDValue Op0 = Op.getOperand(0);
  SDValue Op1 = Op.getOperand(1);
  SDValue Op2 = Op.getOperand(2);
  DebugLoc dl = Op.getDebugLoc();
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(2))->get();

  // Look for X == 0, X == 1, X != 0, or X != 1  
  // We can simplify these to bitwise logic

  if (Op1.getOpcode() == ISD::Constant &&
      (cast<ConstantSDNode>(Op1)->getZExtValue() == 1 ||
       cast<ConstantSDNode>(Op1)->isNullValue()) &&
      (CC == ISD::SETEQ || CC == ISD::SETNE)) {

    return DAG.getNode(ISD::AND, dl, MVT::i1, Op0, Op1);
  }

  return DAG.getNode(ISD::SETCC, dl, MVT::i1, Op0, Op1, Op2);
}

SDValue PTXTargetLowering::
LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) const {
  EVT PtrVT = getPointerTy();
  DebugLoc dl = Op.getDebugLoc();
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Op)->getGlobal();

  assert(PtrVT.isSimple() && "Pointer must be to primitive type.");

  SDValue targetGlobal = DAG.getTargetGlobalAddress(GV, dl, PtrVT);
  SDValue movInstr = DAG.getNode(PTXISD::COPY_ADDRESS,
                                 dl,
                                 PtrVT.getSimpleVT(),
                                 targetGlobal);

  return movInstr;
}

//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

namespace {
struct argmap_entry {
  MVT::SimpleValueType VT;
  TargetRegisterClass *RC;
  TargetRegisterClass::iterator loc;

  argmap_entry(MVT::SimpleValueType _VT, TargetRegisterClass *_RC)
    : VT(_VT), RC(_RC), loc(_RC->begin()) {}

  void reset() { loc = RC->begin(); }
  bool operator==(MVT::SimpleValueType _VT) const { return VT == _VT; }
} argmap[] = {
  argmap_entry(MVT::i1,  PTX::RegPredRegisterClass),
  argmap_entry(MVT::i16, PTX::RegI16RegisterClass),
  argmap_entry(MVT::i32, PTX::RegI32RegisterClass),
  argmap_entry(MVT::i64, PTX::RegI64RegisterClass),
  argmap_entry(MVT::f32, PTX::RegF32RegisterClass),
  argmap_entry(MVT::f64, PTX::RegF64RegisterClass)
};
}                               // end anonymous namespace

SDValue PTXTargetLowering::
  LowerFormalArguments(SDValue Chain,
                       CallingConv::ID CallConv,
                       bool isVarArg,
                       const SmallVectorImpl<ISD::InputArg> &Ins,
                       DebugLoc dl,
                       SelectionDAG &DAG,
                       SmallVectorImpl<SDValue> &InVals) const {
  if (isVarArg) llvm_unreachable("PTX does not support varargs");

  MachineFunction &MF = DAG.getMachineFunction();
  const PTXSubtarget& ST = getTargetMachine().getSubtarget<PTXSubtarget>();
  PTXMachineFunctionInfo *MFI = MF.getInfo<PTXMachineFunctionInfo>();

  switch (CallConv) {
    default:
      llvm_unreachable("Unsupported calling convention");
      break;
    case CallingConv::PTX_Kernel:
      MFI->setKernel(true);
      break;
    case CallingConv::PTX_Device:
      MFI->setKernel(false);
      break;
  }

  // We do one of two things here:
  // IsKernel || SM >= 2.0  ->  Use param space for arguments
  // SM < 2.0               ->  Use registers for arguments
  if (MFI->isKernel() || ST.useParamSpaceForDeviceArgs()) {
    // We just need to emit the proper LOAD_PARAM ISDs
    for (unsigned i = 0, e = Ins.size(); i != e; ++i) {

      assert((!MFI->isKernel() || Ins[i].VT != MVT::i1) &&
             "Kernels cannot take pred operands");

      SDValue ArgValue = DAG.getNode(PTXISD::LOAD_PARAM, dl, Ins[i].VT, Chain,
                                     DAG.getTargetConstant(i, MVT::i32));
      InVals.push_back(ArgValue);

      // Instead of storing a physical register in our argument list, we just
      // store the total size of the parameter, in bits.  The ASM printer
      // knows how to process this.
      MFI->addArgReg(Ins[i].VT.getStoreSizeInBits());
    }
  }
  else {
    // For device functions, we use the PTX calling convention to do register
    // assignments then create CopyFromReg ISDs for the allocated registers

    SmallVector<CCValAssign, 16> ArgLocs;
    CCState CCInfo(CallConv, isVarArg, MF, getTargetMachine(), ArgLocs,
                   *DAG.getContext());

    CCInfo.AnalyzeFormalArguments(Ins, CC_PTX);

    for (unsigned i = 0, e = ArgLocs.size(); i != e; ++i) {

      CCValAssign&         VA    = ArgLocs[i];
      EVT                  RegVT = VA.getLocVT();
      TargetRegisterClass* TRC   = 0;

      assert(VA.isRegLoc() && "CCValAssign must be RegLoc");

      // Determine which register class we need
      if (RegVT == MVT::i1) {
        TRC = PTX::RegPredRegisterClass;
      }
      else if (RegVT == MVT::i16) {
        TRC = PTX::RegI16RegisterClass;
      }
      else if (RegVT == MVT::i32) {
        TRC = PTX::RegI32RegisterClass;
      }
      else if (RegVT == MVT::i64) {
        TRC = PTX::RegI64RegisterClass;
      }
      else if (RegVT == MVT::f32) {
        TRC = PTX::RegF32RegisterClass;
      }
      else if (RegVT == MVT::f64) {
        TRC = PTX::RegF64RegisterClass;
      }
      else {
        llvm_unreachable("Unknown parameter type");
      }

      unsigned Reg = MF.getRegInfo().createVirtualRegister(TRC);
      MF.getRegInfo().addLiveIn(VA.getLocReg(), Reg);

      SDValue ArgValue = DAG.getCopyFromReg(Chain, dl, Reg, RegVT);
      InVals.push_back(ArgValue);

      MFI->addArgReg(VA.getLocReg());
    }
  }

  return Chain;
}

SDValue PTXTargetLowering::
  LowerReturn(SDValue Chain,
              CallingConv::ID CallConv,
              bool isVarArg,
              const SmallVectorImpl<ISD::OutputArg> &Outs,
              const SmallVectorImpl<SDValue> &OutVals,
              DebugLoc dl,
              SelectionDAG &DAG) const {
  if (isVarArg) llvm_unreachable("PTX does not support varargs");

  switch (CallConv) {
    default:
      llvm_unreachable("Unsupported calling convention.");
    case CallingConv::PTX_Kernel:
      assert(Outs.size() == 0 && "Kernel must return void.");
      return DAG.getNode(PTXISD::EXIT, dl, MVT::Other, Chain);
    case CallingConv::PTX_Device:
      //assert(Outs.size() <= 1 && "Can at most return one value.");
      break;
  }

  MachineFunction& MF = DAG.getMachineFunction();
  PTXMachineFunctionInfo *MFI = MF.getInfo<PTXMachineFunctionInfo>();

  SDValue Flag;

  // Even though we could use the .param space for return arguments for
  // device functions if SM >= 2.0 and the number of return arguments is
  // only 1, we just always use registers since this makes the codegen
  // easier.
  SmallVector<CCValAssign, 16> RVLocs;
  CCState CCInfo(CallConv, isVarArg, DAG.getMachineFunction(),
  getTargetMachine(), RVLocs, *DAG.getContext());

  CCInfo.AnalyzeReturn(Outs, RetCC_PTX);

  for (unsigned i = 0, e = RVLocs.size(); i != e; ++i) {
    CCValAssign& VA  = RVLocs[i];

    assert(VA.isRegLoc() && "CCValAssign must be RegLoc");

    unsigned Reg = VA.getLocReg();

    DAG.getMachineFunction().getRegInfo().addLiveOut(Reg);

    Chain = DAG.getCopyToReg(Chain, dl, Reg, OutVals[i], Flag);

    // Guarantee that all emitted copies are stuck together,
    // avoiding something bad
    Flag = Chain.getValue(1);

    MFI->addRetReg(Reg);
  }

  if (Flag.getNode() == 0) {
    return DAG.getNode(PTXISD::RET, dl, MVT::Other, Chain);
  }
  else {
    return DAG.getNode(PTXISD::RET, dl, MVT::Other, Chain, Flag);
  }
}
