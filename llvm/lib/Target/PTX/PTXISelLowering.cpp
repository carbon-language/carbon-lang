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
#include "llvm/Function.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

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
  setBooleanVectorContents(ZeroOrOneBooleanContent); // FIXME: Is this correct?
  setMinFunctionAlignment(2);

  ////////////////////////////////////
  /////////// Expansion //////////////
  ////////////////////////////////////

  // (any/zero/sign) extload => load + (any/zero/sign) extend

  setLoadExtAction(ISD::EXTLOAD, MVT::i16, Expand);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i16, Expand);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i16, Expand);

  // f32 extload => load + fextend

  setLoadExtAction(ISD::EXTLOAD, MVT::f32, Expand);

  // f64 truncstore => trunc + store

  setTruncStoreAction(MVT::f64, MVT::f32, Expand);

  // sign_extend_inreg => sign_extend

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  // br_cc => brcond

  setOperationAction(ISD::BR_CC, MVT::Other, Expand);

  // select_cc => setcc

  setOperationAction(ISD::SELECT_CC, MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::f32, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::f64, Expand);

  ////////////////////////////////////
  //////////// Legal /////////////////
  ////////////////////////////////////

  setOperationAction(ISD::ConstantFP, MVT::f32, Legal);
  setOperationAction(ISD::ConstantFP, MVT::f64, Legal);

  ////////////////////////////////////
  //////////// Custom ////////////////
  ////////////////////////////////////

  // customise setcc to use bitwise logic if possible

  setOperationAction(ISD::SETCC, MVT::i1, Custom);

  // customize translation of memory addresses

  setOperationAction(ISD::GlobalAddress, MVT::i32, Custom);
  setOperationAction(ISD::GlobalAddress, MVT::i64, Custom);

  // Compute derived properties from the register classes
  computeRegisterProperties();
}

EVT PTXTargetLowering::getSetCCResultType(EVT VT) const {
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
    case PTXISD::READ_PARAM:
      return "PTXISD::READ_PARAM";
    case PTXISD::WRITE_PARAM:
      return "PTXISD::WRITE_PARAM";
    case PTXISD::EXIT:
      return "PTXISD::EXIT";
    case PTXISD::RET:
      return "PTXISD::RET";
    case PTXISD::CALL:
      return "PTXISD::CALL";
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
  PTXParamManager &PM = MFI->getParamManager();

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

      unsigned ParamSize = Ins[i].VT.getStoreSizeInBits();
      unsigned Param = PM.addArgumentParam(ParamSize);
      const std::string &ParamName = PM.getParamName(Param);
      SDValue ParamValue = DAG.getTargetExternalSymbol(ParamName.c_str(),
                                                       MVT::Other);
      SDValue ArgValue = DAG.getNode(PTXISD::LOAD_PARAM, dl, Ins[i].VT, Chain,
                                     ParamValue);
      InVals.push_back(ArgValue);
    }
  }
  else {
    for (unsigned i = 0, e = Ins.size(); i != e; ++i) {
      EVT                  RegVT = Ins[i].VT;
      TargetRegisterClass* TRC   = getRegClassFor(RegVT);

      // Use a unique index in the instruction to prevent instruction folding.
      // Yes, this is a hack.
      SDValue Index = DAG.getTargetConstant(i, MVT::i32);
      unsigned Reg = MF.getRegInfo().createVirtualRegister(TRC);
      SDValue ArgValue = DAG.getNode(PTXISD::READ_PARAM, dl, RegVT, Chain,
                                     Index);

      InVals.push_back(ArgValue);

      MFI->addArgReg(Reg);
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
      assert(Outs.size() <= 1 && "Can at most return one value.");
      break;
  }

  MachineFunction& MF = DAG.getMachineFunction();
  PTXMachineFunctionInfo *MFI = MF.getInfo<PTXMachineFunctionInfo>();
  PTXParamManager &PM = MFI->getParamManager();

  SDValue Flag;
  const PTXSubtarget& ST = getTargetMachine().getSubtarget<PTXSubtarget>();

  if (ST.useParamSpaceForDeviceArgs()) {
    assert(Outs.size() < 2 && "Device functions can return at most one value");

    if (Outs.size() == 1) {
      unsigned ParamSize = OutVals[0].getValueType().getSizeInBits();
      unsigned Param = PM.addReturnParam(ParamSize);
      const std::string &ParamName = PM.getParamName(Param);
      SDValue ParamValue = DAG.getTargetExternalSymbol(ParamName.c_str(),
                                                       MVT::Other);
      Chain = DAG.getNode(PTXISD::STORE_PARAM, dl, MVT::Other, Chain,
                          ParamValue, OutVals[0]);
    }
  } else {
    for (unsigned i = 0, e = Outs.size(); i != e; ++i) {
      EVT                  RegVT = Outs[i].VT;
      TargetRegisterClass* TRC = 0;

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

      SDValue Copy = DAG.getCopyToReg(Chain, dl, Reg, OutVals[i]/*, Flag*/);
      SDValue OutReg = DAG.getRegister(Reg, RegVT);

      Chain = DAG.getNode(PTXISD::WRITE_PARAM, dl, MVT::Other, Copy, OutReg);

      MFI->addRetReg(Reg);
    }
  }

  if (Flag.getNode() == 0) {
    return DAG.getNode(PTXISD::RET, dl, MVT::Other, Chain);
  }
  else {
    return DAG.getNode(PTXISD::RET, dl, MVT::Other, Chain, Flag);
  }
}

SDValue
PTXTargetLowering::LowerCall(SDValue Chain, SDValue Callee,
                             CallingConv::ID CallConv, bool isVarArg,
                             bool &isTailCall,
                             const SmallVectorImpl<ISD::OutputArg> &Outs,
                             const SmallVectorImpl<SDValue> &OutVals,
                             const SmallVectorImpl<ISD::InputArg> &Ins,
                             DebugLoc dl, SelectionDAG &DAG,
                             SmallVectorImpl<SDValue> &InVals) const {

  MachineFunction& MF = DAG.getMachineFunction();
  PTXMachineFunctionInfo *PTXMFI = MF.getInfo<PTXMachineFunctionInfo>();
  PTXParamManager &PM = PTXMFI->getParamManager();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  
  assert(getTargetMachine().getSubtarget<PTXSubtarget>().callsAreHandled() &&
         "Calls are not handled for the target device");

  // Identify the callee function
  const GlobalValue *GV = cast<GlobalAddressSDNode>(Callee)->getGlobal();
  const Function *function = cast<Function>(GV);
  
  // allow non-device calls only for printf
  bool isPrintf = function->getName() == "printf" || function->getName() == "puts";	
  
  assert((isPrintf || function->getCallingConv() == CallingConv::PTX_Device) &&
			 "PTX function calls must be to PTX device functions");
  
  unsigned outSize = isPrintf ? 2 : Outs.size();
  
  std::vector<SDValue> Ops;
  // The layout of the ops will be [Chain, #Ins, Ins, Callee, #Outs, Outs]
  Ops.resize(outSize + Ins.size() + 4);

  Ops[0] = Chain;

  // Identify the callee function
  Callee = DAG.getTargetGlobalAddress(GV, dl, getPointerTy());
  Ops[Ins.size()+2] = Callee;

  // #Outs
  Ops[Ins.size()+3] = DAG.getTargetConstant(outSize, MVT::i32);
  
  if (isPrintf) {
    // first argument is the address of the global string variable in memory
    unsigned Param0 = PM.addLocalParam(getPointerTy().getSizeInBits());
    SDValue ParamValue0 = DAG.getTargetExternalSymbol(PM.getParamName(Param0).c_str(),
                                                      MVT::Other);
    Chain = DAG.getNode(PTXISD::STORE_PARAM, dl, MVT::Other, Chain,
                        ParamValue0, OutVals[0]);
    Ops[Ins.size()+4] = ParamValue0;
      
    // alignment is the maximum size of all the arguments
    unsigned alignment = 0;
    for (unsigned i = 1; i < OutVals.size(); ++i) {
      alignment = std::max(alignment, 
    		               OutVals[i].getValueType().getSizeInBits());
    }

    // size is the alignment multiplied by the number of arguments
    unsigned size = alignment * (OutVals.size() - 1);
  
    // second argument is the address of the stack object (unless no arguments)
    unsigned Param1 = PM.addLocalParam(getPointerTy().getSizeInBits());
    SDValue ParamValue1 = DAG.getTargetExternalSymbol(PM.getParamName(Param1).c_str(),
                                                      MVT::Other);
    Ops[Ins.size()+5] = ParamValue1;
    
    if (size > 0)
    {
      // create a local stack object to store the arguments
      unsigned StackObject = MFI->CreateStackObject(size / 8, alignment / 8, false);
      SDValue FrameIndex = DAG.getFrameIndex(StackObject, getPointerTy());
	  
      // store each of the arguments to the stack in turn
      for (unsigned int i = 1; i != OutVals.size(); i++) {
        SDValue FrameAddr = DAG.getNode(ISD::ADD, dl, getPointerTy(), FrameIndex, DAG.getTargetConstant((i - 1) * 8, getPointerTy()));
        Chain = DAG.getStore(Chain, dl, OutVals[i], FrameAddr,
                             MachinePointerInfo(),
                             false, false, 0);
      }

      // copy the address of the local frame index to get the address in non-local space
      SDValue genericAddr = DAG.getNode(PTXISD::COPY_ADDRESS, dl, getPointerTy(), FrameIndex);

      // store this address in the second argument
      Chain = DAG.getNode(PTXISD::STORE_PARAM, dl, MVT::Other, Chain, ParamValue1, genericAddr);
    }
  }
  else
  {
	  // Generate STORE_PARAM nodes for each function argument.  In PTX, function
	  // arguments are explicitly stored into .param variables and passed as
	  // arguments. There is no register/stack-based calling convention in PTX.
	  for (unsigned i = 0; i != OutVals.size(); ++i) {
		unsigned Size = OutVals[i].getValueType().getSizeInBits();
		unsigned Param = PM.addLocalParam(Size);
		const std::string &ParamName = PM.getParamName(Param);
		SDValue ParamValue = DAG.getTargetExternalSymbol(ParamName.c_str(),
														 MVT::Other);
		Chain = DAG.getNode(PTXISD::STORE_PARAM, dl, MVT::Other, Chain,
							ParamValue, OutVals[i]);
		Ops[i+Ins.size()+4] = ParamValue;
	  }
  }
  
  std::vector<SDValue> InParams;

  // Generate list of .param variables to hold the return value(s).
  Ops[1] = DAG.getTargetConstant(Ins.size(), MVT::i32);
  for (unsigned i = 0; i < Ins.size(); ++i) {
    unsigned Size = Ins[i].VT.getStoreSizeInBits();
    unsigned Param = PM.addLocalParam(Size);
    const std::string &ParamName = PM.getParamName(Param);
    SDValue ParamValue = DAG.getTargetExternalSymbol(ParamName.c_str(),
                                                     MVT::Other);
    Ops[i+2] = ParamValue;
    InParams.push_back(ParamValue);
  }

  Ops[0] = Chain;

  // Create the CALL node.
  Chain = DAG.getNode(PTXISD::CALL, dl, MVT::Other, &Ops[0], Ops.size());

  // Create the LOAD_PARAM nodes that retrieve the function return value(s).
  for (unsigned i = 0; i < Ins.size(); ++i) {
    SDValue Load = DAG.getNode(PTXISD::LOAD_PARAM, dl, Ins[i].VT, Chain,
                               InParams[i]);
    InVals.push_back(Load);
  }

  return Chain;
}

unsigned PTXTargetLowering::getNumRegisters(LLVMContext &Context, EVT VT) {
  // All arguments consist of one "register," regardless of the type.
  return 1;
}

