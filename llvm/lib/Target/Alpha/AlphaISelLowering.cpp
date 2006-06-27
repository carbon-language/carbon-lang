//===-- AlphaISelLowering.cpp - Alpha DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Andrew Lenharth and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the AlphaISelLowering class.
//
//===----------------------------------------------------------------------===//

#include "AlphaISelLowering.h"
#include "AlphaTargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Support/CommandLine.h"
#include <iostream>

using namespace llvm;

/// AddLiveIn - This helper function adds the specified physical register to the
/// MachineFunction as a live in value.  It also creates a corresponding virtual
/// register for it.
static unsigned AddLiveIn(MachineFunction &MF, unsigned PReg,
                          TargetRegisterClass *RC) {
  assert(RC->contains(PReg) && "Not the correct regclass!");
  unsigned VReg = MF.getSSARegMap()->createVirtualRegister(RC);
  MF.addLiveIn(PReg, VReg);
  return VReg;
}

AlphaTargetLowering::AlphaTargetLowering(TargetMachine &TM) : TargetLowering(TM) {
  // Set up the TargetLowering object.
  //I am having problems with shr n ubyte 1
  setShiftAmountType(MVT::i64);
  setSetCCResultType(MVT::i64);
  setSetCCResultContents(ZeroOrOneSetCCResult);
  
  addRegisterClass(MVT::i64, Alpha::GPRCRegisterClass);
  addRegisterClass(MVT::f64, Alpha::F8RCRegisterClass);
  addRegisterClass(MVT::f32, Alpha::F4RCRegisterClass);
  
  setOperationAction(ISD::BRIND,        MVT::i64,   Expand);
  setOperationAction(ISD::BR_CC,        MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC,    MVT::Other, Expand);
  
  setOperationAction(ISD::EXTLOAD, MVT::i1,  Promote);
  setOperationAction(ISD::EXTLOAD, MVT::f32, Expand);
  
  setOperationAction(ISD::ZEXTLOAD, MVT::i1,  Promote);
  setOperationAction(ISD::ZEXTLOAD, MVT::i32, Expand);
  
  setOperationAction(ISD::SEXTLOAD, MVT::i1,  Promote);
  setOperationAction(ISD::SEXTLOAD, MVT::i8,  Expand);
  setOperationAction(ISD::SEXTLOAD, MVT::i16, Expand);
  
  setOperationAction(ISD::TRUNCSTORE, MVT::i1, Promote);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  setOperationAction(ISD::FREM, MVT::f32, Expand);
  setOperationAction(ISD::FREM, MVT::f64, Expand);
  
  setOperationAction(ISD::UINT_TO_FP, MVT::i64, Expand);
  setOperationAction(ISD::SINT_TO_FP, MVT::i64, Custom);
  setOperationAction(ISD::FP_TO_UINT, MVT::i64, Expand);
  setOperationAction(ISD::FP_TO_SINT, MVT::i64, Custom);

  if (!TM.getSubtarget<AlphaSubtarget>().hasCT()) {
    setOperationAction(ISD::CTPOP    , MVT::i64  , Expand);
    setOperationAction(ISD::CTTZ     , MVT::i64  , Expand);
    setOperationAction(ISD::CTLZ     , MVT::i64  , Expand);
  }
  setOperationAction(ISD::BSWAP    , MVT::i64, Expand);
  setOperationAction(ISD::ROTL     , MVT::i64, Expand);
  setOperationAction(ISD::ROTR     , MVT::i64, Expand);
  
  setOperationAction(ISD::SREM     , MVT::i64, Custom);
  setOperationAction(ISD::UREM     , MVT::i64, Custom);
  setOperationAction(ISD::SDIV     , MVT::i64, Custom);
  setOperationAction(ISD::UDIV     , MVT::i64, Custom);

  setOperationAction(ISD::MEMMOVE  , MVT::Other, Expand);
  setOperationAction(ISD::MEMSET   , MVT::Other, Expand);
  setOperationAction(ISD::MEMCPY   , MVT::Other, Expand);
  
  // We don't support sin/cos/sqrt
  setOperationAction(ISD::FSIN , MVT::f64, Expand);
  setOperationAction(ISD::FCOS , MVT::f64, Expand);
  setOperationAction(ISD::FSIN , MVT::f32, Expand);
  setOperationAction(ISD::FCOS , MVT::f32, Expand);

  setOperationAction(ISD::FSQRT, MVT::f64, Expand);
  setOperationAction(ISD::FSQRT, MVT::f32, Expand);
  
  // FIXME: Alpha supports fcopysign natively!?
  setOperationAction(ISD::FCOPYSIGN, MVT::f64, Expand);
  setOperationAction(ISD::FCOPYSIGN, MVT::f32, Expand);

  setOperationAction(ISD::SETCC, MVT::f32, Promote);

  // We don't have line number support yet.
  setOperationAction(ISD::LOCATION, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LABEL, MVT::Other, Expand);

  // Not implemented yet.
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand); 
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);
  setOperationAction(ISD::DYNAMIC_STACKALLOC, MVT::i64, Expand);

  // We want to legalize GlobalAddress and ConstantPool and
  // ExternalSymbols nodes into the appropriate instructions to
  // materialize the address.
  setOperationAction(ISD::GlobalAddress,  MVT::i64, Custom);
  setOperationAction(ISD::ConstantPool,   MVT::i64, Custom);
  setOperationAction(ISD::ExternalSymbol, MVT::i64, Custom);

  setOperationAction(ISD::VASTART, MVT::Other, Custom);
  setOperationAction(ISD::VAEND,   MVT::Other, Expand);
  setOperationAction(ISD::VACOPY,  MVT::Other, Custom);
  setOperationAction(ISD::VAARG,   MVT::Other, Custom);
  setOperationAction(ISD::VAARG,   MVT::i32,   Custom);

  setOperationAction(ISD::RET,     MVT::Other, Custom);

  setStackPointerRegisterToSaveRestore(Alpha::R30);

  setOperationAction(ISD::ConstantFP, MVT::f64, Expand);
  setOperationAction(ISD::ConstantFP, MVT::f32, Expand);
  addLegalFPImmediate(+0.0); //F31
  addLegalFPImmediate(-0.0); //-F31

  computeRegisterProperties();

  useITOF = TM.getSubtarget<AlphaSubtarget>().hasF2I();
}

const char *AlphaTargetLowering::getTargetNodeName(unsigned Opcode) const {
  switch (Opcode) {
  default: return 0;
  case AlphaISD::ITOFT_: return "Alpha::ITOFT_";
  case AlphaISD::FTOIT_: return "Alpha::FTOIT_";
  case AlphaISD::CVTQT_: return "Alpha::CVTQT_";
  case AlphaISD::CVTQS_: return "Alpha::CVTQS_";
  case AlphaISD::CVTTQ_: return "Alpha::CVTTQ_";
  case AlphaISD::GPRelHi: return "Alpha::GPRelHi";
  case AlphaISD::GPRelLo: return "Alpha::GPRelLo";
  case AlphaISD::RelLit: return "Alpha::RelLit";
  case AlphaISD::GlobalBaseReg: return "Alpha::GlobalBaseReg";
  case AlphaISD::GlobalRetAddr: return "Alpha::GlobalRetAddr";
  case AlphaISD::CALL:   return "Alpha::CALL";
  case AlphaISD::DivCall: return "Alpha::DivCall";
  case AlphaISD::RET_FLAG: return "Alpha::RET_FLAG";
  }
}

//http://www.cs.arizona.edu/computer.help/policy/DIGITAL_unix/AA-PY8AC-TET1_html/callCH3.html#BLOCK21

//For now, just use variable size stack frame format

//In a standard call, the first six items are passed in registers $16
//- $21 and/or registers $f16 - $f21. (See Section 4.1.2 for details
//of argument-to-register correspondence.) The remaining items are
//collected in a memory argument list that is a naturally aligned
//array of quadwords. In a standard call, this list, if present, must
//be passed at 0(SP).
//7 ... n         0(SP) ... (n-7)*8(SP)

// //#define FP    $15
// //#define RA    $26
// //#define PV    $27
// //#define GP    $29
// //#define SP    $30

static SDOperand LowerFORMAL_ARGUMENTS(SDOperand Op, SelectionDAG &DAG,
				       int &VarArgsBase,
				       int &VarArgsOffset,
				       unsigned int &GP,
				       unsigned int &RA) {
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  SSARegMap *RegMap = MF.getSSARegMap();
  std::vector<SDOperand> ArgValues;
  SDOperand Root = Op.getOperand(0);

  GP = AddLiveIn(MF, Alpha::R29, &Alpha::GPRCRegClass);
  RA = AddLiveIn(MF, Alpha::R26, &Alpha::GPRCRegClass);

  unsigned args_int[] = {
    Alpha::R16, Alpha::R17, Alpha::R18, Alpha::R19, Alpha::R20, Alpha::R21};
  unsigned args_float[] = {
    Alpha::F16, Alpha::F17, Alpha::F18, Alpha::F19, Alpha::F20, Alpha::F21};
  
  for (unsigned ArgNo = 0, e = Op.Val->getNumValues()-1; ArgNo != e; ++ArgNo) {
    SDOperand argt;
    MVT::ValueType ObjectVT = Op.getValue(ArgNo).getValueType();
    SDOperand ArgVal;

    if (ArgNo  < 6) {
      unsigned Vreg;
      switch (ObjectVT) {
      default:
        std::cerr << "Unknown Type " << ObjectVT << "\n";
        abort();
      case MVT::f64:
        args_float[ArgNo] = AddLiveIn(MF, args_float[ArgNo], 
				      &Alpha::F8RCRegClass);
        ArgVal = DAG.getCopyFromReg(Root, args_float[ArgNo], ObjectVT);
        break;
      case MVT::f32:
        args_float[ArgNo] = AddLiveIn(MF, args_float[ArgNo], 
				      &Alpha::F4RCRegClass);
        ArgVal = DAG.getCopyFromReg(Root, args_float[ArgNo], ObjectVT);
        break;
      case MVT::i64:
        args_int[ArgNo] = AddLiveIn(MF, args_int[ArgNo], 
				    &Alpha::GPRCRegClass);
        ArgVal = DAG.getCopyFromReg(Root, args_int[ArgNo], MVT::i64);
        break;
      }
    } else { //more args
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(8, 8 * (ArgNo - 6));

      // Create the SelectionDAG nodes corresponding to a load
      //from this parameter
      SDOperand FIN = DAG.getFrameIndex(FI, MVT::i64);
      ArgVal = DAG.getLoad(ObjectVT, Root, FIN, DAG.getSrcValue(NULL));
    }
    ArgValues.push_back(ArgVal);
  }

  // If the functions takes variable number of arguments, copy all regs to stack
  bool isVarArg = cast<ConstantSDNode>(Op.getOperand(2))->getValue() != 0;
  if (isVarArg) {
    VarArgsOffset = (Op.Val->getNumValues()-1) * 8;
    std::vector<SDOperand> LS;
    for (int i = 0; i < 6; ++i) {
      if (MRegisterInfo::isPhysicalRegister(args_int[i]))
        args_int[i] = AddLiveIn(MF, args_int[i], &Alpha::GPRCRegClass);
      SDOperand argt = DAG.getCopyFromReg(Root, args_int[i], MVT::i64);
      int FI = MFI->CreateFixedObject(8, -8 * (6 - i));
      if (i == 0) VarArgsBase = FI;
      SDOperand SDFI = DAG.getFrameIndex(FI, MVT::i64);
      LS.push_back(DAG.getNode(ISD::STORE, MVT::Other, Root, argt,
                               SDFI, DAG.getSrcValue(NULL)));

      if (MRegisterInfo::isPhysicalRegister(args_float[i]))
        args_float[i] = AddLiveIn(MF, args_float[i], &Alpha::F8RCRegClass);
      argt = DAG.getCopyFromReg(Root, args_float[i], MVT::f64);
      FI = MFI->CreateFixedObject(8, - 8 * (12 - i));
      SDFI = DAG.getFrameIndex(FI, MVT::i64);
      LS.push_back(DAG.getNode(ISD::STORE, MVT::Other, Root, argt,
                               SDFI, DAG.getSrcValue(NULL)));
    }

    //Set up a token factor with all the stack traffic
    Root = DAG.getNode(ISD::TokenFactor, MVT::Other, LS);
  }

  ArgValues.push_back(Root);

  // Return the new list of results.
  std::vector<MVT::ValueType> RetVT(Op.Val->value_begin(),
                                    Op.Val->value_end());
  return DAG.getNode(ISD::MERGE_VALUES, RetVT, ArgValues);
}

static SDOperand LowerRET(SDOperand Op, SelectionDAG &DAG, unsigned int RA) {
  SDOperand Copy = DAG.getCopyToReg(Op.getOperand(0), Alpha::R26, 
				    DAG.getNode(AlphaISD::GlobalRetAddr, MVT::i64),
				    SDOperand());
  switch (Op.getNumOperands()) {
  default:
    assert(0 && "Do not know how to return this many arguments!");
    abort();
  case 1: 
    break;
    //return SDOperand(); // ret void is legal
  case 3: {
    MVT::ValueType ArgVT = Op.getOperand(1).getValueType();
    unsigned ArgReg;
    if (MVT::isInteger(ArgVT))
      ArgReg = Alpha::R0;
    else {
      assert(MVT::isFloatingPoint(ArgVT));
      ArgReg = Alpha::F0;
    }
    Copy = DAG.getCopyToReg(Copy, ArgReg, Op.getOperand(1), Copy.getValue(1));
    if(DAG.getMachineFunction().liveout_empty())
      DAG.getMachineFunction().addLiveOut(ArgReg);
    break;
  }
  }
  return DAG.getNode(AlphaISD::RET_FLAG, MVT::Other, Copy, Copy.getValue(1));
}

std::pair<SDOperand, SDOperand>
AlphaTargetLowering::LowerCallTo(SDOperand Chain,
                                 const Type *RetTy, bool isVarArg,
                                 unsigned CallingConv, bool isTailCall,
                                 SDOperand Callee, ArgListTy &Args,
                                 SelectionDAG &DAG) {
  int NumBytes = 0;
  if (Args.size() > 6)
    NumBytes = (Args.size() - 6) * 8;

  Chain = DAG.getCALLSEQ_START(Chain,
                               DAG.getConstant(NumBytes, getPointerTy()));
  std::vector<SDOperand> args_to_use;
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
  {
    switch (getValueType(Args[i].second)) {
    default: assert(0 && "Unexpected ValueType for argument!");
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      // Promote the integer to 64 bits.  If the input type is signed use a
      // sign extend, otherwise use a zero extend.
      if (Args[i].second->isSigned())
        Args[i].first = DAG.getNode(ISD::SIGN_EXTEND, MVT::i64, Args[i].first);
      else
        Args[i].first = DAG.getNode(ISD::ZERO_EXTEND, MVT::i64, Args[i].first);
      break;
    case MVT::i64:
    case MVT::f64:
    case MVT::f32:
      break;
    }
    args_to_use.push_back(Args[i].first);
  }

  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);
  MVT::ValueType ActualRetTyVT = RetTyVT;
  if (RetTyVT >= MVT::i1 && RetTyVT <= MVT::i32)
    ActualRetTyVT = MVT::i64;

  if (RetTyVT != MVT::isVoid)
    RetVals.push_back(ActualRetTyVT);
  RetVals.push_back(MVT::Other);

  std::vector<SDOperand> Ops;
  Ops.push_back(Chain);
  Ops.push_back(Callee);
  Ops.insert(Ops.end(), args_to_use.begin(), args_to_use.end());
  SDOperand TheCall = DAG.getNode(AlphaISD::CALL, RetVals, Ops);
  Chain = TheCall.getValue(RetTyVT != MVT::isVoid);
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  SDOperand RetVal = TheCall;

  if (RetTyVT != ActualRetTyVT) {
    RetVal = DAG.getNode(RetTy->isSigned() ? ISD::AssertSext : ISD::AssertZext,
                         MVT::i64, RetVal, DAG.getValueType(RetTyVT));
    RetVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, RetVal);
  }

  return std::make_pair(RetVal, Chain);
}

void AlphaTargetLowering::restoreGP(MachineBasicBlock* BB)
{
  BuildMI(BB, Alpha::BIS, 2, Alpha::R29).addReg(GP).addReg(GP);
}
void AlphaTargetLowering::restoreRA(MachineBasicBlock* BB)
{
  BuildMI(BB, Alpha::BIS, 2, Alpha::R26).addReg(RA).addReg(RA);
}

static int getUID()
{
  static int id = 0;
  return ++id;
}

/// LowerOperation - Provide custom lowering hooks for some operations.
///
SDOperand AlphaTargetLowering::LowerOperation(SDOperand Op, SelectionDAG &DAG) {
  switch (Op.getOpcode()) {
  default: assert(0 && "Wasn't expecting to be able to lower this!");
  case ISD::FORMAL_ARGUMENTS: return LowerFORMAL_ARGUMENTS(Op, DAG, 
							   VarArgsBase,
							   VarArgsOffset,
							   GP, RA);
  case ISD::RET: return LowerRET(Op,DAG, getVRegRA());
  case ISD::SINT_TO_FP: {
    assert(MVT::i64 == Op.getOperand(0).getValueType() && 
           "Unhandled SINT_TO_FP type in custom expander!");
    SDOperand LD;
    bool isDouble = MVT::f64 == Op.getValueType();
    if (useITOF) {
      LD = DAG.getNode(AlphaISD::ITOFT_, MVT::f64, Op.getOperand(0));
    } else {
      int FrameIdx =
        DAG.getMachineFunction().getFrameInfo()->CreateStackObject(8, 8);
      SDOperand FI = DAG.getFrameIndex(FrameIdx, MVT::i64);
      SDOperand ST = DAG.getNode(ISD::STORE, MVT::Other, DAG.getEntryNode(),
                                 Op.getOperand(0), FI, DAG.getSrcValue(0));
      LD = DAG.getLoad(MVT::f64, ST, FI, DAG.getSrcValue(0));
      }
    SDOperand FP = DAG.getNode(isDouble?AlphaISD::CVTQT_:AlphaISD::CVTQS_,
                               isDouble?MVT::f64:MVT::f32, LD);
    return FP;
  }
  case ISD::FP_TO_SINT: {
    bool isDouble = MVT::f64 == Op.getOperand(0).getValueType();
    SDOperand src = Op.getOperand(0);

    if (!isDouble) //Promote
      src = DAG.getNode(ISD::FP_EXTEND, MVT::f64, src);
    
    src = DAG.getNode(AlphaISD::CVTTQ_, MVT::f64, src);

    if (useITOF) {
      return DAG.getNode(AlphaISD::FTOIT_, MVT::i64, src);
    } else {
      int FrameIdx =
        DAG.getMachineFunction().getFrameInfo()->CreateStackObject(8, 8);
      SDOperand FI = DAG.getFrameIndex(FrameIdx, MVT::i64);
      SDOperand ST = DAG.getNode(ISD::STORE, MVT::Other, DAG.getEntryNode(),
                                 src, FI, DAG.getSrcValue(0));
      return DAG.getLoad(MVT::i64, ST, FI, DAG.getSrcValue(0));
      }
  }
  case ISD::ConstantPool: {
    ConstantPoolSDNode *CP = cast<ConstantPoolSDNode>(Op);
    Constant *C = CP->get();
    SDOperand CPI = DAG.getTargetConstantPool(C, MVT::i64, CP->getAlignment());
    
    SDOperand Hi = DAG.getNode(AlphaISD::GPRelHi,  MVT::i64, CPI,
			       DAG.getNode(AlphaISD::GlobalBaseReg, MVT::i64));
    SDOperand Lo = DAG.getNode(AlphaISD::GPRelLo, MVT::i64, CPI, Hi);
    return Lo;
  }
  case ISD::GlobalAddress: {
    GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op);
    GlobalValue *GV = GSDN->getGlobal();
    SDOperand GA = DAG.getTargetGlobalAddress(GV, MVT::i64, GSDN->getOffset());

    //    if (!GV->hasWeakLinkage() && !GV->isExternal() && !GV->hasLinkOnceLinkage()) {
    if (GV->hasInternalLinkage()) {
      SDOperand Hi = DAG.getNode(AlphaISD::GPRelHi,  MVT::i64, GA,
				 DAG.getNode(AlphaISD::GlobalBaseReg, MVT::i64));
      SDOperand Lo = DAG.getNode(AlphaISD::GPRelLo, MVT::i64, GA, Hi);
      return Lo;
    } else
      return DAG.getNode(AlphaISD::RelLit, MVT::i64, GA, DAG.getNode(AlphaISD::GlobalBaseReg, MVT::i64));
  }
  case ISD::ExternalSymbol: {
    return DAG.getNode(AlphaISD::RelLit, MVT::i64, 
		       DAG.getTargetExternalSymbol(cast<ExternalSymbolSDNode>(Op)->getSymbol(), MVT::i64),
		       DAG.getNode(AlphaISD::GlobalBaseReg, MVT::i64));
  }

  case ISD::UREM:
  case ISD::SREM:
    //Expand only on constant case
    if (Op.getOperand(1).getOpcode() == ISD::Constant) {
      MVT::ValueType VT = Op.Val->getValueType(0);
      unsigned Opc = Op.Val->getOpcode() == ISD::UREM ? ISD::UDIV : ISD::SDIV;
      SDOperand Tmp1 = Op.Val->getOpcode() == ISD::UREM ?
	BuildUDIV(Op.Val, DAG, NULL) :
	BuildSDIV(Op.Val, DAG, NULL);
      Tmp1 = DAG.getNode(ISD::MUL, VT, Tmp1, Op.getOperand(1));
      Tmp1 = DAG.getNode(ISD::SUB, VT, Op.getOperand(0), Tmp1);
      return Tmp1;
    }
    //fall through
  case ISD::SDIV:
  case ISD::UDIV:
    if (MVT::isInteger(Op.getValueType())) {
      if (Op.getOperand(1).getOpcode() == ISD::Constant)
	return Op.getOpcode() == ISD::SDIV ? BuildSDIV(Op.Val, DAG, NULL) 
	  : BuildUDIV(Op.Val, DAG, NULL);
      const char* opstr = 0;
      switch(Op.getOpcode()) {
      case ISD::UREM: opstr = "__remqu"; break;
      case ISD::SREM: opstr = "__remq";  break;
      case ISD::UDIV: opstr = "__divqu"; break;
      case ISD::SDIV: opstr = "__divq";  break;
      }
      SDOperand Tmp1 = Op.getOperand(0),
        Tmp2 = Op.getOperand(1),
        Addr = DAG.getExternalSymbol(opstr, MVT::i64);
      return DAG.getNode(AlphaISD::DivCall, MVT::i64, Addr, Tmp1, Tmp2);
    }
    break;

  case ISD::VAARG: {
    SDOperand Chain = Op.getOperand(0);
    SDOperand VAListP = Op.getOperand(1);
    SDOperand VAListS = Op.getOperand(2);
    
    SDOperand Base = DAG.getLoad(MVT::i64, Chain, VAListP, VAListS);
    SDOperand Tmp = DAG.getNode(ISD::ADD, MVT::i64, VAListP,
                                DAG.getConstant(8, MVT::i64));
    SDOperand Offset = DAG.getExtLoad(ISD::SEXTLOAD, MVT::i64, Base.getValue(1),
                                      Tmp, DAG.getSrcValue(0), MVT::i32);
    SDOperand DataPtr = DAG.getNode(ISD::ADD, MVT::i64, Base, Offset);
    if (MVT::isFloatingPoint(Op.getValueType()))
    {
      //if fp && Offset < 6*8, then subtract 6*8 from DataPtr
      SDOperand FPDataPtr = DAG.getNode(ISD::SUB, MVT::i64, DataPtr,
                                        DAG.getConstant(8*6, MVT::i64));
      SDOperand CC = DAG.getSetCC(MVT::i64, Offset,
                                  DAG.getConstant(8*6, MVT::i64), ISD::SETLT);
      DataPtr = DAG.getNode(ISD::SELECT, MVT::i64, CC, FPDataPtr, DataPtr);
    }

    SDOperand NewOffset = DAG.getNode(ISD::ADD, MVT::i64, Offset,
                                      DAG.getConstant(8, MVT::i64));
    SDOperand Update = DAG.getNode(ISD::TRUNCSTORE, MVT::Other,
                                   Offset.getValue(1), NewOffset,
                                   Tmp, DAG.getSrcValue(0),
                                   DAG.getValueType(MVT::i32));
    
    SDOperand Result;
    if (Op.getValueType() == MVT::i32)
      Result = DAG.getExtLoad(ISD::SEXTLOAD, MVT::i64, Update, DataPtr,
                              DAG.getSrcValue(0), MVT::i32);
    else
      Result = DAG.getLoad(Op.getValueType(), Update, DataPtr, 
                           DAG.getSrcValue(0));
    return Result;
  }
  case ISD::VACOPY: {
    SDOperand Chain = Op.getOperand(0);
    SDOperand DestP = Op.getOperand(1);
    SDOperand SrcP = Op.getOperand(2);
    SDOperand DestS = Op.getOperand(3);
    SDOperand SrcS = Op.getOperand(4);
    
    SDOperand Val = DAG.getLoad(getPointerTy(), Chain, SrcP, SrcS);
    SDOperand Result = DAG.getNode(ISD::STORE, MVT::Other, Val.getValue(1), Val,
                                   DestP, DestS);
    SDOperand NP = DAG.getNode(ISD::ADD, MVT::i64, SrcP, 
                               DAG.getConstant(8, MVT::i64));
    Val = DAG.getExtLoad(ISD::SEXTLOAD, MVT::i64, Result, NP,
                         DAG.getSrcValue(0), MVT::i32);
    SDOperand NPD = DAG.getNode(ISD::ADD, MVT::i64, DestP,
                                DAG.getConstant(8, MVT::i64));
    return DAG.getNode(ISD::TRUNCSTORE, MVT::Other, Val.getValue(1),
                       Val, NPD, DAG.getSrcValue(0),DAG.getValueType(MVT::i32));
  }
  case ISD::VASTART: {
    SDOperand Chain = Op.getOperand(0);
    SDOperand VAListP = Op.getOperand(1);
    SDOperand VAListS = Op.getOperand(2);
    
    // vastart stores the address of the VarArgsBase and VarArgsOffset
    SDOperand FR  = DAG.getFrameIndex(VarArgsBase, MVT::i64);
    SDOperand S1  = DAG.getNode(ISD::STORE, MVT::Other, Chain, FR, VAListP,
                                VAListS);
    SDOperand SA2 = DAG.getNode(ISD::ADD, MVT::i64, VAListP,
                                DAG.getConstant(8, MVT::i64));
    return DAG.getNode(ISD::TRUNCSTORE, MVT::Other, S1,
                       DAG.getConstant(VarArgsOffset, MVT::i64), SA2,
                       DAG.getSrcValue(0), DAG.getValueType(MVT::i32));
  }
  }

  return SDOperand();
}

SDOperand AlphaTargetLowering::CustomPromoteOperation(SDOperand Op, 
                                                      SelectionDAG &DAG) {
  assert(Op.getValueType() == MVT::i32 && 
         Op.getOpcode() == ISD::VAARG &&
         "Unknown node to custom promote!");
  
  // The code in LowerOperation already handles i32 vaarg
  return LowerOperation(Op, DAG);
}


//Inline Asm

/// getConstraintType - Given a constraint letter, return the type of
/// constraint it is for this target.
AlphaTargetLowering::ConstraintType 
AlphaTargetLowering::getConstraintType(char ConstraintLetter) const {
  switch (ConstraintLetter) {
  default: break;
  case 'f':
  case 'r':
    return C_RegisterClass;
  }  
  return TargetLowering::getConstraintType(ConstraintLetter);
}

std::vector<unsigned> AlphaTargetLowering::
getRegClassForInlineAsmConstraint(const std::string &Constraint,
                                  MVT::ValueType VT) const {
  if (Constraint.size() == 1) {
    switch (Constraint[0]) {
    default: break;  // Unknown constriant letter
    case 'f': 
      return make_vector<unsigned>(Alpha::F0 , Alpha::F1 , Alpha::F2 ,
				   Alpha::F3 , Alpha::F4 , Alpha::F5 , 
				   Alpha::F6 , Alpha::F7 , Alpha::F8 , 
				   Alpha::F9 , Alpha::F10, Alpha::F11, 
                                   Alpha::F12, Alpha::F13, Alpha::F14, 
				   Alpha::F15, Alpha::F16, Alpha::F17, 
				   Alpha::F18, Alpha::F19, Alpha::F20, 
				   Alpha::F21, Alpha::F22, Alpha::F23, 
                                   Alpha::F24, Alpha::F25, Alpha::F26, 
				   Alpha::F27, Alpha::F28, Alpha::F29, 
				   Alpha::F30, Alpha::F31, 0);
    case 'r': 
      return make_vector<unsigned>(Alpha::R0 , Alpha::R1 , Alpha::R2 , 
				   Alpha::R3 , Alpha::R4 , Alpha::R5 , 
				   Alpha::R6 , Alpha::R7 , Alpha::R8 , 
				   Alpha::R9 , Alpha::R10, Alpha::R11, 
                                   Alpha::R12, Alpha::R13, Alpha::R14, 
				   Alpha::R15, Alpha::R16, Alpha::R17, 
				   Alpha::R18, Alpha::R19, Alpha::R20, 
				   Alpha::R21, Alpha::R22, Alpha::R23, 
                                   Alpha::R24, Alpha::R25, Alpha::R26, 
				   Alpha::R27, Alpha::R28, Alpha::R29, 
				   Alpha::R30, Alpha::R31, 0);
 
    }
  }
  
  return std::vector<unsigned>();
}
