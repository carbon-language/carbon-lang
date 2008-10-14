//===-- PIC16ISelLowering.cpp - PIC16 DAG Lowering Implementation ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the interfaces that PIC16 uses to lower LLVM code into a
// selection DAG.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pic16-lower"

#include "PIC16ISelLowering.h"
#include "PIC16TargetMachine.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Function.h"
#include "llvm/Intrinsics.h"
#include "llvm/CallingConv.h"
#include "llvm/CodeGen/CallingConvLower.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/Support/Debug.h"
#include <queue>
#include <set>

using namespace llvm;

const char *PIC16TargetLowering:: getTargetNodeName(unsigned Opcode) const 
{
  switch (Opcode) {
    case PIC16ISD::Hi        : return "PIC16ISD::Hi";
    case PIC16ISD::Lo        : return "PIC16ISD::Lo";
    case PIC16ISD::Package   : return "PIC16ISD::Package";
    case PIC16ISD::Wrapper   : return "PIC16ISD::Wrapper";
    case PIC16ISD::SetBank   : return "PIC16ISD::SetBank";
    case PIC16ISD::SetPage   : return "PIC16ISD::SetPage";
    case PIC16ISD::Branch    : return "PIC16ISD::Branch";
    case PIC16ISD::Cmp       : return "PIC16ISD::Cmp";
    case PIC16ISD::BTFSS     : return "PIC16ISD::BTFSS";
    case PIC16ISD::BTFSC     : return "PIC16ISD::BTFSC";
    case PIC16ISD::XORCC     : return "PIC16ISD::XORCC";
    case PIC16ISD::SUBCC     : return "PIC16ISD::SUBCC";
    default                  : return NULL;
  }
}

PIC16TargetLowering::
PIC16TargetLowering(PIC16TargetMachine &TM): TargetLowering(TM) 
{
  // Set up the register classes.
  addRegisterClass(MVT::i8, PIC16::CPURegsRegisterClass);
  addRegisterClass(MVT::i16, PIC16::PTRRegsRegisterClass);

  // Load extented operations for i1 types must be promoted .
  setLoadExtAction(ISD::EXTLOAD, MVT::i1,  Promote);
  setLoadExtAction(ISD::ZEXTLOAD, MVT::i1,  Promote);
  setLoadExtAction(ISD::SEXTLOAD, MVT::i1,  Promote);

  setOperationAction(ISD::ADD, MVT::i1, Promote);
  setOperationAction(ISD::ADD, MVT::i8, Legal);
  setOperationAction(ISD::ADD, MVT::i16, Custom);
  setOperationAction(ISD::ADD, MVT::i32, Expand);
  setOperationAction(ISD::ADD, MVT::i64, Expand);

  setOperationAction(ISD::SUB, MVT::i1, Promote);
  setOperationAction(ISD::SUB, MVT::i8, Legal);
  setOperationAction(ISD::SUB, MVT::i16, Custom);
  setOperationAction(ISD::SUB, MVT::i32, Expand);
  setOperationAction(ISD::SUB, MVT::i64, Expand);

  setOperationAction(ISD::ADDC, MVT::i1, Promote);
  setOperationAction(ISD::ADDC, MVT::i8, Legal);
  setOperationAction(ISD::ADDC, MVT::i16, Custom);
  setOperationAction(ISD::ADDC, MVT::i32, Expand);
  setOperationAction(ISD::ADDC, MVT::i64, Expand);

  setOperationAction(ISD::ADDE, MVT::i1, Promote);
  setOperationAction(ISD::ADDE, MVT::i8, Legal);
  setOperationAction(ISD::ADDE, MVT::i16, Custom);
  setOperationAction(ISD::ADDE, MVT::i32, Expand);
  setOperationAction(ISD::ADDE, MVT::i64, Expand);

  setOperationAction(ISD::SUBC, MVT::i1, Promote);
  setOperationAction(ISD::SUBC, MVT::i8, Legal);
  setOperationAction(ISD::SUBC, MVT::i16, Custom);
  setOperationAction(ISD::SUBC, MVT::i32, Expand);
  setOperationAction(ISD::SUBC, MVT::i64, Expand);

  setOperationAction(ISD::SUBE, MVT::i1, Promote);
  setOperationAction(ISD::SUBE, MVT::i8, Legal);
  setOperationAction(ISD::SUBE, MVT::i16, Custom);
  setOperationAction(ISD::SUBE, MVT::i32, Expand);
  setOperationAction(ISD::SUBE, MVT::i64, Expand);

  // PIC16 does not have these NodeTypes below.
  setOperationAction(ISD::SETCC, MVT::i1, Expand);
  setOperationAction(ISD::SETCC, MVT::i8, Expand);
  setOperationAction(ISD::SETCC, MVT::Other, Expand);
  setOperationAction(ISD::SELECT_CC, MVT::i1, Custom);
  setOperationAction(ISD::SELECT_CC, MVT::i8, Custom);

  setOperationAction(ISD::BRCOND, MVT::i1, Expand);
  setOperationAction(ISD::BRCOND, MVT::i8, Expand);
  setOperationAction(ISD::BRCOND, MVT::Other, Expand);

  setOperationAction(ISD::BR_CC, MVT::i1, Custom);
  setOperationAction(ISD::BR_CC, MVT::i8, Custom);

  setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);

  
  // FIXME: Do we really need to Custom lower the GA ??
  setOperationAction(ISD::GlobalAddress, MVT::i8, Custom);
  setOperationAction(ISD::RET, MVT::Other, Custom);

  setOperationAction(ISD::CTPOP, MVT::i32, Expand);
  setOperationAction(ISD::CTTZ, MVT::i32, Expand);
  setOperationAction(ISD::CTLZ, MVT::i32, Expand);
  setOperationAction(ISD::ROTL, MVT::i32, Expand);
  setOperationAction(ISD::ROTR, MVT::i32, Expand);
  setOperationAction(ISD::BSWAP, MVT::i32, Expand);

  setOperationAction(ISD::SHL_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRA_PARTS, MVT::i32, Expand);
  setOperationAction(ISD::SRL_PARTS, MVT::i32, Expand);

  // We don't have line number support yet.
  setOperationAction(ISD::DBG_STOPPOINT, MVT::Other, Expand);
  setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
  setOperationAction(ISD::DBG_LABEL, MVT::Other, Expand);
  setOperationAction(ISD::EH_LABEL, MVT::Other, Expand);

  // Use the default for now.
  setOperationAction(ISD::STACKSAVE, MVT::Other, Expand);
  setOperationAction(ISD::STACKRESTORE, MVT::Other, Expand);

  setOperationAction(ISD::LOAD, MVT::i1, Promote);
  setOperationAction(ISD::LOAD, MVT::i8, Legal);

  setTargetDAGCombine(ISD::LOAD);
  setTargetDAGCombine(ISD::STORE);
  setTargetDAGCombine(ISD::ADDE);
  setTargetDAGCombine(ISD::ADDC);
  setTargetDAGCombine(ISD::ADD);
  setTargetDAGCombine(ISD::SUBE);
  setTargetDAGCombine(ISD::SUBC);
  setTargetDAGCombine(ISD::SUB);

  setStackPointerRegisterToSaveRestore(PIC16::STKPTR);
  computeRegisterProperties();
}


SDValue PIC16TargetLowering:: LowerOperation(SDValue Op, SelectionDAG &DAG) 
{
  SDVTList VTList16 = DAG.getVTList(MVT::i16, MVT::i16, MVT::Other);
  switch (Op.getOpcode()) {
    case ISD::STORE: 
      DOUT << "reduce store\n"; 
      break;

    case ISD::FORMAL_ARGUMENTS:   
      DOUT << "==== lowering formal args\n";
      return LowerFORMAL_ARGUMENTS(Op, DAG);

    case ISD::GlobalAddress:      
      DOUT << "==== lowering GA\n";
      return LowerGlobalAddress(Op, DAG);

    case ISD::RET:                
      DOUT << "==== lowering ret\n";
      return LowerRET(Op, DAG);

    case ISD::FrameIndex:                
      DOUT << "==== lowering frame index\n";
      return LowerFrameIndex(Op, DAG);

    case ISD::ADDE: 
      DOUT << "==== lowering adde\n"; 
      break;

    case ISD::LOAD:
    case ISD::ADD: 
      break;

    case ISD::BR_CC:                
      DOUT << "==== lowering BR_CC\n"; 
      return LowerBR_CC(Op, DAG); 
  } // end switch.
  return SDValue();
}


//===----------------------------------------------------------------------===//
//  Lower helper functions
//===----------------------------------------------------------------------===//

SDValue PIC16TargetLowering::LowerBR_CC(SDValue Op, SelectionDAG &DAG) 
{
  MVT VT = Op.getValueType();
  SDValue Chain = Op.getOperand(0);
  ISD::CondCode CC = cast<CondCodeSDNode>(Op.getOperand(1))->get();
  SDValue LHS = Op.getOperand(2);
  SDValue RHS = Op.getOperand(3);
  SDValue JumpVal = Op.getOperand(4);
  SDValue Result;
  unsigned  cmpOpcode;
  unsigned  branchOpcode;
  SDValue branchOperand;

  SDValue StatusReg = DAG.getRegister(PIC16::STATUSREG, MVT::i8);
  SDValue CPUReg = DAG.getRegister(PIC16::WREG, MVT::i8);
  switch(CC) {
    default:
      assert(0 && "This condition code is not handled yet!!");
      abort();

    case ISD::SETNE:
      DOUT << "setne\n";
      cmpOpcode = PIC16ISD::XORCC;
      branchOpcode = PIC16ISD::BTFSS;
      branchOperand = DAG.getConstant(2, MVT::i8);
      break;

    case ISD::SETEQ:
      DOUT << "seteq\n";
      cmpOpcode = PIC16ISD::XORCC;
      branchOpcode = PIC16ISD::BTFSC;
      branchOperand = DAG.getConstant(2, MVT::i8);
      break;

    case ISD::SETGT:
      assert(0 && "Greater Than condition code is not handled yet!!");
      abort();
      break;

    case ISD::SETGE:
      DOUT << "setge\n";
      cmpOpcode = PIC16ISD::SUBCC;
      branchOpcode = PIC16ISD::BTFSS;
      branchOperand = DAG.getConstant(1, MVT::i8);
      break;

    case ISD::SETLT:
      DOUT << "setlt\n";
      cmpOpcode = PIC16ISD::SUBCC;
      branchOpcode = PIC16ISD::BTFSC;
      branchOperand = DAG.getConstant(1,MVT::i8);
      break;

    case ISD::SETLE:
      assert(0 && "Less Than Equal condition code is not handled yet!!");
      abort();
      break;
  }  // End of Switch

   SDVTList VTList = DAG.getVTList(MVT::i8, MVT::Flag);
   SDValue CmpValue = DAG.getNode(cmpOpcode, VTList, LHS, RHS).getValue(1);
   Result = DAG.getNode(branchOpcode, VT, Chain, JumpVal, branchOperand, 
                        StatusReg, CmpValue);
   return Result;
}


//===----------------------------------------------------------------------===//
//  Misc Lower Operation implementation
//===----------------------------------------------------------------------===//

// LowerGlobalAddress - Create a constant pool entry for global value 
// and wrap it in a wrapper node.
SDValue
PIC16TargetLowering::LowerGlobalAddress(SDValue Op, SelectionDAG &DAG) 
{
  MVT PtrVT = getPointerTy();
  GlobalAddressSDNode *GSDN = cast<GlobalAddressSDNode>(Op);
  GlobalValue *GV = GSDN->getGlobal();

  // FIXME: for now only do the ram.
  SDValue CPAddr = DAG.getTargetConstantPool(GV, PtrVT, 2);
  SDValue CPBank = DAG.getNode(PIC16ISD::SetBank, MVT::i8, CPAddr);
  CPAddr = DAG.getNode(PIC16ISD::Wrapper, MVT::i8, CPAddr,CPBank);

  return CPAddr;
}

SDValue
PIC16TargetLowering::LowerRET(SDValue Op, SelectionDAG &DAG) 
{
  switch(Op.getNumOperands()) {
    default:
      assert(0 && "Do not know how to return this many arguments!");
      abort();

    case 1:
      return SDValue(); // ret void is legal
  }
}

SDValue
PIC16TargetLowering::LowerFrameIndex(SDValue N, SelectionDAG &DAG) 
{
  if (FrameIndexSDNode *FIN = dyn_cast<FrameIndexSDNode>(N)) {
    return DAG.getTargetFrameIndex(FIN->getIndex(), MVT::i32);
  }

  return N;
}

SDValue
PIC16TargetLowering::LowerLOAD(SDNode *N,
                               SelectionDAG &DAG,
                               DAGCombinerInfo &DCI) const
{
  SDValue Outs[2];
  SDValue TF; //TokenFactor
  SDValue OutChains[2];
  SDValue Chain = N->getOperand(0);  
  SDValue Src   = N->getOperand(1);
  SDValue retVal;
  SDVTList VTList;

  // If this load is directly stored, replace the load value with the stored
  // value.
  // FIXME: Handle store large -> read small portion.
  // FIXME: Handle TRUNCSTORE/LOADEXT
  LoadSDNode *LD  = cast<LoadSDNode>(N);
  SDValue Ptr   = LD->getBasePtr();
  if (LD->getExtensionType() == ISD::NON_EXTLOAD) {
    if (ISD::isNON_TRUNCStore(Chain.getNode())) {
      StoreSDNode *PrevST = cast<StoreSDNode>(Chain);
      if (PrevST->getBasePtr() == Ptr &&
          PrevST->getValue().getValueType() == N->getValueType(0))
        return DCI.CombineTo(N, Chain.getOperand(1), Chain);
    }
  }

  if (N->getValueType(0) != MVT::i16)
    return SDValue();

  SDValue toWorklist;
  Outs[0] = DAG.getLoad(MVT::i8, Chain, Src, NULL, 0);
  toWorklist = DAG.getNode(ISD::ADD, MVT::i16, Src,
                           DAG.getConstant(1, MVT::i16));
  Outs[1] = DAG.getLoad(MVT::i8, Chain, toWorklist, NULL, 0);
  // FIXME: Add to worklist may not be needed. 
  // It is meant to merge sequences of add with constant into one. 
  DCI.AddToWorklist(toWorklist.getNode());   
  
  // Create the tokenfactors and carry it on to the build_pair node
  OutChains[0] = Outs[0].getValue(1);
  OutChains[1] = Outs[1].getValue(1);
  TF = DAG.getNode(ISD::TokenFactor, MVT::Other, &OutChains[0], 2);
  
  VTList = DAG.getVTList(MVT::i16, MVT::Flag);
  retVal = DAG.getNode (PIC16ISD::Package, VTList, &Outs[0], 2);

  DCI.CombineTo (N, retVal, TF);

  return retVal;
}

SDValue
PIC16TargetLowering::LowerADDSUB(SDNode *N, SelectionDAG &DAG,
                                 DAGCombinerInfo &DCI) const
{
  bool changed = false;
  int i;
  SDValue LoOps[3], HiOps[3];
  SDValue OutOps[3]; // [0]:left, [1]:right, [2]:carry
  SDValue InOp[2];
  SDValue retVal;
  SDValue as1,as2;
  SDVTList VTList;
  unsigned AS = 0, ASE = 0, ASC=0;

  InOp[0] = N->getOperand(0);
  InOp[1] = N->getOperand(1);  

  switch (N->getOpcode()) {
    case ISD::ADD:
      if (InOp[0].getOpcode() == ISD::Constant &&
          InOp[1].getOpcode() == ISD::Constant) {
        ConstantSDNode *CST0 = dyn_cast<ConstantSDNode>(InOp[0]);
        ConstantSDNode *CST1 = dyn_cast<ConstantSDNode>(InOp[1]);
        return DAG.getConstant(CST0->getZExtValue() + CST1->getZExtValue(),
                               MVT::i16);
      }
      break;

    case ISD::ADDE:
    case ISD::ADDC:
      AS  = ISD::ADD;
      ASE = ISD::ADDE;
      ASC = ISD::ADDC;
      break;

    case ISD::SUB:
      if (InOp[0].getOpcode() == ISD::Constant &&
          InOp[1].getOpcode() == ISD::Constant) {
        ConstantSDNode *CST0 = dyn_cast<ConstantSDNode>(InOp[0]);
        ConstantSDNode *CST1 = dyn_cast<ConstantSDNode>(InOp[1]);
        return DAG.getConstant(CST0->getZExtValue() - CST1->getZExtValue(),
                               MVT::i16);
      }
      break;

    case ISD::SUBE:
    case ISD::SUBC:
      AS  = ISD::SUB;
      ASE = ISD::SUBE;
      ASC = ISD::SUBC;
      break;
  } // end switch.

  assert ((N->getValueType(0) == MVT::i16) 
           && "expecting an MVT::i16 node for lowering");
  assert ((N->getOperand(0).getValueType() == MVT::i16) 
           && (N->getOperand(1).getValueType() == MVT::i16) 
            && "both inputs to addx/subx:i16 must be i16");

  for (i = 0; i < 2; i++) {
    if (InOp[i].getOpcode() == ISD::GlobalAddress) {
      // We don't want to lower subs/adds with global address yet.
      return SDValue();
    }
    else if (InOp[i].getOpcode() == ISD::Constant) {
      changed = true;
      ConstantSDNode *CST = dyn_cast<ConstantSDNode>(InOp[i]);
      LoOps[i] = DAG.getConstant(CST->getZExtValue() & 0xFF, MVT::i8);
      HiOps[i] = DAG.getConstant(CST->getZExtValue() >> 8, MVT::i8);
    }
    else if (InOp[i].getOpcode() == PIC16ISD::Package) {
      LoOps[i] = InOp[i].getOperand(0);
      HiOps[i] = InOp[i].getOperand(1);
    }
    else if (InOp[i].getOpcode() == ISD::LOAD) {
      changed = true;
      // LowerLOAD returns a Package node or it may combine and return 
      // anything else.
      SDValue lowered = LowerLOAD(InOp[i].getNode(), DAG, DCI);

      // So If LowerLOAD returns something other than Package, 
      // then just call ADD again.
      if (lowered.getOpcode() != PIC16ISD::Package)
        return LowerADDSUB(N, DAG, DCI);
          
      LoOps[i] = lowered.getOperand(0);
      HiOps[i] = lowered.getOperand(1);
    }
    else if ((InOp[i].getOpcode() == ISD::ADD) || 
             (InOp[i].getOpcode() == ISD::ADDE) ||
             (InOp[i].getOpcode() == ISD::ADDC) ||
             (InOp[i].getOpcode() == ISD::SUB) ||
             (InOp[i].getOpcode() == ISD::SUBE) ||
             (InOp[i].getOpcode() == ISD::SUBC)) {
      changed = true;
      // Must call LowerADDSUB recursively here,
      // LowerADDSUB returns a Package node.
      SDValue lowered = LowerADDSUB(InOp[i].getNode(), DAG, DCI);

      LoOps[i] = lowered.getOperand(0);
      HiOps[i] = lowered.getOperand(1);
    }
    else if (InOp[i].getOpcode() == ISD::SIGN_EXTEND) {
      // FIXME: I am just zero extending. for now.
      changed = true;
      LoOps[i] = InOp[i].getOperand(0);
      HiOps[i] = DAG.getConstant(0, MVT::i8);
    }
    else {
      DAG.setGraphColor(N, "blue");
      DAG.viewGraph();
      assert (0 && "not implemented yet");
    }
  } // end for.

  assert (changed && "nothing changed while lowering SUBx/ADDx");

  VTList = DAG.getVTList(MVT::i8, MVT::Flag);
  if (N->getOpcode() == ASE) { 
    // We must take in the existing carry
    // if this node is part of an existing subx/addx sequence.
    LoOps[2] = N->getOperand(2).getValue(1);
    as1 = DAG.getNode (ASE, VTList, LoOps, 3);
  }
  else {
    as1 = DAG.getNode (ASC, VTList, LoOps, 2);
  }
  HiOps[2] = as1.getValue(1);
  as2 = DAG.getNode (ASE, VTList, HiOps, 3);
  // We must build a pair that also provides the carry from sube/adde.
  OutOps[0] = as1;
  OutOps[1] = as2;
  OutOps[2] = as2.getValue(1);
  // Breaking an original i16, so lets make the Package also an i16.
  if (N->getOpcode() == ASE) {
    VTList = DAG.getVTList(MVT::i16, MVT::Flag);
    retVal = DAG.getNode (PIC16ISD::Package, VTList, OutOps, 3);
    DCI.CombineTo (N, retVal, OutOps[2]);
  }
  else if (N->getOpcode() == ASC) {
    VTList = DAG.getVTList(MVT::i16, MVT::Flag);
    retVal = DAG.getNode (PIC16ISD::Package, VTList, OutOps, 2);
    DCI.CombineTo (N, retVal, OutOps[2]);
  }
  else if (N->getOpcode() == AS) {
    VTList = DAG.getVTList(MVT::i16);
    retVal = DAG.getNode (PIC16ISD::Package, VTList, OutOps, 2);
    DCI.CombineTo (N, retVal);
  }

  return retVal;
}


//===----------------------------------------------------------------------===//
//                      Calling Convention Implementation
//===----------------------------------------------------------------------===//

#include "PIC16GenCallingConv.inc"

//===----------------------------------------------------------------------===//
//                  CALL Calling Convention Implementation
//===----------------------------------------------------------------------===//


//===----------------------------------------------------------------------===//
//             FORMAL_ARGUMENTS Calling Convention Implementation
//===----------------------------------------------------------------------===//
SDValue PIC16TargetLowering::
LowerFORMAL_ARGUMENTS(SDValue Op, SelectionDAG &DAG)
{
  SmallVector<SDValue, 8> ArgValues;
  SDValue Root = Op.getOperand(0);

  // Return the new list of results.
  // FIXME: Just copy right now.
  ArgValues.push_back(Root);

  return DAG.getMergeValues(Op.getNode()->getVTList(), &ArgValues[0],
                            ArgValues.size()).getValue(Op.getResNo());
}


//===----------------------------------------------------------------------===//
//               Return Value Calling Convention Implementation
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
//                           PIC16 Inline Assembly Support
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Target Optimization Hooks
//===----------------------------------------------------------------------===//

SDValue PIC16TargetLowering::PerformDAGCombine(SDNode *N, 
                                                 DAGCombinerInfo &DCI) const 
{
  int i;
  ConstantSDNode *CST;
  SelectionDAG &DAG = DCI.DAG;

  switch (N->getOpcode()) {
    default: 
      break;

    case PIC16ISD::Package:
      DOUT << "==== combining PIC16ISD::Package\n";
      return SDValue();

    case ISD::ADD:
    case ISD::SUB:
      if ((N->getOperand(0).getOpcode() == ISD::GlobalAddress) ||
          (N->getOperand(0).getOpcode() == ISD::FrameIndex)) {
        // Do not touch pointer adds.
        return SDValue ();
      }
      break;

    case ISD::ADDE :
    case ISD::ADDC :
    case ISD::SUBE :
    case ISD::SUBC :
      if (N->getValueType(0) == MVT::i16) {
        SDValue retVal = LowerADDSUB(N, DAG,DCI); 
        // LowerADDSUB has already combined the result, 
        // so we just return nothing to avoid assertion failure from llvm 
        // if N has been deleted already.
        return SDValue();
      }
      else if (N->getValueType(0) == MVT::i8) { 
        // Sanity check ....
        for (int i=0; i<2; i++) {
          if (N->getOperand (i).getOpcode() == PIC16ISD::Package) {
            assert (0 && 
                    "don't want to have PIC16ISD::Package as intput to add:i8");
          }
        }
      }
      break;

    // FIXME: split this large chunk of code.
    case ISD::STORE :
    {
      SDValue Chain = N->getOperand(0);  
      SDValue Src = N->getOperand(1);
      SDValue Dest = N->getOperand(2);
      unsigned int DstOff = 0;
      int NUM_STORES = 0;
      SDValue Stores[6];

      // if source operand is expected to be extended to 
      // some higher type then - remove this extension 
      // SDNode and do the extension manually
      if ((Src.getOpcode() == ISD::ANY_EXTEND) ||
          (Src.getOpcode() == ISD::SIGN_EXTEND) || 
          (Src.getOpcode() == ISD::ZERO_EXTEND)) {
        Src = Src.getNode()->getOperand(0);
        Stores[0] = DAG.getStore(Chain, Src, Dest, NULL,0);
        return Stores[0];
      }

      switch(Src.getValueType().getSimpleVT()) {
        default:
          assert(false && "Invalid value type!");

        case MVT::i8:  
          break;

        case MVT::i16: 
          NUM_STORES = 2;
          break;

        case MVT::i32: 
          NUM_STORES = 4;
          break;

        case MVT::i64: 
          NUM_STORES = 8; 
          break;
      }

      if (isa<GlobalAddressSDNode>(Dest) && isa<LoadSDNode>(Src) && 
          (Src.getValueType() != MVT::i8)) {
        //create direct addressing a = b
        Chain = Src.getOperand(0);
        for (i=0; i<NUM_STORES; i++) {
          SDValue ADN = DAG.getNode(ISD::ADD, MVT::i16, Src.getOperand(1),
                                      DAG.getConstant(DstOff, MVT::i16));
          SDValue LDN = DAG.getLoad(MVT::i8, Chain, ADN, NULL, 0);
          SDValue DSTADDR = DAG.getNode(ISD::ADD, MVT::i16, Dest,
                                          DAG.getConstant(DstOff, MVT::i16));
          Stores[i] = DAG.getStore(Chain, LDN, DSTADDR, NULL, 0);
          Chain = Stores[i];
          DstOff += 1;
        } 
        
        Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, &Stores[0], i);
        return Chain;
      }
      else if (isa<GlobalAddressSDNode>(Dest) && isa<ConstantSDNode>(Src) 
               && (Src.getValueType() != MVT::i8)) {
        //create direct addressing a = CONST
        CST = dyn_cast<ConstantSDNode>(Src);
        for (i = 0; i < NUM_STORES; i++) {
          SDValue CNST = DAG.getConstant(CST->getZExtValue() >> i*8, MVT::i8);
          SDValue ADN = DAG.getNode(ISD::ADD, MVT::i16, Dest,
                                      DAG.getConstant(DstOff, MVT::i16));
          Stores[i] = DAG.getStore(Chain, CNST, ADN, NULL, 0);
          Chain = Stores[i];
          DstOff += 1;
        } 
          
        Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, &Stores[0], i);
        return Chain;
      }
      else if (isa<LoadSDNode>(Dest) && isa<ConstantSDNode>(Src) 
              && (Src.getValueType() != MVT::i8)) {
        // Create indirect addressing.
        CST = dyn_cast<ConstantSDNode>(Src);
        Chain = Dest.getOperand(0);  
        SDValue Load;
        Load = DAG.getLoad(MVT::i16, Chain,Dest.getOperand(1), NULL, 0);
        Chain = Load.getValue(1);
        for (i=0; i<NUM_STORES; i++) {
          SDValue CNST = DAG.getConstant(CST->getZExtValue() >> i*8, MVT::i8);
          Stores[i] = DAG.getStore(Chain, CNST, Load, NULL, 0);
          Chain = Stores[i];
          DstOff += 1;
        } 
          
        Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, &Stores[0], i);
        return Chain;
      }
      else if (isa<LoadSDNode>(Dest) && isa<GlobalAddressSDNode>(Src)) {
        // GlobalAddressSDNode *GAD = dyn_cast<GlobalAddressSDNode>(Src);
        return SDValue();
      }
      else if (Src.getOpcode() == PIC16ISD::Package) {
        StoreSDNode *st = dyn_cast<StoreSDNode>(N);
        SDValue toWorkList, retVal;
        Chain = N->getOperand(0);

        if (st->isTruncatingStore()) {
          retVal = DAG.getStore(Chain, Src.getOperand(0), Dest, NULL, 0);
        }
        else {
          toWorkList = DAG.getNode(ISD::ADD, MVT::i16, Dest,
                                   DAG.getConstant(1, MVT::i16));
          Stores[1] = DAG.getStore(Chain, Src.getOperand(0), Dest, NULL, 0);
          Stores[0] = DAG.getStore(Chain, Src.getOperand(1), toWorkList, NULL, 
                                   0);

          // We want to merge sequence of add with constant to one add and a 
          // constant, so add the ADD node to worklist to have llvm do that 
          // automatically.
          DCI.AddToWorklist(toWorkList.getNode()); 

          // We don't need the Package so add to worklist so llvm deletes it
          DCI.AddToWorklist(Src.getNode());
          retVal = DAG.getNode(ISD::TokenFactor, MVT::Other, &Stores[0], 2);
        }

        return retVal;
      }
      else if (Src.getOpcode() == ISD::TRUNCATE) {
      }
      else {
      }
    } // end ISD::STORE.
    break;

    case ISD::LOAD :
    {
      SDValue Ptr = N->getOperand(1);
      if (Ptr.getOpcode() == PIC16ISD::Package) {
        assert (0 && "not implemented yet");
       }
    }
    break;
  } // end switch.

  return SDValue();
}

//===----------------------------------------------------------------------===//
//               Utility functions
//===----------------------------------------------------------------------===//
const SDValue *PIC16TargetLowering::
findLoadi8(const SDValue &Src, SelectionDAG &DAG) const
{
  unsigned int i;
  if ((Src.getOpcode() == ISD::LOAD) && (Src.getValueType() == MVT::i8))
    return &Src;
  for (i=0; i<Src.getNumOperands(); i++) {
    const SDValue *retVal = findLoadi8(Src.getOperand(i),DAG);
    if (retVal) return retVal;
  }

  return NULL;
}
