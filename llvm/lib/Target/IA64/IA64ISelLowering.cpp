//===-- IA64ISelLowering.cpp - IA64 DAG Lowering Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IA64ISelLowering class.
//
//===----------------------------------------------------------------------===//

#include "IA64ISelLowering.h"
#include "IA64MachineFunctionInfo.h"
#include "IA64TargetMachine.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Constants.h"
#include "llvm/Function.h"
using namespace llvm;

IA64TargetLowering::IA64TargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
 
      // register class for general registers
      addRegisterClass(MVT::i64, IA64::GRRegisterClass);

      // register class for FP registers
      addRegisterClass(MVT::f64, IA64::FPRegisterClass);

      // register class for predicate registers
      addRegisterClass(MVT::i1, IA64::PRRegisterClass);

      setOperationAction(ISD::BRCONDTWOWAY     , MVT::Other, Expand);
      setOperationAction(ISD::BRTWOWAY_CC      , MVT::Other, Expand);
      setOperationAction(ISD::FP_ROUND_INREG   , MVT::f32  , Expand);

      setSetCCResultType(MVT::i1);
      setShiftAmountType(MVT::i64);

      setOperationAction(ISD::EXTLOAD          , MVT::i1   , Promote);

      setOperationAction(ISD::ZEXTLOAD         , MVT::i1   , Expand);

      setOperationAction(ISD::SEXTLOAD         , MVT::i1   , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i8   , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i16  , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i32  , Expand);

      setOperationAction(ISD::FREM             , MVT::f32  , Expand);
      setOperationAction(ISD::FREM             , MVT::f64  , Expand);

      setOperationAction(ISD::UREM             , MVT::f32  , Expand);
      setOperationAction(ISD::UREM             , MVT::f64  , Expand);

      setOperationAction(ISD::MEMMOVE          , MVT::Other, Expand);
      setOperationAction(ISD::MEMSET           , MVT::Other, Expand);
      setOperationAction(ISD::MEMCPY           , MVT::Other, Expand);
      
      setOperationAction(ISD::SINT_TO_FP       , MVT::i1   , Promote);
      setOperationAction(ISD::UINT_TO_FP       , MVT::i1   , Promote);

      // We don't support sin/cos/sqrt
      setOperationAction(ISD::FSIN , MVT::f64, Expand);
      setOperationAction(ISD::FCOS , MVT::f64, Expand);
      setOperationAction(ISD::FSQRT, MVT::f64, Expand);
      setOperationAction(ISD::FSIN , MVT::f32, Expand);
      setOperationAction(ISD::FCOS , MVT::f32, Expand);
      setOperationAction(ISD::FSQRT, MVT::f32, Expand);

      // We don't have line number support yet.
      setOperationAction(ISD::LOCATION, MVT::Other, Expand);
      setOperationAction(ISD::DEBUG_LOC, MVT::Other, Expand);
      setOperationAction(ISD::DEBUG_LABEL, MVT::Other, Expand);

      //IA64 has these, but they are not implemented
      setOperationAction(ISD::CTTZ , MVT::i64  , Expand);
      setOperationAction(ISD::CTLZ , MVT::i64  , Expand);

      computeRegisterProperties();

      addLegalFPImmediate(+0.0);
      addLegalFPImmediate(+1.0);
}

/// isFloatingPointZero - Return true if this is 0.0 or -0.0.
static bool isFloatingPointZero(SDOperand Op) {
  if (ConstantFPSDNode *CFP = dyn_cast<ConstantFPSDNode>(Op))
    return CFP->isExactlyValue(-0.0) || CFP->isExactlyValue(0.0);
  else if (Op.getOpcode() == ISD::EXTLOAD || Op.getOpcode() == ISD::LOAD) {
    // Maybe this has already been legalized into the constant pool?
    if (ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Op.getOperand(1)))
      if (ConstantFP *CFP = dyn_cast<ConstantFP>(CP->get()))
        return CFP->isExactlyValue(-0.0) || CFP->isExactlyValue(0.0);
  }
  return false;
}

std::vector<SDOperand>
IA64TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  std::vector<SDOperand> ArgValues;
  //
  // add beautiful description of IA64 stack frame format
  // here (from intel 24535803.pdf most likely)
  //
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();
  
  GP = MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
  SP = MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
  RP = MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
  
  MachineBasicBlock& BB = MF.front();

  unsigned args_int[] = {IA64::r32, IA64::r33, IA64::r34, IA64::r35,
                         IA64::r36, IA64::r37, IA64::r38, IA64::r39};

  unsigned args_FP[] = {IA64::F8, IA64::F9, IA64::F10, IA64::F11,
                        IA64::F12,IA64::F13,IA64::F14, IA64::F15};

  unsigned argVreg[8];
  unsigned argPreg[8];
  unsigned argOpc[8];

  unsigned used_FPArgs = 0; // how many FP args have been used so far?

  unsigned ArgOffset = 0;
  int count = 0;

  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
    {
      SDOperand newroot, argt;
      if(count < 8) { // need to fix this logic? maybe.

        switch (getValueType(I->getType())) {
          default:
            assert(0 && "ERROR in LowerArgs: can't lower this type of arg.\n"); 
          case MVT::f32:
            // fixme? (well, will need to for weird FP structy stuff,
            // see intel ABI docs)
          case MVT::f64:
//XXX            BuildMI(&BB, IA64::IDEF, 0, args_FP[used_FPArgs]);
            MF.addLiveIn(args_FP[used_FPArgs]); // mark this reg as liveIn
            // floating point args go into f8..f15 as-needed, the increment
            argVreg[count] =                              // is below..:
            MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::f64));
            // FP args go into f8..f15 as needed: (hence the ++)
            argPreg[count] = args_FP[used_FPArgs++];
            argOpc[count] = IA64::FMOV;
            argt = newroot = DAG.getCopyFromReg(DAG.getRoot(), argVreg[count],
                                                MVT::f64);
            if (I->getType() == Type::FloatTy)
              argt = DAG.getNode(ISD::FP_ROUND, MVT::f32, argt);
            break;
          case MVT::i1: // NOTE: as far as C abi stuff goes,
                        // bools are just boring old ints
          case MVT::i8:
          case MVT::i16:
          case MVT::i32:
          case MVT::i64:
//XXX            BuildMI(&BB, IA64::IDEF, 0, args_int[count]);
            MF.addLiveIn(args_int[count]); // mark this register as liveIn
            argVreg[count] =
            MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
            argPreg[count] = args_int[count];
            argOpc[count] = IA64::MOV;
            argt = newroot =
              DAG.getCopyFromReg(DAG.getRoot(), argVreg[count], MVT::i64);
            if ( getValueType(I->getType()) != MVT::i64)
              argt = DAG.getNode(ISD::TRUNCATE, getValueType(I->getType()),
                  newroot);
            break;
        }
      } else { // more than 8 args go into the frame
        // Create the frame index object for this incoming parameter...
        ArgOffset = 16 + 8 * (count - 8);
        int FI = MFI->CreateFixedObject(8, ArgOffset);

        // Create the SelectionDAG nodes corresponding to a load
        //from this parameter
        SDOperand FIN = DAG.getFrameIndex(FI, MVT::i64);
        argt = newroot = DAG.getLoad(getValueType(I->getType()),
                                     DAG.getEntryNode(), FIN, DAG.getSrcValue(NULL));
      }
      ++count;
      DAG.setRoot(newroot.getValue(1));
      ArgValues.push_back(argt);
    }


  // Create a vreg to hold the output of (what will become)
  // the "alloc" instruction
  VirtGPR = MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
  BuildMI(&BB, IA64::PSEUDO_ALLOC, 0, VirtGPR);
  // we create a PSEUDO_ALLOC (pseudo)instruction for now
/*
  BuildMI(&BB, IA64::IDEF, 0, IA64::r1);

  // hmm:
  BuildMI(&BB, IA64::IDEF, 0, IA64::r12);
  BuildMI(&BB, IA64::IDEF, 0, IA64::rp);
  // ..hmm.
  
  BuildMI(&BB, IA64::MOV, 1, GP).addReg(IA64::r1);

  // hmm:
  BuildMI(&BB, IA64::MOV, 1, SP).addReg(IA64::r12);
  BuildMI(&BB, IA64::MOV, 1, RP).addReg(IA64::rp);
  // ..hmm.
*/

  unsigned tempOffset=0;

  // if this is a varargs function, we simply lower llvm.va_start by
  // pointing to the first entry
  if(F.isVarArg()) {
    tempOffset=0;
    VarArgsFrameIndex = MFI->CreateFixedObject(8, tempOffset);
  }

  // here we actually do the moving of args, and store them to the stack
  // too if this is a varargs function:
  for (int i = 0; i < count && i < 8; ++i) {
    BuildMI(&BB, argOpc[i], 1, argVreg[i]).addReg(argPreg[i]);
    if(F.isVarArg()) {
      // if this is a varargs function, we copy the input registers to the stack
      int FI = MFI->CreateFixedObject(8, tempOffset);
      tempOffset+=8;   //XXX: is it safe to use r22 like this?
      BuildMI(&BB, IA64::MOV, 1, IA64::r22).addFrameIndex(FI);
      // FIXME: we should use st8.spill here, one day
      BuildMI(&BB, IA64::ST8, 1, IA64::r22).addReg(argPreg[i]);
    }
  }

  // Finally, inform the code generator which regs we return values in.
  // (see the ISD::RET: case in the instruction selector)
  switch (getValueType(F.getReturnType())) {
  default: assert(0 && "i have no idea where to return this type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
  case MVT::i64:
    MF.addLiveOut(IA64::r8);
    break;
  case MVT::f32:
  case MVT::f64:
    MF.addLiveOut(IA64::F8);
    break;
  }

  return ArgValues;
}

std::pair<SDOperand, SDOperand>
IA64TargetLowering::LowerCallTo(SDOperand Chain,
                                const Type *RetTy, bool isVarArg,
                                unsigned CallingConv, bool isTailCall,
                                SDOperand Callee, ArgListTy &Args,
                                SelectionDAG &DAG) {

  MachineFunction &MF = DAG.getMachineFunction();

  unsigned NumBytes = 16;
  unsigned outRegsUsed = 0;

  if (Args.size() > 8) {
    NumBytes += (Args.size() - 8) * 8;
    outRegsUsed = 8;
  } else {
    outRegsUsed = Args.size();
  }

  // FIXME? this WILL fail if we ever try to pass around an arg that
  // consumes more than a single output slot (a 'real' double, int128
  // some sort of aggregate etc.), as we'll underestimate how many 'outX'
  // registers we use. Hopefully, the assembler will notice.
  MF.getInfo<IA64FunctionInfo>()->outRegsUsed=
    std::max(outRegsUsed, MF.getInfo<IA64FunctionInfo>()->outRegsUsed);

  // keep stack frame 16-byte aligned
  //assert(NumBytes==((NumBytes+15) & ~15) && "stack frame not 16-byte aligned!");
  NumBytes = (NumBytes+15) & ~15;
  
  Chain = DAG.getNode(ISD::CALLSEQ_START, MVT::Other, Chain,
                        DAG.getConstant(NumBytes, getPointerTy()));

  SDOperand StackPtr, NullSV;
  std::vector<SDOperand> Stores;
  std::vector<SDOperand> Converts;
  std::vector<SDOperand> RegValuesToPass;
  unsigned ArgOffset = 16;
  
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
    {
      SDOperand Val = Args[i].first;
      MVT::ValueType ObjectVT = Val.getValueType();
      SDOperand ValToStore, ValToConvert;
      unsigned ObjSize=8;
      switch (ObjectVT) {
      default: assert(0 && "unexpected argument type!");
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
        //promote to 64-bits, sign/zero extending based on type
        //of the argument
        if(Args[i].second->isSigned())
          Val = DAG.getNode(ISD::SIGN_EXTEND, MVT::i64, Val);
        else
          Val = DAG.getNode(ISD::ZERO_EXTEND, MVT::i64, Val);
        // XXX: fall through
      case MVT::i64:
        //ObjSize = 8;
        if(RegValuesToPass.size() >= 8) {
          ValToStore = Val;
        } else {
          RegValuesToPass.push_back(Val);
        }
        break;
      case MVT::f32:
        //promote to 64-bits
        Val = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Val);
        // XXX: fall through
      case MVT::f64:
        if(RegValuesToPass.size() >= 8) {
          ValToStore = Val;
        } else {
          RegValuesToPass.push_back(Val);
	  if(1 /* TODO: if(calling external or varadic function)*/ ) {
	    ValToConvert = Val; // additionally pass this FP value as an int
	  }
        }
        break;
      }
      
      if(ValToStore.Val) {
        if(!StackPtr.Val) {
          StackPtr = DAG.getRegister(IA64::r12, MVT::i64);
          NullSV = DAG.getSrcValue(NULL);
        }
        SDOperand PtrOff = DAG.getConstant(ArgOffset, getPointerTy());
        PtrOff = DAG.getNode(ISD::ADD, MVT::i64, StackPtr, PtrOff);
        Stores.push_back(DAG.getNode(ISD::STORE, MVT::Other, Chain,
                                     ValToStore, PtrOff, NullSV));
        ArgOffset += ObjSize;
      }

      if(ValToConvert.Val) {
	Converts.push_back(DAG.getNode(IA64ISD::GETFD, MVT::i64, ValToConvert)); 
      }
    }

  // Emit all stores, make sure they occur before any copies into physregs.
  if (!Stores.empty())
    Chain = DAG.getNode(ISD::TokenFactor, MVT::Other, Stores);

  static const unsigned IntArgRegs[] = {
    IA64::out0, IA64::out1, IA64::out2, IA64::out3, 
    IA64::out4, IA64::out5, IA64::out6, IA64::out7
  };

  static const unsigned FPArgRegs[] = {
    IA64::F8,  IA64::F9,  IA64::F10, IA64::F11, 
    IA64::F12, IA64::F13, IA64::F14, IA64::F15
  };

  SDOperand InFlag;
  
  // save the current GP, SP and RP : FIXME: do we need to do all 3 always?
  SDOperand GPBeforeCall = DAG.getCopyFromReg(Chain, IA64::r1, MVT::i64, InFlag);
  Chain = GPBeforeCall;
  InFlag = Chain.getValue(1);
  SDOperand SPBeforeCall = DAG.getCopyFromReg(Chain, IA64::r12, MVT::i64, InFlag);
  Chain = SPBeforeCall;
  InFlag = Chain.getValue(1);
  SDOperand RPBeforeCall = DAG.getCopyFromReg(Chain, IA64::rp, MVT::i64, InFlag);
  Chain = RPBeforeCall;
  InFlag = Chain.getValue(1);

  // Build a sequence of copy-to-reg nodes chained together with token chain
  // and flag operands which copy the outgoing integer args into regs out[0-7]
  // mapped 1:1 and the FP args into regs F8-F15 "lazily"
  // TODO: for performance, we should only copy FP args into int regs when we
  // know this is required (i.e. for varardic or external (unknown) functions)

  // first to the FP->(integer representation) conversions, these are
  // free-floating
  unsigned seenConverts = 0;
  for (unsigned i = 0, e = RegValuesToPass.size(); i != e; ++i) {
    if(MVT::isFloatingPoint(RegValuesToPass[i].getValueType())) {
      Chain = DAG.getCopyToReg(Chain, IntArgRegs[i], Converts[seenConverts++]);
    }
  }

  // next copy args into the usual places, these are flagged
  unsigned usedFPArgs = 0;
  for (unsigned i = 0, e = RegValuesToPass.size(); i != e; ++i) {
    Chain = DAG.getCopyToReg(Chain,
      MVT::isInteger(RegValuesToPass[i].getValueType()) ?
                                          IntArgRegs[i] : FPArgRegs[usedFPArgs++],
      RegValuesToPass[i], InFlag);
    InFlag = Chain.getValue(1);
  }

  // If the callee is a GlobalAddress node (quite common, every direct call is)
  // turn it into a TargetGlobalAddress node so that legalize doesn't hack it.
/*
  if (GlobalAddressSDNode *G = dyn_cast<GlobalAddressSDNode>(Callee)) {
    Callee = DAG.getTargetGlobalAddress(G->getGlobal(), MVT::i64);
  }
*/

  std::vector<MVT::ValueType> NodeTys;
  std::vector<SDOperand> CallOperands;
  NodeTys.push_back(MVT::Other);   // Returns a chain
  NodeTys.push_back(MVT::Flag);    // Returns a flag for retval copy to use.
  CallOperands.push_back(Chain);
  CallOperands.push_back(Callee);

  // emit the call itself
  if (InFlag.Val)
    CallOperands.push_back(InFlag);

/* out with the old...
    Chain = SDOperand(DAG.getCall(NodeTys, Chain, Callee, InFlag), 0);
  else
    Chain = SDOperand(DAG.getCall(NodeTys, Chain, Callee), 0);
*/
  // to make way for a hack:
  Chain = DAG.getNode(IA64ISD::BRCALL, NodeTys, CallOperands);
  InFlag = Chain.getValue(1);

  // restore the GP, SP and RP after the call  
  Chain = DAG.getCopyToReg(Chain, IA64::r1, GPBeforeCall, InFlag);
  InFlag = Chain.getValue(1);
  Chain = DAG.getCopyToReg(Chain, IA64::r12, SPBeforeCall, InFlag);
  InFlag = Chain.getValue(1);
  Chain = DAG.getCopyToReg(Chain, IA64::rp, RPBeforeCall, InFlag);
  InFlag = Chain.getValue(1);
 
  std::vector<MVT::ValueType> RetVals;
  RetVals.push_back(MVT::Other);
  RetVals.push_back(MVT::Flag);
 
  MVT::ValueType RetTyVT = getValueType(RetTy);
  SDOperand RetVal;
  if (RetTyVT != MVT::isVoid) {
    switch (RetTyVT) {
    default: assert(0 && "Unknown value type to return!");
    case MVT::i1:/* { // bools are just like other integers (returned in r8)
      SDOperand boolInR8 = DAG.getCopyFromReg(Chain, IA64::r8, MVT::i64, InFlag);
      RetVal = DAG.getTargetNode(IA64::CMPNE, MVT::i1, // FIXME: is this flagged correctly?
        DAG.getRegister(IA64::r0, MVT::i64), boolInR8, Chain, InFlag);
      Chain = RetVal.getValue(1);
      // Add a note to keep track of whether it is sign or zero extended - TODO: bools
      RetVal = DAG.getNode(RetTy->isSigned() ? ISD::AssertSext :ISD::AssertZext,
                           MVT::i64, RetVal, DAG.getValueType(RetTyVT));
      RetVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, RetVal);
      break;
    }*/
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
      RetVal = DAG.getCopyFromReg(Chain, IA64::r8, MVT::i64, InFlag);
      Chain = RetVal.getValue(1);
      
      // Add a note to keep track of whether it is sign or zero extended - TODO: bools
      RetVal = DAG.getNode(RetTy->isSigned() ? ISD::AssertSext :ISD::AssertZext,
                           MVT::i64, RetVal, DAG.getValueType(RetTyVT));
      RetVal = DAG.getNode(ISD::TRUNCATE, RetTyVT, RetVal);
      break;
    case MVT::i64:
      RetVal = DAG.getCopyFromReg(Chain, IA64::r8, MVT::i64, InFlag);
      Chain = RetVal.getValue(1);
      break;
    case MVT::f64:
      RetVal = DAG.getCopyFromReg(Chain, IA64::F8, MVT::f64, InFlag);
      Chain = RetVal.getValue(1);
      break;
    }
  }
  
  Chain = DAG.getNode(ISD::CALLSEQ_END, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  
  return std::make_pair(RetVal, Chain);
}

SDOperand
IA64TargetLowering::LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                 Value *VAListV, SelectionDAG &DAG) {
  // vastart just stores the address of the VarArgsFrameIndex slot.
  SDOperand FR = DAG.getFrameIndex(VarArgsFrameIndex, MVT::i64);
  return DAG.getNode(ISD::STORE, MVT::Other, Chain, FR,
                     VAListP, DAG.getSrcValue(VAListV));
}

std::pair<SDOperand,SDOperand> IA64TargetLowering::
LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
           const Type *ArgTy, SelectionDAG &DAG) {

  MVT::ValueType ArgVT = getValueType(ArgTy);
  SDOperand Val = DAG.getLoad(MVT::i64, Chain,
                              VAListP, DAG.getSrcValue(VAListV));
  SDOperand Result = DAG.getLoad(ArgVT, DAG.getEntryNode(), Val,
                                 DAG.getSrcValue(NULL));
  unsigned Amt;
  if (ArgVT == MVT::i32 || ArgVT == MVT::f32)
    Amt = 8;
  else {
    assert((ArgVT == MVT::i64 || ArgVT == MVT::f64) &&
           "Other types should have been promoted for varargs!");
    Amt = 8;
  }
  Val = DAG.getNode(ISD::ADD, Val.getValueType(), Val,
                    DAG.getConstant(Amt, Val.getValueType()));
  Chain = DAG.getNode(ISD::STORE, MVT::Other, Chain,
                      Val, VAListP, DAG.getSrcValue(VAListV));
  return std::make_pair(Result, Chain);
}



std::pair<SDOperand, SDOperand> IA64TargetLowering::
LowerFrameReturnAddress(bool isFrameAddress, SDOperand Chain, unsigned Depth,
                        SelectionDAG &DAG) {
  assert(0 && "LowerFrameReturnAddress unimplemented");
  abort();
}

