//===-- IA64ISelPattern.cpp - A pattern matching inst selector for IA64 ---===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Duraid Madina and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for IA64.
//
//===----------------------------------------------------------------------===//

#include "IA64.h"
#include "IA64InstrBuilder.h"
#include "IA64RegisterInfo.h"
#include "IA64MachineFunctionInfo.h"
#include "llvm/Constants.h"                   // FIXME: REMOVE
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineConstantPool.h" // FIXME: REMOVE
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/Statistic.h"
#include <set>
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  IA64TargetLowering - IA64 Implementation of the TargetLowering interface
namespace {
  class IA64TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    
    //int ReturnAddrIndex;              // FrameIndex for return slot.
    unsigned GP, SP, RP; // FIXME - clean this mess up
  public:

   unsigned VirtGPR; // this is public so it can be accessed in the selector
   // for ISD::RET down below. add an accessor instead? FIXME

   IA64TargetLowering(TargetMachine &TM) : TargetLowering(TM) {
      
      // register class for general registers
      addRegisterClass(MVT::i64, IA64::GRRegisterClass);

      // register class for FP registers
      addRegisterClass(MVT::f64, IA64::FPRegisterClass);
      
      // register class for predicate registers 
      addRegisterClass(MVT::i1, IA64::PRRegisterClass);
      
      setOperationAction(ISD::FP_ROUND_INREG   , MVT::f32  , Expand);

      setSetCCResultType(MVT::i1); 
      setShiftAmountType(MVT::i64);

      setOperationAction(ISD::EXTLOAD          , MVT::i1   , Promote);
      setOperationAction(ISD::EXTLOAD          , MVT::f32  , Promote);

      setOperationAction(ISD::ZEXTLOAD         , MVT::i1   , Expand);
      setOperationAction(ISD::ZEXTLOAD         , MVT::i32  , Expand);

      setOperationAction(ISD::SEXTLOAD         , MVT::i1   , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i8   , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i16  , Expand);

      setOperationAction(ISD::SREM             , MVT::f32  , Expand);
      setOperationAction(ISD::SREM             , MVT::f64  , Expand);

      setOperationAction(ISD::UREM             , MVT::f32  , Expand);
      setOperationAction(ISD::UREM             , MVT::f64  , Expand);
      
      setOperationAction(ISD::MEMMOVE          , MVT::Other, Expand);
      setOperationAction(ISD::MEMSET           , MVT::Other, Expand);
      setOperationAction(ISD::MEMCPY           , MVT::Other, Expand);

      
      computeRegisterProperties();

      addLegalFPImmediate(+0.0);
      addLegalFPImmediate(+1.0);
      addLegalFPImmediate(-0.0);
      addLegalFPImmediate(-1.0);
    }

    /// LowerArguments - This hook must be implemented to indicate how we should
    /// lower the arguments for the specified function, into the specified DAG.
    virtual std::vector<SDOperand>
    LowerArguments(Function &F, SelectionDAG &DAG);

    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
    LowerCallTo(SDOperand Chain, const Type *RetTy, SDOperand Callee,
                ArgListTy &Args, SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerVAStart(SDOperand Chain, SelectionDAG &DAG);

    virtual std::pair<SDOperand,SDOperand>
    LowerVAArgNext(bool isVANext, SDOperand Chain, SDOperand VAList,
                   const Type *ArgTy, SelectionDAG &DAG);

    virtual std::pair<SDOperand, SDOperand>
    LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                            SelectionDAG &DAG);

    void restoreGP_SP_RP(MachineBasicBlock* BB)
    {
      BuildMI(BB, IA64::MOV, 1, IA64::r1).addReg(GP);
      BuildMI(BB, IA64::MOV, 1, IA64::r12).addReg(SP);
      BuildMI(BB, IA64::MOV, 1, IA64::rp).addReg(RP);
    }

    void restoreRP(MachineBasicBlock* BB)
    {
      BuildMI(BB, IA64::MOV, 1, IA64::rp).addReg(RP);
    }

    void restoreGP(MachineBasicBlock* BB)
    {
      BuildMI(BB, IA64::MOV, 1, IA64::r1).addReg(GP);
    }

  };
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

  unsigned used_FPArgs=0; // how many FP args have been used so far?
  
  int count = 0;
  for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
    {
      SDOperand newroot, argt;
      if(count < 8) { // need to fix this logic? maybe.
	  
	switch (getValueType(I->getType())) {
	  default:
	    std::cerr << "ERROR in LowerArgs: unknown type "
	      << getValueType(I->getType()) << "\n";
	    abort();
	  case MVT::f32:
	    // fixme? (well, will need to for weird FP structy stuff, 
	    // see intel ABI docs)
	  case MVT::f64:
	    BuildMI(&BB, IA64::IDEF, 0, args_FP[used_FPArgs]);
	    // floating point args go into f8..f15 as-needed, the increment
	    argVreg[count] =                              // is below..:
	    MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::f64));
	    // FP args go into f8..f15 as needed: (hence the ++)
	    argPreg[count] = args_FP[used_FPArgs++];
	    argOpc[count] = IA64::FMOV;
	    argt = newroot = DAG.getCopyFromReg(argVreg[count],
		getValueType(I->getType()), DAG.getRoot());
	    break;
	  case MVT::i1: // NOTE: as far as C abi stuff goes,
	                // bools are just boring old ints
	  case MVT::i8:
	  case MVT::i16:
	  case MVT::i32:
	  case MVT::i64:
	    BuildMI(&BB, IA64::IDEF, 0, args_int[count]);
	    argVreg[count] = 
	    MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
	    argPreg[count] = args_int[count];
	    argOpc[count] = IA64::MOV; 
	    argt = newroot =
	      DAG.getCopyFromReg(argVreg[count], MVT::i64, DAG.getRoot());
	    if ( getValueType(I->getType()) != MVT::i64)
	      argt = DAG.getNode(ISD::TRUNCATE, getValueType(I->getType()),
		  newroot);
	    break;
	}
      } else { // more than 8 args go into the frame
	// Create the frame index object for this incoming parameter...
	int FI = MFI->CreateFixedObject(8, 16 + 8 * (count - 8));
        
	// Create the SelectionDAG nodes corresponding to a load 
	//from this parameter
	SDOperand FIN = DAG.getFrameIndex(FI, MVT::i64);
	argt = newroot = DAG.getLoad(getValueType(I->getType()), 
	    DAG.getEntryNode(), FIN);
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

  for (int i = 0; i < count && i < 8; ++i) {
    BuildMI(&BB, argOpc[i], 1, argVreg[i]).addReg(argPreg[i]);
  }
 
  return ArgValues;
}
  
std::pair<SDOperand, SDOperand>
IA64TargetLowering::LowerCallTo(SDOperand Chain,
                               const Type *RetTy, SDOperand Callee,
                               ArgListTy &Args, SelectionDAG &DAG) {

  MachineFunction &MF = DAG.getMachineFunction();

// fow now, we are overly-conservative and pretend that all 8
// outgoing registers (out0-out7) are always used. FIXME

// update comment line 137 of MachineFunction.h
  MF.getInfo<IA64FunctionInfo>()->outRegsUsed=8;
  
  unsigned NumBytes = 16;
  if (Args.size() > 8)
    NumBytes += (Args.size() - 8) * 8;
  
  Chain = DAG.getNode(ISD::ADJCALLSTACKDOWN, MVT::Other, Chain,
                        DAG.getConstant(NumBytes, getPointerTy()));
  
  std::vector<SDOperand> args_to_use;
  for (unsigned i = 0, e = Args.size(); i != e; ++i)
    {
      switch (getValueType(Args[i].second)) {
      default: assert(0 && "unexpected argument type!");
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
	//promote to 64-bits, sign/zero extending based on type
	//of the argument
	if(Args[i].second->isSigned())
	  Args[i].first = DAG.getNode(ISD::SIGN_EXTEND, MVT::i64,
	      Args[i].first);
	else
	  Args[i].first = DAG.getNode(ISD::ZERO_EXTEND, MVT::i64,
	      Args[i].first);
	break;
      case MVT::f32:
	//promote to 64-bits
	Args[i].first = DAG.getNode(ISD::FP_EXTEND, MVT::f64, Args[i].first);
      case MVT::f64:
      case MVT::i64:
        break;
      }
      args_to_use.push_back(Args[i].first);
    }

  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);
  if (RetTyVT != MVT::isVoid)
    RetVals.push_back(RetTyVT);
  RetVals.push_back(MVT::Other);

  SDOperand TheCall = SDOperand(DAG.getCall(RetVals, Chain,
	Callee, args_to_use), 0);
  Chain = TheCall.getValue(RetTyVT != MVT::isVoid);
  Chain = DAG.getNode(ISD::ADJCALLSTACKUP, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
  return std::make_pair(TheCall, Chain);
}

std::pair<SDOperand, SDOperand>
IA64TargetLowering::LowerVAStart(SDOperand Chain, SelectionDAG &DAG) {
  // vastart just returns the address of the VarArgsFrameIndex slot.
  return std::make_pair(DAG.getFrameIndex(VarArgsFrameIndex, MVT::i64), Chain);
}

std::pair<SDOperand,SDOperand> IA64TargetLowering::
LowerVAArgNext(bool isVANext, SDOperand Chain, SDOperand VAList,
               const Type *ArgTy, SelectionDAG &DAG) {
 
  assert(0 && "LowerVAArgNext not done yet!\n");
}
               

std::pair<SDOperand, SDOperand> IA64TargetLowering::
LowerFrameReturnAddress(bool isFrameAddress, SDOperand Chain, unsigned Depth,
                        SelectionDAG &DAG) {

  assert(0 && "LowerFrameReturnAddress not done yet\n");
}


namespace {

  //===--------------------------------------------------------------------===//
  /// ISel - IA64 specific code to select IA64 machine instructions for
  /// SelectionDAG operations.
  ///
  class ISel : public SelectionDAGISel {
    /// IA64Lowering - This object fully describes how to lower LLVM code to an
    /// IA64-specific SelectionDAG.
    IA64TargetLowering IA64Lowering;

    /// ExprMap - As shared expressions are codegen'd, we keep track of which
    /// vreg the value is produced in, so we only emit one copy of each compiled
    /// tree.
    std::map<SDOperand, unsigned> ExprMap;
    std::set<SDOperand> LoweredTokens;

  public:
    ISel(TargetMachine &TM) : SelectionDAGISel(IA64Lowering), IA64Lowering(TM) {
    }

    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);

//    bool isFoldableLoad(SDOperand Op);
//    void EmitFoldedLoad(SDOperand Op, IA64AddressMode &AM);

    unsigned SelectExpr(SDOperand N);
    void Select(SDOperand N);
  };
}

/// InstructionSelectBasicBlock - This callback is invoked by SelectionDAGISel
/// when it has created a SelectionDAG for us to codegen.
void ISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {

  // Codegen the basic block.
  Select(DAG.getRoot());

  // Clear state used for selection.
  ExprMap.clear();
  LoweredTokens.clear();
}

unsigned ISel::SelectExpr(SDOperand N) {
  unsigned Result;
  unsigned Tmp1, Tmp2, Tmp3;
  unsigned Opc = 0;
  MVT::ValueType DestType = N.getValueType();

  unsigned opcode = N.getOpcode();

  SDNode *Node = N.Val;
  SDOperand Op0, Op1;

  if (Node->getOpcode() == ISD::CopyFromReg)
    // Just use the specified register as our input.
    return dyn_cast<RegSDNode>(Node)->getReg();
  
  unsigned &Reg = ExprMap[N];
  if (Reg) return Reg;
  
  if (N.getOpcode() != ISD::CALL)
    Reg = Result = (N.getValueType() != MVT::Other) ?
      MakeReg(N.getValueType()) : 1;
  else {
    // If this is a call instruction, make sure to prepare ALL of the result
    // values as well as the chain.
    if (Node->getNumValues() == 1)
      Reg = Result = 1;  // Void call, just a chain.
    else {
      Result = MakeReg(Node->getValueType(0));
      ExprMap[N.getValue(0)] = Result;
      for (unsigned i = 1, e = N.Val->getNumValues()-1; i != e; ++i)
        ExprMap[N.getValue(i)] = MakeReg(Node->getValueType(i));
      ExprMap[SDOperand(Node, Node->getNumValues()-1)] = 1;
    }
  }
  
  switch (N.getOpcode()) {
  default:
    Node->dump();
    assert(0 && "Node not handled!\n");

  case ISD::FrameIndex: {
    Tmp1 = cast<FrameIndexSDNode>(N)->getIndex();
    BuildMI(BB, IA64::MOV, 1, Result).addFrameIndex(Tmp1);
    return Result;
  }

  case ISD::ConstantPool: {
    Tmp1 = cast<ConstantPoolSDNode>(N)->getIndex();
    IA64Lowering.restoreGP(BB); // FIXME: do i really need this?
    BuildMI(BB, IA64::ADD, 2, Result).addConstantPoolIndex(Tmp1)
      .addReg(IA64::r1);
    return Result;
  }

  case ISD::ConstantFP: {
    Tmp1 = Result;   // Intermediate Register
    if (cast<ConstantFPSDNode>(N)->getValue() < 0.0 ||
        cast<ConstantFPSDNode>(N)->isExactlyValue(-0.0))
      Tmp1 = MakeReg(MVT::f64);

    if (cast<ConstantFPSDNode>(N)->isExactlyValue(+0.0) ||
        cast<ConstantFPSDNode>(N)->isExactlyValue(-0.0))
      BuildMI(BB, IA64::FMOV, 1, Tmp1).addReg(IA64::F0); // load 0.0
    else if (cast<ConstantFPSDNode>(N)->isExactlyValue(+1.0) ||
             cast<ConstantFPSDNode>(N)->isExactlyValue(-1.0))
      BuildMI(BB, IA64::FMOV, 1, Tmp1).addReg(IA64::F1); // load 1.0
    else
      assert(0 && "Unexpected FP constant!");
    if (Tmp1 != Result)
      // we multiply by +1.0, negate (this is FNMA), and then add 0.0
      BuildMI(BB, IA64::FNMA, 3, Result).addReg(Tmp1).addReg(IA64::F1)
	.addReg(IA64::F0);
    return Result;
  }

  case ISD::DYNAMIC_STACKALLOC: {
    // Generate both result values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    // FIXME: We are currently ignoring the requested alignment for handling
    // greater than the stack alignment.  This will need to be revisited at some
    // point.  Align = N.getOperand(2);

    if (!isa<ConstantSDNode>(N.getOperand(2)) ||
        cast<ConstantSDNode>(N.getOperand(2))->getValue() != 0) {
      std::cerr << "Cannot allocate stack object with greater alignment than"
                << " the stack alignment yet!";
      abort();
    }
  
    Select(N.getOperand(0));
    if (ConstantSDNode* CN = dyn_cast<ConstantSDNode>(N.getOperand(1)))
    {
      if (CN->getValue() < 32000)
      {
        BuildMI(BB, IA64::ADDIMM22, 2, IA64::r12).addReg(IA64::r12)
	  .addImm(-CN->getValue());
      } else {
        Tmp1 = SelectExpr(N.getOperand(1));
        // Subtract size from stack pointer, thereby allocating some space.
        BuildMI(BB, IA64::SUB, 2, IA64::r12).addReg(IA64::r12).addReg(Tmp1);
      }
    } else {
      Tmp1 = SelectExpr(N.getOperand(1));
      // Subtract size from stack pointer, thereby allocating some space.
      BuildMI(BB, IA64::SUB, 2, IA64::r12).addReg(IA64::r12).addReg(Tmp1);
    }

    // Put a pointer to the space into the result register, by copying the
    // stack pointer.
    BuildMI(BB, IA64::MOV, 1, Result).addReg(IA64::r12);
    return Result;
  }
    
  case ISD::SELECT: {
      Tmp1 = SelectExpr(N.getOperand(0)); //Cond
      Tmp2 = SelectExpr(N.getOperand(1)); //Use if TRUE
      Tmp3 = SelectExpr(N.getOperand(2)); //Use if FALSE

      // a temporary predicate register to hold the complement of the
      // condition:
      unsigned CondComplement=MakeReg(MVT::i1);
      unsigned bogusTemp=MakeReg(MVT::i1);

      unsigned bogoResult;
      
      switch (N.getOperand(1).getValueType()) {
	default: assert(0 &&
	"ISD::SELECT: 'select'ing something other than i64 or f64!\n");
	case MVT::i64:
	  bogoResult=MakeReg(MVT::i64);
	  break;
	case MVT::f64:
	  bogoResult=MakeReg(MVT::f64);
	  break;
      }
      // set up the complement predicate reg (CondComplement = NOT Tmp1)
      BuildMI(BB, IA64::CMPEQ, 2, bogusTemp).addReg(IA64::r0).addReg(IA64::r0);
      BuildMI(BB, IA64::TPCMPNE, 3, CondComplement).addReg(bogusTemp)
	.addReg(IA64::r0).addReg(IA64::r0).addReg(Tmp1);
	
      // and do a 'conditional move'
      BuildMI(BB, IA64::PMOV, 2, bogoResult).addReg(Tmp2).addReg(Tmp1);
      BuildMI(BB, IA64::CMOV, 2, Result).addReg(bogoResult).addReg(Tmp3)
	.addReg(CondComplement);
 
      return Result;
  }
  
  case ISD::Constant: {
    unsigned depositPos=0;
    unsigned depositLen=0;
    switch (N.getValueType()) {
      default: assert(0 && "Cannot use constants of this type!");
      case MVT::i1: { // if a bool, we don't 'load' so much as generate
		      // the constant:
		      if(cast<ConstantSDNode>(N)->getValue())  // true:
			BuildMI(BB, IA64::CMPEQ, 2, Result)
			  .addReg(IA64::r0).addReg(IA64::r0);
		      else // false:
			BuildMI(BB, IA64::CMPNE, 2, Result)
			  .addReg(IA64::r0).addReg(IA64::r0);
		      return Result;
		    }
      case MVT::i64: Opc = IA64::MOVLI32; break;
    }
   
    int64_t immediate = cast<ConstantSDNode>(N)->getValue();
    if(immediate>>32) { // if our immediate really is big:
      int highPart = immediate>>32;
      int lowPart = immediate&0xFFFFFFFF;
      unsigned dummy = MakeReg(MVT::i64);
      unsigned dummy2 = MakeReg(MVT::i64);
      unsigned dummy3 = MakeReg(MVT::i64);
     
      BuildMI(BB, IA64::MOVLI32, 1, dummy).addImm(highPart);
      BuildMI(BB, IA64::SHLI, 2, dummy2).addReg(dummy).addImm(32);
      BuildMI(BB, IA64::MOVLI32, 1, dummy3).addImm(lowPart);
      BuildMI(BB, IA64::ADD, 2, Result).addReg(dummy2).addReg(dummy3);
    } else {
      BuildMI(BB, IA64::MOVLI32, 1, Result).addImm(immediate);
    }

  return Result;
  }
    
  case ISD::GlobalAddress: {
    GlobalValue *GV = cast<GlobalAddressSDNode>(N)->getGlobal();
    unsigned Tmp1 = MakeReg(MVT::i64);
    BuildMI(BB, IA64::ADD, 2, Tmp1).addGlobalAddress(GV).addReg(IA64::r1);
                                                        //r1==GP
    BuildMI(BB, IA64::LD8, 1, Result).addReg(Tmp1);
    return Result;
  }
  
  case ISD::ExternalSymbol: {
    const char *Sym = cast<ExternalSymbolSDNode>(N)->getSymbol();
    assert(0 && "ISD::ExternalSymbol not done yet\n");
    //XXX BuildMI(BB, IA64::MOV, 1, Result).addExternalSymbol(Sym);
    return Result;
  }

  case ISD::FP_EXTEND: {
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, IA64::FMOV, 1, Result).addReg(Tmp1);
    return Result;
  }

  case ISD::ZERO_EXTEND: {
    Tmp1 = SelectExpr(N.getOperand(0)); // value
    
    switch (N.getOperand(0).getValueType()) {
    default: assert(0 && "Cannot zero-extend this type!");
    case MVT::i8:  Opc = IA64::ZXT1; break;
    case MVT::i16: Opc = IA64::ZXT2; break;
    case MVT::i32: Opc = IA64::ZXT4; break;

    // we handle bools differently! : 
    case MVT::i1: { // if the predicate reg has 1, we want a '1' in our GR.
		    unsigned dummy = MakeReg(MVT::i64);
		    // first load zero:
		    BuildMI(BB, IA64::MOV, 1, dummy).addReg(IA64::r0);
		    // ...then conditionally (PR:Tmp1) add 1:
		    BuildMI(BB, IA64::CADDIMM22, 3, Result).addReg(dummy)
		      .addImm(1).addReg(Tmp1);
		    return Result; // XXX early exit!
		  }
    }

    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;
   }

  case ISD::SIGN_EXTEND: {   // we should only have to handle i1 -> i64 here!!!

assert(0 && "hmm, ISD::SIGN_EXTEND: shouldn't ever be reached. bad luck!\n");

    Tmp1 = SelectExpr(N.getOperand(0)); // value
    
    switch (N.getOperand(0).getValueType()) {
    default: assert(0 && "Cannot sign-extend this type!");
    case MVT::i1:  assert(0 && "trying to sign extend a bool? ow.\n");
		   Opc = IA64::SXT1; break;
		   // FIXME: for now, we treat bools the same as i8s
    case MVT::i8:  Opc = IA64::SXT1; break;
    case MVT::i16: Opc = IA64::SXT2; break;
    case MVT::i32: Opc = IA64::SXT4; break;
    }

    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;
   }

  case ISD::TRUNCATE: {
    // we use the funky dep.z (deposit (zero)) instruction to deposit bits
    // of R0 appropriately.
    switch (N.getOperand(0).getValueType()) {
    default: assert(0 && "Unknown truncate!");
    case MVT::i64: break;
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    unsigned depositPos, depositLen;

    switch (N.getValueType()) {
    default: assert(0 && "Unknown truncate!");
    case MVT::i1: {
      // if input (normal reg) is 0, 0!=0 -> false (0), if 1, 1!=0 ->true (1):
		    BuildMI(BB, IA64::CMPNE, 2, Result).addReg(Tmp1)
		      .addReg(IA64::r0);
		    return Result; // XXX early exit!
		  }
    case MVT::i8:  depositPos=0; depositLen=8;  break;
    case MVT::i16: depositPos=0; depositLen=16; break;
    case MVT::i32: depositPos=0; depositLen=32; break;
    }
    BuildMI(BB, IA64::DEPZ, 1, Result).addReg(Tmp1)
      .addImm(depositPos).addImm(depositLen);
    return Result;
  }

/*			
  case ISD::FP_ROUND: {
    assert (DestType == MVT::f32 && N.getOperand(0).getValueType() == MVT::f64 &&
	"error: trying to FP_ROUND something other than f64 -> f32!\n");
    Tmp1 = SelectExpr(N.getOperand(0));
    BuildMI(BB, IA64::FADDS, 2, Result).addReg(Tmp1).addReg(IA64::F0);
    // we add 0.0 using a single precision add to do rounding
    return Result;
  }
*/

// FIXME: the following 4 cases need cleaning
  case ISD::SINT_TO_FP: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = MakeReg(MVT::f64);
    unsigned dummy = MakeReg(MVT::f64);
    BuildMI(BB, IA64::SETFSIG, 1, Tmp2).addReg(Tmp1);
    BuildMI(BB, IA64::FCVTXF, 1, dummy).addReg(Tmp2);
    BuildMI(BB, IA64::FNORMD, 1, Result).addReg(dummy);
    return Result;
  }

  case ISD::UINT_TO_FP: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = MakeReg(MVT::f64);
    unsigned dummy = MakeReg(MVT::f64);
    BuildMI(BB, IA64::SETFSIG, 1, Tmp2).addReg(Tmp1);
    BuildMI(BB, IA64::FCVTXUF, 1, dummy).addReg(Tmp2);
    BuildMI(BB, IA64::FNORMD, 1, Result).addReg(dummy);
    return Result;
  }

  case ISD::FP_TO_SINT: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = MakeReg(MVT::f64);
    BuildMI(BB, IA64::FCVTFXTRUNC, 1, Tmp2).addReg(Tmp1);
    BuildMI(BB, IA64::GETFSIG, 1, Result).addReg(Tmp2);
    return Result;
  }

  case ISD::FP_TO_UINT: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = MakeReg(MVT::f64);
    BuildMI(BB, IA64::FCVTFXUTRUNC, 1, Tmp2).addReg(Tmp1);
    BuildMI(BB, IA64::GETFSIG, 1, Result).addReg(Tmp2);
    return Result;
  }

  case ISD::ADD: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    if(DestType != MVT::f64)
      BuildMI(BB, IA64::ADD, 2, Result).addReg(Tmp1).addReg(Tmp2); // int
    else
      BuildMI(BB, IA64::FADD, 2, Result).addReg(Tmp1).addReg(Tmp2); // FP
    return Result;
  }

  case ISD::MUL: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    if(DestType != MVT::f64) { // integer multiply, emit some code (FIXME)
      unsigned TempFR1=MakeReg(MVT::f64);
      unsigned TempFR2=MakeReg(MVT::f64);
      unsigned TempFR3=MakeReg(MVT::f64);
      BuildMI(BB, IA64::SETFSIG, 1, TempFR1).addReg(Tmp1);
      BuildMI(BB, IA64::SETFSIG, 1, TempFR2).addReg(Tmp2);
      BuildMI(BB, IA64::XMAL, 1, TempFR3).addReg(TempFR1).addReg(TempFR2)
	.addReg(IA64::F0);
      BuildMI(BB, IA64::GETFSIG, 1, Result).addReg(TempFR3);
    }
    else  // floating point multiply
      BuildMI(BB, IA64::FMPY, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;
  }
  
  case ISD::SUB: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    if(DestType != MVT::f64)
      BuildMI(BB, IA64::SUB, 2, Result).addReg(Tmp1).addReg(Tmp2);
    else
      BuildMI(BB, IA64::FSUB, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;
  }
		 
  case ISD::AND: {
     switch (N.getValueType()) {
    default: assert(0 && "Cannot AND this type!");
    case MVT::i1: { // if a bool, we emit a pseudocode AND
      unsigned pA = SelectExpr(N.getOperand(0));
      unsigned pB = SelectExpr(N.getOperand(1));
       
/* our pseudocode for AND is:
 *
(pA) cmp.eq.unc pC,p0 = r0,r0   // pC = pA
     cmp.eq pTemp,p0 = r0,r0    // pTemp = NOT pB
     ;;
(pB) cmp.ne pTemp,p0 = r0,r0
     ;;
(pTemp)cmp.ne pC,p0 = r0,r0    // if (NOT pB) pC = 0

*/
      unsigned pTemp = MakeReg(MVT::i1);
     
      unsigned bogusTemp1 = MakeReg(MVT::i1);
      unsigned bogusTemp2 = MakeReg(MVT::i1);
      unsigned bogusTemp3 = MakeReg(MVT::i1);
      unsigned bogusTemp4 = MakeReg(MVT::i1);
    
      BuildMI(BB, IA64::PCMPEQUNC, 3, bogusTemp1)
	.addReg(IA64::r0).addReg(IA64::r0).addReg(pA);
      BuildMI(BB, IA64::CMPEQ, 2, bogusTemp2)
	.addReg(IA64::r0).addReg(IA64::r0);
      BuildMI(BB, IA64::TPCMPNE, 3, pTemp)
	.addReg(bogusTemp2).addReg(IA64::r0).addReg(IA64::r0).addReg(pB);
      BuildMI(BB, IA64::TPCMPNE, 3, Result)
	.addReg(bogusTemp1).addReg(IA64::r0).addReg(IA64::r0).addReg(pTemp);
      break;
    }
    // if not a bool, we just AND away:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64: {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::AND, 2, Result).addReg(Tmp1).addReg(Tmp2);
      break;
    }
    }
    return Result;
  }
 
  case ISD::OR: {
  switch (N.getValueType()) {
    default: assert(0 && "Cannot OR this type!");
    case MVT::i1: { // if a bool, we emit a pseudocode OR
      unsigned pA = SelectExpr(N.getOperand(0));
      unsigned pB = SelectExpr(N.getOperand(1));

      unsigned pTemp1 = MakeReg(MVT::i1);
       
/* our pseudocode for OR is:
 *

pC = pA OR pB
-------------

(pA)	cmp.eq.unc pC,p0 = r0,r0  // pC = pA
	;;
(pB)	cmp.eq pC,p0 = r0,r0	// if (pB) pC = 1

*/
      BuildMI(BB, IA64::PCMPEQUNC, 3, pTemp1)
	.addReg(IA64::r0).addReg(IA64::r0).addReg(pA);
      BuildMI(BB, IA64::TPCMPEQ, 3, Result)
	.addReg(pTemp1).addReg(IA64::r0).addReg(IA64::r0).addReg(pB);
      break;
    }
    // if not a bool, we just OR away:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64: {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::OR, 2, Result).addReg(Tmp1).addReg(Tmp2);
      break;
    }
    }
    return Result;
  }
	 
  case ISD::XOR: {
     switch (N.getValueType()) {
    default: assert(0 && "Cannot XOR this type!");
    case MVT::i1: { // if a bool, we emit a pseudocode XOR
      unsigned pY = SelectExpr(N.getOperand(0));
      unsigned pZ = SelectExpr(N.getOperand(1));

/* one possible routine for XOR is:

      // Compute px = py ^ pz
        // using sum of products: px = (py & !pz) | (pz & !py)
        // Uses 5 instructions in 3 cycles.
        // cycle 1
(pz)    cmp.eq.unc      px = r0, r0     // px = pz
(py)    cmp.eq.unc      pt = r0, r0     // pt = py
        ;;
        // cycle 2
(pt)    cmp.ne.and      px = r0, r0     // px = px & !pt (px = pz & !pt)
(pz)    cmp.ne.and      pt = r0, r0     // pt = pt & !pz
        ;;
        } { .mmi
        // cycle 3
(pt)    cmp.eq.or       px = r0, r0     // px = px | pt

*** Another, which we use here, requires one scratch GR. it is:

        mov             rt = 0          // initialize rt off critical path
        ;;

        // cycle 1
(pz)    cmp.eq.unc      px = r0, r0     // px = pz
(pz)    mov             rt = 1          // rt = pz
        ;;
        // cycle 2
(py)    cmp.ne          px = 1, rt      // if (py) px = !pz

.. these routines kindly provided by Jim Hull
*/
      unsigned rt = MakeReg(MVT::i64);

      // these two temporaries will never actually appear,
      // due to the two-address form of some of the instructions below
      unsigned bogoPR = MakeReg(MVT::i1);  // becomes Result
      unsigned bogoGR = MakeReg(MVT::i64); // becomes rt

      BuildMI(BB, IA64::MOV, 1, bogoGR).addReg(IA64::r0);
      BuildMI(BB, IA64::PCMPEQUNC, 3, bogoPR)
	.addReg(IA64::r0).addReg(IA64::r0).addReg(pZ);
      BuildMI(BB, IA64::TPCADDIMM22, 2, rt)
	.addReg(bogoGR).addImm(1).addReg(pZ);
      BuildMI(BB, IA64::TPCMPIMM8NE, 3, Result)
	.addReg(bogoPR).addImm(1).addReg(rt).addReg(pY);
      break;
    }
    // if not a bool, we just XOR away:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64: {
      Tmp1 = SelectExpr(N.getOperand(0));
      Tmp2 = SelectExpr(N.getOperand(1));
      BuildMI(BB, IA64::XOR, 2, Result).addReg(Tmp1).addReg(Tmp2);
      break;
    }
    }
    return Result;
  }

  case ISD::SHL: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, IA64::SHL, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;
  }
  case ISD::SRL: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, IA64::SHRU, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;
  }
  case ISD::SRA: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, IA64::SHRS, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;
  }

  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SREM:
  case ISD::UREM: {

    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));

    bool isFP=false;

    if(DestType == MVT::f64) // XXX: we're not gonna be fed MVT::f32, are we?
      isFP=true;

    bool isModulus=false; // is it a division or a modulus?
    bool isSigned=false;

    switch(N.getOpcode()) {
      case ISD::SDIV:  isModulus=false; isSigned=true;  break;
      case ISD::UDIV:  isModulus=false; isSigned=false; break;
      case ISD::SREM:  isModulus=true;  isSigned=true;  break;
      case ISD::UREM:  isModulus=true;  isSigned=false; break;
    }

    unsigned TmpPR=MakeReg(MVT::i1);  // we need a scratch predicate register,
    unsigned TmpF1=MakeReg(MVT::f64); // and one metric truckload of FP regs.
    unsigned TmpF2=MakeReg(MVT::f64); // lucky we have IA64?
    unsigned TmpF3=MakeReg(MVT::f64); // well, the real FIXME is to have
    unsigned TmpF4=MakeReg(MVT::f64); // isTwoAddress forms of these
    unsigned TmpF5=MakeReg(MVT::f64); // FP instructions so we can end up with
    unsigned TmpF6=MakeReg(MVT::f64); // stuff like setf.sig f10=f10 etc.
    unsigned TmpF7=MakeReg(MVT::f64);
    unsigned TmpF8=MakeReg(MVT::f64);
    unsigned TmpF9=MakeReg(MVT::f64);
    unsigned TmpF10=MakeReg(MVT::f64);
    unsigned TmpF11=MakeReg(MVT::f64);
    unsigned TmpF12=MakeReg(MVT::f64);
    unsigned TmpF13=MakeReg(MVT::f64);
    unsigned TmpF14=MakeReg(MVT::f64);
    unsigned TmpF15=MakeReg(MVT::f64);
  
    // OK, emit some code:

    if(!isFP) {
      // first, load the inputs into FP regs.
      BuildMI(BB, IA64::SETFSIG, 1, TmpF1).addReg(Tmp1);
      BuildMI(BB, IA64::SETFSIG, 1, TmpF2).addReg(Tmp2);
      
      // next, convert the inputs to FP
      if(isSigned) {
	BuildMI(BB, IA64::FCVTXF, 1, TmpF3).addReg(TmpF1);
	BuildMI(BB, IA64::FCVTXF, 1, TmpF4).addReg(TmpF2);
      } else {
	BuildMI(BB, IA64::FCVTXUFS1, 1, TmpF3).addReg(TmpF1);
	BuildMI(BB, IA64::FCVTXUFS1, 1, TmpF4).addReg(TmpF2);
      }
      
    } else { // this is an FP divide/remainder, so we 'leak' some temp
             // regs and assign TmpF3=Tmp1, TmpF4=Tmp2
      TmpF3=Tmp1;
      TmpF4=Tmp2;
    }

    // we start by computing an approximate reciprocal (good to 9 bits?)
    // note, this instruction writes _both_ TmpF5 (answer) and tmpPR (predicate)
    // FIXME: or at least, it should!!
    BuildMI(BB, IA64::FRCPAS1FLOAT, 2, TmpF5).addReg(TmpF3).addReg(TmpF4);
    BuildMI(BB, IA64::FRCPAS1PREDICATE, 2, TmpPR).addReg(TmpF3).addReg(TmpF4);

    // now we apply newton's method, thrice! (FIXME: this is ~72 bits of
    // precision, don't need this much for f32/i32)
    BuildMI(BB, IA64::CFNMAS1, 4, TmpF6)
      .addReg(TmpF4).addReg(TmpF5).addReg(IA64::F1).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4, TmpF7)
      .addReg(TmpF3).addReg(TmpF5).addReg(IA64::F0).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4, TmpF8)
      .addReg(TmpF6).addReg(TmpF6).addReg(IA64::F0).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4, TmpF9)
      .addReg(TmpF6).addReg(TmpF7).addReg(TmpF7).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4,TmpF10)
      .addReg(TmpF6).addReg(TmpF5).addReg(TmpF5).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4,TmpF11)
      .addReg(TmpF8).addReg(TmpF9).addReg(TmpF9).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4,TmpF12)
      .addReg(TmpF8).addReg(TmpF10).addReg(TmpF10).addReg(TmpPR);
    BuildMI(BB, IA64::CFNMAS1, 4,TmpF13)
      .addReg(TmpF4).addReg(TmpF11).addReg(TmpF3).addReg(TmpPR);
    BuildMI(BB, IA64::CFMAS1,  4,TmpF14)
      .addReg(TmpF13).addReg(TmpF12).addReg(TmpF11).addReg(TmpPR);

    if(!isFP) {
      // round to an integer
      if(isSigned)
	BuildMI(BB, IA64::FCVTFXTRUNCS1, 1, TmpF15).addReg(TmpF14);
      else
	BuildMI(BB, IA64::FCVTFXUTRUNCS1, 1, TmpF15).addReg(TmpF14);
    } else {
      BuildMI(BB, IA64::FMOV, 1, TmpF15).addReg(TmpF14);
     // EXERCISE: can you see why TmpF15=TmpF14 does not work here, and
     // we really do need the above FMOV? ;)
    }

    if(!isModulus) {
      if(isFP)
	BuildMI(BB, IA64::FMOV, 1, Result).addReg(TmpF15);
      else
	BuildMI(BB, IA64::GETFSIG, 1, Result).addReg(TmpF15);
    } else { // this is a modulus
      if(!isFP) {
	// answer = q * (-b) + a
	unsigned ModulusResult = MakeReg(MVT::f64);
	unsigned TmpF = MakeReg(MVT::f64);
	unsigned TmpI = MakeReg(MVT::i64);
	BuildMI(BB, IA64::SUB, 2, TmpI).addReg(IA64::r0).addReg(Tmp2);
	BuildMI(BB, IA64::SETFSIG, 1, TmpF).addReg(TmpI);
	BuildMI(BB, IA64::XMAL, 3, ModulusResult)
	  .addReg(TmpF15).addReg(TmpF).addReg(TmpF1);
	BuildMI(BB, IA64::GETFSIG, 1, Result).addReg(ModulusResult);
      } else { // FP modulus! The horror... the horror....
	assert(0 && "sorry, no FP modulus just yet!\n!\n");
      }
    }

    return Result;
  }

  case ISD::ZERO_EXTEND_INREG: {
    Tmp1 = SelectExpr(N.getOperand(0));
    MVTSDNode* MVN = dyn_cast<MVTSDNode>(Node);
    switch(MVN->getExtraValueType())
    {
    default:
      Node->dump();
      assert(0 && "don't know how to zero extend this type");
      break;
    case MVT::i8: Opc = IA64::ZXT1; break;
    case MVT::i16: Opc = IA64::ZXT2; break;
    case MVT::i32: Opc = IA64::ZXT4; break;
    }
    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;
  }
 
  case ISD::SIGN_EXTEND_INREG: {
    Tmp1 = SelectExpr(N.getOperand(0));
    MVTSDNode* MVN = dyn_cast<MVTSDNode>(Node);
    switch(MVN->getExtraValueType())
    {
    default:
      Node->dump();
      assert(0 && "don't know how to sign extend this type");
      break;
    case MVT::i8: Opc = IA64::SXT1; break;
    case MVT::i16: Opc = IA64::SXT2; break;
    case MVT::i32: Opc = IA64::SXT4; break;
    }
    BuildMI(BB, Opc, 1, Result).addReg(Tmp1);
    return Result;
  }

  case ISD::SETCC: {
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    if (SetCCSDNode *SetCC = dyn_cast<SetCCSDNode>(Node)) {
      if (MVT::isInteger(SetCC->getOperand(0).getValueType())) {
	switch (SetCC->getCondition()) {
	default: assert(0 && "Unknown integer comparison!");
	case ISD::SETEQ:
	  BuildMI(BB, IA64::CMPEQ, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETGT:
	  BuildMI(BB, IA64::CMPGT, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETGE:
	  BuildMI(BB, IA64::CMPGE, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETLT:
	  BuildMI(BB, IA64::CMPLT, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETLE:
	  BuildMI(BB, IA64::CMPLE, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETNE:
	  BuildMI(BB, IA64::CMPNE, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETULT:
	  BuildMI(BB, IA64::CMPLTU, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETUGT:
	  BuildMI(BB, IA64::CMPGTU, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETULE:
	  BuildMI(BB, IA64::CMPLEU, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETUGE:
	  BuildMI(BB, IA64::CMPGEU, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	}
      }
      else { // if not integer, should be FP. FIXME: what about bools? ;)
	assert(SetCC->getOperand(0).getValueType() != MVT::f32 &&
	    "error: SETCC should have had incoming f32 promoted to f64!\n");
	switch (SetCC->getCondition()) {
	default: assert(0 && "Unknown FP comparison!");
	case ISD::SETEQ:
	  BuildMI(BB, IA64::FCMPEQ, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETGT:
	  BuildMI(BB, IA64::FCMPGT, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETGE:
	  BuildMI(BB, IA64::FCMPGE, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETLT:
	  BuildMI(BB, IA64::FCMPLT, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETLE:
	  BuildMI(BB, IA64::FCMPLE, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETNE:
	  BuildMI(BB, IA64::FCMPNE, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETULT:
	  BuildMI(BB, IA64::FCMPLTU, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETUGT:
	  BuildMI(BB, IA64::FCMPGTU, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETULE:
	  BuildMI(BB, IA64::FCMPLEU, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETUGE:
	  BuildMI(BB, IA64::FCMPGEU, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	}
      }
    }
    else
      assert(0 && "this setcc not implemented yet");

    return Result;
  }

  case ISD::EXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::LOAD: {
    // Make sure we generate both values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

    bool isBool=false;
    
    if(opcode == ISD::LOAD) { // this is a LOAD
      switch (Node->getValueType(0)) {
	default: assert(0 && "Cannot load this type!");
	case MVT::i1:  Opc = IA64::LD1; isBool=true; break;
	      // FIXME: for now, we treat bool loads the same as i8 loads */
	case MVT::i8:  Opc = IA64::LD1; break;
	case MVT::i16: Opc = IA64::LD2; break;
	case MVT::i32: Opc = IA64::LD4; break;
	case MVT::i64: Opc = IA64::LD8; break;
		       
	case MVT::f32: Opc = IA64::LDF4; break;
	case MVT::f64: Opc = IA64::LDF8; break;
      }
    } else { // this is an EXTLOAD or ZEXTLOAD
      MVT::ValueType TypeBeingLoaded = cast<MVTSDNode>(Node)->getExtraValueType();
      switch (TypeBeingLoaded) {
	default: assert(0 && "Cannot extload/zextload this type!");
	// FIXME: bools?
	case MVT::i8: Opc = IA64::LD1; break;
	case MVT::i16: Opc = IA64::LD2; break;
	case MVT::i32: Opc = IA64::LD4; break;
	case MVT::f32: Opc = IA64::LDF4; break;
      }
    }
    
    SDOperand Chain = N.getOperand(0);
    SDOperand Address = N.getOperand(1);

    if(Address.getOpcode() == ISD::GlobalAddress) {
      Select(Chain);
      unsigned dummy = MakeReg(MVT::i64);
      unsigned dummy2 = MakeReg(MVT::i64);
      BuildMI(BB, IA64::ADD, 2, dummy)
	.addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal())
	.addReg(IA64::r1);
      BuildMI(BB, IA64::LD8, 1, dummy2).addReg(dummy);
      if(!isBool)
	BuildMI(BB, Opc, 1, Result).addReg(dummy2);
      else { // emit a little pseudocode to load a bool (stored in one byte)
	     // into a predicate register
	assert(Opc==IA64::LD1 && "problem loading a bool");
	unsigned dummy3 = MakeReg(MVT::i64);
	BuildMI(BB, Opc, 1, dummy3).addReg(dummy2);
	// we compare to 0. true? 0. false? 1.
	BuildMI(BB, IA64::CMPNE, 2, Result).addReg(dummy3).addReg(IA64::r0);
      }
    } else if(ConstantPoolSDNode *CP = dyn_cast<ConstantPoolSDNode>(Address)) {
      Select(Chain);
      IA64Lowering.restoreGP(BB);
      unsigned dummy = MakeReg(MVT::i64);
      BuildMI(BB, IA64::ADD, 2, dummy).addConstantPoolIndex(CP->getIndex())
	.addReg(IA64::r1); // CPI+GP
      if(!isBool)
	BuildMI(BB, Opc, 1, Result).addReg(dummy);
      else { // emit a little pseudocode to load a bool (stored in one byte)
	     // into a predicate register
	assert(Opc==IA64::LD1 && "problem loading a bool");
	unsigned dummy3 = MakeReg(MVT::i64);
	BuildMI(BB, Opc, 1, dummy3).addReg(dummy);
	// we compare to 0. true? 0. false? 1.
	BuildMI(BB, IA64::CMPNE, 2, Result).addReg(dummy3).addReg(IA64::r0);
      }
    } else if(Address.getOpcode() == ISD::FrameIndex) {
      Select(Chain);  // FIXME ? what about bools?
      unsigned dummy = MakeReg(MVT::i64);
      BuildMI(BB, IA64::MOV, 1, dummy)
	.addFrameIndex(cast<FrameIndexSDNode>(Address)->getIndex());
      if(!isBool)
	BuildMI(BB, Opc, 1, Result).addReg(dummy);
      else { // emit a little pseudocode to load a bool (stored in one byte)
	     // into a predicate register
	assert(Opc==IA64::LD1 && "problem loading a bool");
	unsigned dummy3 = MakeReg(MVT::i64);
	BuildMI(BB, Opc, 1, dummy3).addReg(dummy);
	// we compare to 0. true? 0. false? 1.
	BuildMI(BB, IA64::CMPNE, 2, Result).addReg(dummy3).addReg(IA64::r0);
      }
    } else { // none of the above... 
      Select(Chain);
      Tmp2 = SelectExpr(Address);
      if(!isBool)
	BuildMI(BB, Opc, 1, Result).addReg(Tmp2);
      else { // emit a little pseudocode to load a bool (stored in one byte)
	     // into a predicate register
	assert(Opc==IA64::LD1 && "problem loading a bool");
	unsigned dummy = MakeReg(MVT::i64);
	BuildMI(BB, Opc, 1, dummy).addReg(Tmp2);
	// we compare to 0. true? 0. false? 1.
	BuildMI(BB, IA64::CMPNE, 2, Result).addReg(dummy).addReg(IA64::r0);
      }	
    }

    return Result;
  }
  
  case ISD::CopyFromReg: {
    if (Result == 1)
        Result = ExprMap[N.getValue(0)] = 
	  MakeReg(N.getValue(0).getValueType());
                                                                                
      SDOperand Chain   = N.getOperand(0);

      Select(Chain);
      unsigned r = dyn_cast<RegSDNode>(Node)->getReg();

      if(N.getValueType() == MVT::i1) // if a bool, we use pseudocode
	BuildMI(BB, IA64::PCMPEQUNC, 3, Result)
	  .addReg(IA64::r0).addReg(IA64::r0).addReg(r);
                            // (r) Result =cmp.eq.unc(r0,r0)
      else
	BuildMI(BB, IA64::MOV, 1, Result).addReg(r); // otherwise MOV
      return Result;
  }

  case ISD::CALL: {
      Select(N.getOperand(0));

      // The chain for this call is now lowered.
      ExprMap.insert(std::make_pair(N.getValue(Node->getNumValues()-1), 1));
      
      //grab the arguments
      std::vector<unsigned> argvregs;

      for(int i = 2, e = Node->getNumOperands(); i < e; ++i)
	argvregs.push_back(SelectExpr(N.getOperand(i)));
      
      // see section 8.5.8 of "Itanium Software Conventions and 
      // Runtime Architecture Guide to see some examples of what's going
      // on here. (in short: int args get mapped 1:1 'slot-wise' to out0->out7,
      // while FP args get mapped to F8->F15 as needed)

      unsigned used_FPArgs=0; // how many FP Args have been used so far?
      
      // in reg args
      for(int i = 0, e = std::min(8, (int)argvregs.size()); i < e; ++i)
      {
	unsigned intArgs[] = {IA64::out0, IA64::out1, IA64::out2, IA64::out3, 
			      IA64::out4, IA64::out5, IA64::out6, IA64::out7 };
	unsigned FPArgs[] = {IA64::F8, IA64::F9, IA64::F10, IA64::F11,
	                     IA64::F12, IA64::F13, IA64::F14, IA64::F15 };

	switch(N.getOperand(i+2).getValueType())
	{
	  default:  // XXX do we need to support MVT::i1 here?
	    Node->dump();
	    N.getOperand(i).Val->dump();
	    std::cerr << "Type for " << i << " is: " << 
	      N.getOperand(i+2).getValueType() << std::endl;
	    assert(0 && "Unknown value type for call");
	  case MVT::i64:
	    BuildMI(BB, IA64::MOV, 1, intArgs[i]).addReg(argvregs[i]);
	    break;
	  case MVT::f64:
	    BuildMI(BB, IA64::FMOV, 1, FPArgs[used_FPArgs++])
	      .addReg(argvregs[i]);
	    BuildMI(BB, IA64::GETFD, 1, intArgs[i]).addReg(argvregs[i]);
	    break;
	  }
      }

      //in mem args
      for (int i = 8, e = argvregs.size(); i < e; ++i)
      {
	unsigned tempAddr = MakeReg(MVT::i64);
	
        switch(N.getOperand(i+2).getValueType()) {
        default: 
          Node->dump(); 
          N.getOperand(i).Val->dump();
          std::cerr << "Type for " << i << " is: " << 
            N.getOperand(i+2).getValueType() << "\n";
          assert(0 && "Unknown value type for call");
        case MVT::i1: // FIXME?
        case MVT::i8:
        case MVT::i16:
        case MVT::i32:
        case MVT::i64:
	  BuildMI(BB, IA64::ADDIMM22, 2, tempAddr)
	    .addReg(IA64::r12).addImm(16 + (i - 8) * 8); // r12 is SP
	  BuildMI(BB, IA64::ST8, 2).addReg(tempAddr).addReg(argvregs[i]);
          break;
        case MVT::f32:
        case MVT::f64:
          BuildMI(BB, IA64::ADDIMM22, 2, tempAddr)
	    .addReg(IA64::r12).addImm(16 + (i - 8) * 8); // r12 is SP
	  BuildMI(BB, IA64::STF8, 2).addReg(tempAddr).addReg(argvregs[i]);
          break;
        }
      }
    //build the right kind of call
    if (GlobalAddressSDNode *GASD =
               dyn_cast<GlobalAddressSDNode>(N.getOperand(1))) 
      {
	BuildMI(BB, IA64::BRCALL, 1).addGlobalAddress(GASD->getGlobal(),true);
	IA64Lowering.restoreGP_SP_RP(BB);
      }

    else if (ExternalSymbolSDNode *ESSDN =
	     dyn_cast<ExternalSymbolSDNode>(N.getOperand(1))) 
      {
	BuildMI(BB, IA64::BRCALL, 0)
	  .addExternalSymbol(ESSDN->getSymbol(), true);
	IA64Lowering.restoreGP_SP_RP(BB);
      }
    else {
      // no need to restore GP as we are doing an indirect call
      Tmp1 = SelectExpr(N.getOperand(1));
      // b6 is a scratch branch register, we load the target:
      BuildMI(BB, IA64::MOV, 1, IA64::B6).addReg(Tmp1);
      // and then jump: (well, call)
      BuildMI(BB, IA64::BRCALL, 1).addReg(IA64::B6);
      IA64Lowering.restoreGP_SP_RP(BB);
  }

    switch (Node->getValueType(0)) {
    default: assert(0 && "Unknown value type for call result!");
    case MVT::Other: return 1;
    case MVT::i1:
      BuildMI(BB, IA64::CMPNE, 2, Result)
	.addReg(IA64::r8).addReg(IA64::r0);
      break;
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
      BuildMI(BB, IA64::MOV, 1, Result).addReg(IA64::r8);
      break;
    case MVT::f64:
      BuildMI(BB, IA64::FMOV, 1, Result).addReg(IA64::F8);
      break;
    }
    return Result+N.ResNo;
  }

  } // <- uhhh XXX 
  return 0;
}

void ISel::Select(SDOperand N) {
  unsigned Tmp1, Tmp2, Opc;
  unsigned opcode = N.getOpcode();

  // FIXME: Disable for our current expansion model!
  if (/*!N->hasOneUse() &&*/ !LoweredTokens.insert(N).second)
    return;  // Already selected.

  SDNode *Node = N.Val;

  switch (Node->getOpcode()) {
  default:
    Node->dump(); std::cerr << "\n";
    assert(0 && "Node not handled yet!");

  case ISD::EntryToken: return;  // Noop
  
  case ISD::TokenFactor: {
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Select(Node->getOperand(i));
    return;
  }

  case ISD::CopyToReg: {
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));   
    Tmp2 = cast<RegSDNode>(N)->getReg();
    
    if (Tmp1 != Tmp2) {
      if(N.getValueType() == MVT::i1) // if a bool, we use pseudocode
	BuildMI(BB, IA64::PCMPEQUNC, 3, Tmp2)
	  .addReg(IA64::r0).addReg(IA64::r0).addReg(Tmp1);
                                   // (Tmp1) Tmp2 = cmp.eq.unc(r0,r0)
      else
	BuildMI(BB, IA64::MOV, 1, Tmp2).addReg(Tmp1);
                      // XXX is this the right way 'round? ;)
    }
    return;
  }
  
  case ISD::RET: {

  /* what the heck is going on here:

<_sabre_> ret with two operands is obvious: chain and value
<camel_> yep
<_sabre_> ret with 3 values happens when 'expansion' occurs
<_sabre_> e.g. i64 gets split into 2x i32
<camel_> oh right
<_sabre_> you don't have this case on ia64
<camel_> yep
<_sabre_> so the two returned values go into EAX/EDX on ia32
<camel_> ahhh *memories*
<_sabre_> :)
<camel_> ok, thanks :)
<_sabre_> so yeah, everything that has a side effect takes a 'token chain'
<_sabre_> this is the first operand always
<_sabre_> these operand often define chains, they are the last operand
<_sabre_> they are printed as 'ch' if you do DAG.dump()
  */
  
    switch (N.getNumOperands()) {
    default:
      assert(0 && "Unknown return instruction!");
    case 2:
        Select(N.getOperand(0));
        Tmp1 = SelectExpr(N.getOperand(1));
      switch (N.getOperand(1).getValueType()) {
      default: assert(0 && "All other types should have been promoted!!");
	       // FIXME: do I need to add support for bools here?
	       // (return '0' or '1' r8, basically...)
      case MVT::i64:
	BuildMI(BB, IA64::MOV, 1, IA64::r8).addReg(Tmp1);
	break;
      case MVT::f64:
	BuildMI(BB, IA64::FMOV, 1, IA64::F8).addReg(Tmp1);
      }
      break;
    case 1:
      Select(N.getOperand(0));
      break;
    }
    // before returning, restore the ar.pfs register (set by the 'alloc' up top)
    BuildMI(BB, IA64::MOV, 1).addReg(IA64::AR_PFS).addReg(IA64Lowering.VirtGPR);
    BuildMI(BB, IA64::RET, 0); // and then just emit a 'ret' instruction
    return;
  }
  
  case ISD::BR: {
    Select(N.getOperand(0));
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(1))->getBasicBlock();
    BuildMI(BB, IA64::BRLCOND_NOTCALL, 1).addReg(IA64::p0).addMBB(Dest);
    // XXX HACK! we do _not_ need long branches all the time
    return;
  }

  case ISD::ImplicitDef: {
    Select(N.getOperand(0));
    BuildMI(BB, IA64::IDEF, 0, cast<RegSDNode>(N)->getReg());
    return;
  }

  case ISD::BRCOND: {
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(2))->getBasicBlock();

    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    BuildMI(BB, IA64::BRLCOND_NOTCALL, 1).addReg(Tmp1).addMBB(Dest);
    // XXX HACK! we do _not_ need long branches all the time
    return;
  }
  
  case ISD::EXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::LOAD:
  case ISD::CALL:
  case ISD::CopyFromReg:
  case ISD::DYNAMIC_STACKALLOC:
    SelectExpr(N);
    return;

  case ISD::TRUNCSTORE:
  case ISD::STORE: {
      Select(N.getOperand(0));
      Tmp1 = SelectExpr(N.getOperand(1)); // value

      bool isBool=false;
     
      if(opcode == ISD::STORE) {
	switch (N.getOperand(1).getValueType()) {
	  default: assert(0 && "Cannot store this type!");
	  case MVT::i1:  Opc = IA64::ST1; isBool=true; break;
	      // FIXME?: for now, we treat bool loads the same as i8 stores */
	  case MVT::i8:  Opc = IA64::ST1; break;
	  case MVT::i16: Opc = IA64::ST2; break;
	  case MVT::i32: Opc = IA64::ST4; break;
	  case MVT::i64: Opc = IA64::ST8; break;
			 
	  case MVT::f32: Opc = IA64::STF4; break;
	  case MVT::f64: Opc = IA64::STF8; break;
	}
      } else { // truncstore
	switch(cast<MVTSDNode>(Node)->getExtraValueType()) {
	  default: assert(0 && "unknown type in truncstore");
	  case MVT::i1: Opc = IA64::ST1; isBool=true; break;
			//FIXME: DAG does not promote this load?
	  case MVT::i8: Opc = IA64::ST1; break;
	  case MVT::i16: Opc = IA64::ST2; break;
	  case MVT::i32: Opc = IA64::ST4; break;
	  case MVT::f32: Opc = IA64::STF4; break; 
	}
      }

      if(N.getOperand(2).getOpcode() == ISD::GlobalAddress) {
	unsigned dummy = MakeReg(MVT::i64);
	unsigned dummy2 = MakeReg(MVT::i64);
	BuildMI(BB, IA64::ADD, 2, dummy)
	  .addGlobalAddress(cast<GlobalAddressSDNode>
	      (N.getOperand(2))->getGlobal()).addReg(IA64::r1);
	BuildMI(BB, IA64::LD8, 1, dummy2).addReg(dummy);
      
	if(!isBool)
	  BuildMI(BB, Opc, 2).addReg(dummy2).addReg(Tmp1);
	else { // we are storing a bool, so emit a little pseudocode
	       // to store a predicate register as one byte
	  assert(Opc==IA64::ST1);
	  unsigned dummy3 = MakeReg(MVT::i64);
	  unsigned dummy4 = MakeReg(MVT::i64);
	  BuildMI(BB, IA64::MOV, 1, dummy3).addReg(IA64::r0);
	  BuildMI(BB, IA64::CADDIMM22, 3, dummy4)
	    .addReg(dummy3).addImm(1).addReg(Tmp1); // if(Tmp1) dummy=0+1;
	  BuildMI(BB, Opc, 2).addReg(dummy2).addReg(dummy4);
	}
      } else if(N.getOperand(2).getOpcode() == ISD::FrameIndex) {

	// FIXME? (what about bools?)
	
	unsigned dummy = MakeReg(MVT::i64);
	BuildMI(BB, IA64::MOV, 1, dummy)
	  .addFrameIndex(cast<FrameIndexSDNode>(N.getOperand(2))->getIndex());
	BuildMI(BB, Opc, 2).addReg(dummy).addReg(Tmp1);
      } else { // otherwise
	Tmp2 = SelectExpr(N.getOperand(2)); //address
	if(!isBool) 
	  BuildMI(BB, Opc, 2).addReg(Tmp2).addReg(Tmp1);
	else { // we are storing a bool, so emit a little pseudocode
	       // to store a predicate register as one byte
	  assert(Opc==IA64::ST1);
	  unsigned dummy3 = MakeReg(MVT::i64);
	  unsigned dummy4 = MakeReg(MVT::i64);
	  BuildMI(BB, IA64::MOV, 1, dummy3).addReg(IA64::r0);
	  BuildMI(BB, IA64::CADDIMM22, 3, dummy4)
	    .addReg(dummy3).addImm(1).addReg(Tmp1); // if(Tmp1) dummy=0+1;
	  BuildMI(BB, Opc, 2).addReg(Tmp2).addReg(dummy4);
	}
      }
    return;
  }
  
  case ISD::ADJCALLSTACKDOWN:
  case ISD::ADJCALLSTACKUP: {
    Select(N.getOperand(0));
    Tmp1 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
   
    Opc = N.getOpcode() == ISD::ADJCALLSTACKDOWN ? IA64::ADJUSTCALLSTACKDOWN :
                                                   IA64::ADJUSTCALLSTACKUP;
    BuildMI(BB, Opc, 1).addImm(Tmp1);
    return;
  }

    return;
  }
  assert(0 && "GAME OVER. INSERT COIN?");
}


/// createIA64PatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *llvm::createIA64PatternInstructionSelector(TargetMachine &TM) {
  return new ISel(TM);  
}


