//===- AlphaISelPattern.cpp - A pattern matching inst selector for Alpha -===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for Alpha.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "AlphaRegisterInfo.h"
#include "llvm/Constants.h"                   // FIXME: REMOVE
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
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
using namespace llvm;

//===----------------------------------------------------------------------===//
//  AlphaTargetLowering - Alpha Implementation of the TargetLowering interface
namespace {
  class AlphaTargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
    unsigned GP; //GOT vreg
  public:
    AlphaTargetLowering(TargetMachine &TM) : TargetLowering(TM) {
      // Set up the TargetLowering object.
      addRegisterClass(MVT::i64, Alpha::GPRCRegisterClass);
      addRegisterClass(MVT::f64, Alpha::FPRCRegisterClass);

      setOperationAction(ISD::EXTLOAD          , MVT::i1   , Expand);

      setOperationAction(ISD::ZEXTLOAD         , MVT::i1   , Expand);
      setOperationAction(ISD::ZEXTLOAD         , MVT::i32  , Expand);

      setOperationAction(ISD::SEXTLOAD         , MVT::i1   , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i8   , Expand);
      setOperationAction(ISD::SEXTLOAD         , MVT::i16  , Expand);

      setOperationAction(ISD::ZERO_EXTEND_INREG, MVT::i1, Expand);
      setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1, Expand);


      computeRegisterProperties();
      
      //      addLegalFPImmediate(+0.0); // FLD0
      //      addLegalFPImmediate(+1.0); // FLD1
      //      addLegalFPImmediate(-0.0); // FLD0/FCHS
      //      addLegalFPImmediate(-1.0); // FLD1/FCHS
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

    void restoreGP(MachineBasicBlock* BB)
    {
      BuildMI(BB, Alpha::BIS, 2, Alpha::R29).addReg(GP).addReg(GP);
    }
  };
}

//http://www.cs.arizona.edu/computer.help/policy/DIGITAL_unix/AA-PY8AC-TET1_html/callCH3.html#BLOCK21

//For now, just use variable size stack frame format

//In a standard call, the first six items are passed in registers $16
//- $21 and/or registers $f16 - $f21. (See Section 4.1.2 for details
//of argument-to-register correspondence.) The remaining items are
//collected in a memory argument list that is a naturally aligned
//array of quadwords. In a standard call, this list, if present, must
//be passed at 0(SP).
//7 ... n  	  	  	0(SP) ... (n-7)*8(SP)

std::vector<SDOperand>
AlphaTargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) 
{
  std::vector<SDOperand> ArgValues;
  
  // //#define FP    $15
  // //#define RA    $26
  // //#define PV    $27
  // //#define GP    $29
  // //#define SP    $30
  
  //  assert(0 && "TODO");
  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo *MFI = MF.getFrameInfo();

  GP = MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64));
  MachineBasicBlock& BB = MF.front();

  //Handle the return address
  //BuildMI(&BB, Alpha::IDEF, 0, Alpha::R26);

  unsigned args[] = {Alpha::R16, Alpha::R17, Alpha::R18, 
		     Alpha::R19, Alpha::R20, Alpha::R21};
  std::vector<unsigned> argVreg;

  int count = 0;
  for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
    {
      ++count;
      assert(count <= 6 && "More than 6 args not supported");
      assert(getValueType(I->getType()) != MVT::f64 && "No floats yet");
      BuildMI(&BB, Alpha::IDEF, 0, args[count - 1]);
      argVreg.push_back(MF.getSSARegMap()->createVirtualRegister(getRegClassFor(MVT::i64)));
    }

  BuildMI(&BB, Alpha::IDEF, 0, Alpha::R29);
  BuildMI(&BB, Alpha::BIS, 2, GP).addReg(Alpha::R29).addReg(Alpha::R29);
  count = 0;
  for (Function::aiterator I = F.abegin(), E = F.aend(); I != E; ++I)
    {
      BuildMI(&BB, Alpha::BIS, 2, argVreg[count]).addReg(args[count]).addReg(args[count]);
      
      SDOperand argt, newroot;
      switch (getValueType(I->getType()))
	{
	case MVT::i64:
	  argt = newroot = DAG.getCopyFromReg(argVreg[count], MVT::i64, DAG.getRoot());
	  break;
	case MVT::i32:
	  argt = newroot = DAG.getCopyFromReg(argVreg[count], MVT::i32, DAG.getRoot());
	  break;
	default:
	  newroot = DAG.getCopyFromReg(argVreg[count], MVT::i64, DAG.getRoot());
	  argt =  DAG.getNode(ISD::TRUNCATE, getValueType(I->getType()), newroot);
	}
      DAG.setRoot(newroot.getValue(1));
      ArgValues.push_back(argt);
      ++count;
    }
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
AlphaTargetLowering::LowerCallTo(SDOperand Chain,
				 const Type *RetTy, SDOperand Callee,
				 ArgListTy &Args, SelectionDAG &DAG) {
  int NumBytes = 0;
  Chain = DAG.getNode(ISD::ADJCALLSTACKDOWN, MVT::Other, Chain,
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
	break;
      }
      args_to_use.push_back(Args[i].first);
    }

  std::vector<MVT::ValueType> RetVals;
  MVT::ValueType RetTyVT = getValueType(RetTy);
  if (RetTyVT != MVT::isVoid)
    RetVals.push_back(RetTyVT);
  RetVals.push_back(MVT::Other);

  SDOperand TheCall = SDOperand(DAG.getCall(RetVals, Chain, Callee, args_to_use), 0);
  Chain = TheCall.getValue(RetTyVT != MVT::isVoid);
  Chain = DAG.getNode(ISD::ADJCALLSTACKUP, MVT::Other, Chain,
                      DAG.getConstant(NumBytes, getPointerTy()));
   return std::make_pair(TheCall, Chain);
}

std::pair<SDOperand, SDOperand>
AlphaTargetLowering::LowerVAStart(SDOperand Chain, SelectionDAG &DAG) {
  //vastart just returns the address of the VarArgsFrameIndex slot.
  return std::make_pair(DAG.getFrameIndex(VarArgsFrameIndex, MVT::i64), Chain);
}

std::pair<SDOperand,SDOperand> AlphaTargetLowering::
LowerVAArgNext(bool isVANext, SDOperand Chain, SDOperand VAList,
                const Type *ArgTy, SelectionDAG &DAG) {
  abort();
}
               

std::pair<SDOperand, SDOperand> AlphaTargetLowering::
LowerFrameReturnAddress(bool isFrameAddress, SDOperand Chain, unsigned Depth,
                        SelectionDAG &DAG) {
  abort();
}





namespace {

  //===--------------------------------------------------------------------===//
  /// ISel - Alpha specific code to select Alpha machine instructions for
  /// SelectionDAG operations.
  ///
  class ISel : public SelectionDAGISel {

    /// AlphaLowering - This object fully describes how to lower LLVM code to an
    /// Alpha-specific SelectionDAG.
    AlphaTargetLowering AlphaLowering;


    /// ExprMap - As shared expressions are codegen'd, we keep track of which
    /// vreg the value is produced in, so we only emit one copy of each compiled
    /// tree.
    std::map<SDOperand, unsigned> ExprMap;
    std::set<SDOperand> LoweredTokens;

  public:
    ISel(TargetMachine &TM) : SelectionDAGISel(AlphaLowering), AlphaLowering(TM) {
    }

    /// InstructionSelectBasicBlock - This callback is invoked by
    /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
    virtual void InstructionSelectBasicBlock(SelectionDAG &DAG) {
      // Codegen the basic block.
      Select(DAG.getRoot());

      // Clear state used for selection.
      ExprMap.clear();
      LoweredTokens.clear();
    }

    unsigned SelectExpr(SDOperand N);
    void Select(SDOperand N);
  };
}

unsigned ISel::SelectExpr(SDOperand N) {
  unsigned Result;
  unsigned Tmp1, Tmp2, Tmp3;
  unsigned Opc = 0;

  SDNode *Node = N.Val;

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
 
  case ISD::FrameIndex:
    Tmp1 = cast<FrameIndexSDNode>(N)->getIndex();
    BuildMI(BB, Alpha::LDA, 2, Result).addImm(Tmp1 * 8).addReg(Alpha::R30);
    return Result;
  
  case ISD::EXTLOAD:
    // Make sure we generate both values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());
    
    Select(Node->getOperand(0)); // chain
    Tmp1 = SelectExpr(Node->getOperand(1));
    
    switch(Node->getValueType(0)) {
    default: assert(0 && "Unknown type to sign extend to.");
    case MVT::i64:
      switch (cast<MVTSDNode>(Node)->getExtraValueType()) {
      default:
	std::cerr << cast<MVTSDNode>(Node)->getExtraValueType() 
		  << "(i1 is " << MVT::i1 
		  << " i8 is " << MVT::i8
		  << " i16 is " << MVT::i16
		  << " i32 is " << MVT::i32
		  << " i64 is " << MVT::i64
		  << ")\n";
        assert(0 && "Bad extend load!");
      case MVT::i64:
	BuildMI(BB, Alpha::LDQ, 2, Result).addImm(0).addReg(Tmp1);
	break;
      case MVT::i32:
	BuildMI(BB, Alpha::LDL, 2, Result).addImm(0).addReg(Tmp1);
        break;
      case MVT::i16:
	BuildMI(BB, Alpha::LDWU, 2, Result).addImm(0).addReg(Tmp1);
        break;
      case MVT::i8:
      case MVT::i1: //FIXME: DAG does not expand i8??
	BuildMI(BB, Alpha::LDBU, 2, Result).addImm(0).addReg(Tmp1);
        break;
      }
      break;
    }
    return Result;

  case ISD::SEXTLOAD:
    // Make sure we generate both values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());
    
    Select(Node->getOperand(0)); // chain
    Tmp1 = SelectExpr(Node->getOperand(1));
    switch(Node->getValueType(0)) {
    default: assert(0 && "Unknown type to sign extend to.");
    case MVT::i64:
      switch (cast<MVTSDNode>(Node)->getExtraValueType()) {
      default:
        assert(0 && "Bad sign extend!");
      case MVT::i32:
	BuildMI(BB, Alpha::LDL, 2, Result).addImm(0).addReg(Tmp1);
        break;
//       case MVT::i16:
// 	BuildMI(BB, Alpha::LDW, 2, Result).addImm(0).addReg(Tmp1);
//         break;
//       case MVT::i8:
// 	BuildMI(BB, Alpha::LDB, 2, Result).addImm(0).addReg(Tmp1);
//         break;
      }
      break;
    }
    return Result;

  case ISD::ZEXTLOAD:
    // Make sure we generate both values.
    if (Result != 1)
      ExprMap[N.getValue(1)] = 1;   // Generate the token
    else
      Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());
    
    Select(Node->getOperand(0)); // chain
    Tmp1 = SelectExpr(Node->getOperand(1));
    switch(Node->getValueType(0)) {
    default: assert(0 && "Unknown type to zero extend to.");
    case MVT::i64:
      switch (cast<MVTSDNode>(Node)->getExtraValueType()) {
      default:
        assert(0 && "Bad sign extend!");
      case MVT::i16:
	BuildMI(BB, Alpha::LDWU, 2, Result).addImm(0).addReg(Tmp1);
        break;
      case MVT::i8:
	BuildMI(BB, Alpha::LDBU, 2, Result).addImm(0).addReg(Tmp1);
        break;
      }
      break;
    }
    return Result;


  case ISD::GlobalAddress:
    AlphaLowering.restoreGP(BB);
    BuildMI(BB, Alpha::LOAD_ADDR, 1, Result)
      .addGlobalAddress(cast<GlobalAddressSDNode>(N)->getGlobal());
    return Result;

  case ISD::CALL:
    {
      Select(N.getOperand(0));

      // The chain for this call is now lowered.
      ExprMap.insert(std::make_pair(N.getValue(Node->getNumValues()-1), 1));
      
      //grab the arguments
      std::vector<unsigned> argvregs;
      assert(Node->getNumOperands() < 8 && "Only 6 args supported");
      for(int i = 2, e = Node->getNumOperands(); i < e; ++i)
      {
	argvregs.push_back(SelectExpr(N.getOperand(i)));
      }
      for(int i = 0, e = argvregs.size(); i < e; ++i)
      {
	unsigned args[] = {Alpha::R16, Alpha::R17, Alpha::R18, 
			   Alpha::R19, Alpha::R20, Alpha::R21};
	
	BuildMI(BB, Alpha::BIS, 2, args[i]).addReg(argvregs[i]).addReg(argvregs[i]);
      }

    //build the right kind of call
    if (GlobalAddressSDNode *GASD =
               dyn_cast<GlobalAddressSDNode>(N.getOperand(1))) 
      {
	Select(N.getOperand(0));
	AlphaLowering.restoreGP(BB);
	BuildMI(BB, Alpha::CALL, 1).addGlobalAddress(GASD->getGlobal(),true);
      }
    else if (ExternalSymbolSDNode *ESSDN =
	     dyn_cast<ExternalSymbolSDNode>(N.getOperand(1))) 
      {
	Select(N.getOperand(0));
	AlphaLowering.restoreGP(BB);
	BuildMI(BB, Alpha::CALL, 0).addExternalSymbol(ESSDN->getSymbol(), true);
      } 
    else {
      Select(N.getOperand(0));
      Tmp1 = SelectExpr(N.getOperand(1));
      BuildMI(BB, Alpha::CALL, 1).addReg(Tmp1);
      AlphaLowering.restoreGP(BB);
    }

    //push the result into a virtual register
    //    if (Result != 1)
    //      BuildMI(BB, Alpha::BIS, 2, Result).addReg(Alpha::R0).addReg(Alpha::R0);

    switch (Node->getValueType(0)) {
    default: assert(0 && "Unknown value type for call result!");
    case MVT::Other: return 1;
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32:
    case MVT::i64:
      BuildMI(BB, Alpha::BIS, 2, Result).addReg(Alpha::R0).addReg(Alpha::R0);
    break;
    }
    return Result+N.ResNo;
    }    
  
  case ISD::SIGN_EXTEND:
  case ISD::SIGN_EXTEND_INREG:
    {
      Tmp1 = SelectExpr(N.getOperand(0));
      MVTSDNode* MVN = dyn_cast<MVTSDNode>(Node);
      std::cerr << "SrcT: " << MVN->getExtraValueType() << "\n";
      switch(MVN->getExtraValueType())
	{
	default:
	  assert(0 && "Sign Extend InReg not there yet");
	  break;
	case MVT::i32:
	  {
	    BuildMI(BB, Alpha::ADDLi, 2, Result).addReg(Tmp1).addImm(0);
	    break;
	  }
	case MVT::i16:
	  BuildMI(BB, Alpha::SEXTW, 1, Result).addReg(Tmp1);
	  break;
	case MVT::i8:
	  BuildMI(BB, Alpha::SEXTB, 1, Result).addReg(Tmp1);
	  break;
	}
      return Result;
    }
  case ISD::ZERO_EXTEND_INREG:
    {
      Tmp1 = SelectExpr(N.getOperand(0));
      MVTSDNode* MVN = dyn_cast<MVTSDNode>(Node);
      std::cerr << "SrcT: " << MVN->getExtraValueType() << "\n";
      switch(MVN->getExtraValueType())
	{
	default:
	  assert(0 && "Zero Extend InReg not there yet");
	  break;
	case MVT::i32: Tmp2 = 0xf0; break;
	case MVT::i16: Tmp2 = 0xfc; break;
	case MVT::i8: Tmp2 = 0xfe; break;
	}
      BuildMI(BB, Alpha::ZAPi, 2, Result).addReg(Tmp1).addImm(Tmp2);
     return Result;
    }
    
  case ISD::SETCC:
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    if (SetCCSDNode *SetCC = dyn_cast<SetCCSDNode>(Node)) {
      if (MVT::isInteger(SetCC->getOperand(0).getValueType())) {
	switch (SetCC->getCondition()) {
	default: assert(0 && "Unknown integer comparison!");
	case ISD::SETEQ:
	  BuildMI(BB, Alpha::CMPEQ, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETGT:
	  BuildMI(BB, Alpha::CMPLT, 2, Result).addReg(Tmp2).addReg(Tmp1);
	  break;
	case ISD::SETGE:
	  BuildMI(BB, Alpha::CMPLE, 2, Result).addReg(Tmp2).addReg(Tmp1);
	  break;
	case ISD::SETLT:
	  BuildMI(BB, Alpha::CMPLT, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETLE:
	  BuildMI(BB, Alpha::CMPLE, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETNE:
	  {
	    unsigned Tmp3 = MakeReg(MVT::i64);
	    BuildMI(BB, Alpha::CMPEQ, 2, Tmp3).addReg(Tmp1).addReg(Tmp2);
	    BuildMI(BB, Alpha::CMPEQ, 2, Result).addReg(Tmp3).addReg(Alpha::R31);
	    break;
	  }
	case ISD::SETULT:
	  BuildMI(BB, Alpha::CMPULT, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETUGT:
	  BuildMI(BB, Alpha::CMPULT, 2, Result).addReg(Tmp2).addReg(Tmp1);
	  break;
	case ISD::SETULE:
	  BuildMI(BB, Alpha::CMPULE, 2, Result).addReg(Tmp1).addReg(Tmp2);
	  break;
	case ISD::SETUGE:
	  BuildMI(BB, Alpha::CMPULE, 2, Result).addReg(Tmp2).addReg(Tmp1);
	  break;
	}
      }
      else
	assert(0 && "only integer");
    }
    else
      assert(0 && "Not a setcc in setcc");

    return Result;

  case ISD::CopyFromReg:
    {
      if (Result == 1)
	Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());
      
      SDOperand Chain   = N.getOperand(0);

      Select(Chain);
      unsigned r = dyn_cast<RegSDNode>(Node)->getReg();
      //std::cerr << "CopyFromReg " << Result << " = " << r << "\n";
      BuildMI(BB, Alpha::BIS, 2, Result).addReg(r).addReg(r);
      return Result;
    }

    //Most of the plain arithmetic and logic share the same form, and the same 
    //constant immediate test
  case ISD::AND:
  case ISD::OR:
  case ISD::XOR:
  case ISD::SHL:
  case ISD::SRL:
  case ISD::MUL:
    if(N.getOperand(1).getOpcode() == ISD::Constant &&
       cast<ConstantSDNode>(N.getOperand(1))->getValue() >= 0 &&
       cast<ConstantSDNode>(N.getOperand(1))->getValue() <= 255)
      {
	switch(N.getOpcode()) {
	case ISD::AND: Opc = Alpha::ANDi; break;
	case ISD::OR:  Opc = Alpha::BISi; break;
	case ISD::XOR: Opc = Alpha::XORi; break;
	case ISD::SHL: Opc = Alpha::SLi; break;
	case ISD::SRL: Opc = Alpha::SRLi; break;
	case ISD::SRA: Opc = Alpha::SRAi; break;
	case ISD::MUL: Opc = Alpha::MULQi; break;
	};
	Tmp1 = SelectExpr(N.getOperand(0));
	Tmp2 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
	BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
      }
    else
      {
	switch(N.getOpcode()) {
	case ISD::AND: Opc = Alpha::AND; break;
	case ISD::OR:  Opc = Alpha::BIS; break;
	case ISD::XOR: Opc = Alpha::XOR; break;
	case ISD::SHL: Opc = Alpha::SL; break;
	case ISD::SRL: Opc = Alpha::SRL; break;
	case ISD::SRA: Opc = Alpha::SRA; break;
	case ISD::MUL: Opc = Alpha::MULQ; break;
	};
	Tmp1 = SelectExpr(N.getOperand(0));
	Tmp2 = SelectExpr(N.getOperand(1));
	BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
      }
    return Result;

  case ISD::ADD:
  case ISD::SUB:
    {
      bool isAdd = N.getOpcode() == ISD::ADD;

      //FIXME: first check for Scaled Adds and Subs!
      if(N.getOperand(1).getOpcode() == ISD::Constant &&
	 cast<ConstantSDNode>(N.getOperand(1))->getValue() >= 0 &&
	 cast<ConstantSDNode>(N.getOperand(1))->getValue() <= 255)
	{ //Normal imm add/sub
	  Opc = isAdd ? Alpha::ADDQi : Alpha::SUBQi;
	  Tmp1 = SelectExpr(N.getOperand(0));
	  Tmp2 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
	  BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addImm(Tmp2);
	}
      else if(N.getOperand(1).getOpcode() == ISD::Constant &&
	      cast<ConstantSDNode>(N.getOperand(1))->getValue() >= 0 &&
	 cast<ConstantSDNode>(N.getOperand(1))->getValue() <= 32767)
	{ //LDA  //FIXME: expand the above condition a bit
	  Tmp1 = SelectExpr(N.getOperand(0));
	  Tmp2 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
	  if (!isAdd)
	    Tmp2 = -Tmp2;
	  BuildMI(BB, Alpha::LDA, 2, Result).addImm(Tmp2).addReg(Tmp1);
	}
      else
	{ //Normal add/sub
	  Opc = isAdd ? Alpha::ADDQ : Alpha::SUBQ;
	  Tmp1 = SelectExpr(N.getOperand(0));
	  Tmp2 = SelectExpr(N.getOperand(1));
	  BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);

	}
	  return Result;
      }

  case ISD::UREM:
  case ISD::SREM:
  case ISD::SDIV:
  case ISD::UDIV:
    //FIXME: alpha really doesn't support any of these operations, 
    // the ops are expanded into special library calls with
    // special calling conventions
    switch(N.getOpcode()) {
    case ISD::UREM: Opc = Alpha::REMQU; break;
    case ISD::SREM: Opc = Alpha::REMQ; break;
    case ISD::UDIV: Opc = Alpha::DIVQU; break;
    case ISD::SDIV: Opc = Alpha::DIVQ; break;
    };
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  case ISD::SELECT:
    {
      Tmp2 = SelectExpr(N.getOperand(1)); //Use if TRUE
      Tmp3 = SelectExpr(N.getOperand(2)); //Use if FALSE
      Tmp1 = SelectExpr(N.getOperand(0)); //Cond
      // Get the condition into the zero flag.
      unsigned dummy = MakeReg(MVT::i64);
      BuildMI(BB, Alpha::BIS, 2, dummy).addReg(Tmp3).addReg(Tmp3);
      BuildMI(BB, Alpha::CMOVEQ, 2, Result).addReg(Tmp2).addReg(Tmp1);
      return Result;
    }

  case ISD::Constant:
    {
      long val = cast<ConstantSDNode>(N)->getValue();
      BuildMI(BB, Alpha::LOAD_IMM, 1, Result).addImm(val);
      return Result;
    }

  case ISD::LOAD: 
    {
      // Make sure we generate both values.
      if (Result != 1)
	ExprMap[N.getValue(1)] = 1;   // Generate the token
      else
	Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());
      
      SDOperand Chain   = N.getOperand(0);
      SDOperand Address = N.getOperand(1);

      if (Address.getOpcode() == ISD::GlobalAddress)
	{
	  Select(Chain);
	  AlphaLowering.restoreGP(BB);
	  BuildMI(BB, Alpha::LOAD, 1, Result).addGlobalAddress(cast<GlobalAddressSDNode>(Address)->getGlobal());
	}
      else
	{
	  Select(Chain);
	  Tmp2 = SelectExpr(Address);
	  BuildMI(BB, Alpha::LDQ, 2, Result).addImm(0).addReg(Tmp2);
	}
      return Result;
    }
  }

  return 0;
}

void ISel::Select(SDOperand N) {
  unsigned Tmp1, Tmp2, Opc;

  // FIXME: Disable for our current expansion model!
  if (/*!N->hasOneUse() &&*/ !LoweredTokens.insert(N).second)
    return;  // Already selected.

  SDNode *Node = N.Val;

  switch (N.getOpcode()) {

  default:
    Node->dump(); std::cerr << "\n";
    assert(0 && "Node not handled yet!");

  case ISD::BRCOND: {
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(2))->getBasicBlock();

    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    BuildMI(BB, Alpha::BNE, 2).addReg(Tmp1).addMBB(Dest);
    return;
  }

  case ISD::BR: {
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(1))->getBasicBlock();

    Select(N.getOperand(0));
    BuildMI(BB, Alpha::BR, 1, Alpha::R31).addMBB(Dest);
    return;
  }

  case ISD::ImplicitDef:
    Select(N.getOperand(0));
    BuildMI(BB, Alpha::IDEF, 0, cast<RegSDNode>(N)->getReg());
    return;
    
  case ISD::EntryToken: return;  // Noop

  case ISD::TokenFactor:
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Select(Node->getOperand(i));
    
    //N.Val->dump(); std::cerr << "\n";
    //assert(0 && "Node not handled yet!");
    
    return;

  case ISD::CopyToReg:
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    Tmp2 = cast<RegSDNode>(N)->getReg();
    
    if (Tmp1 != Tmp2) {
      BuildMI(BB, Alpha::BIS, 2, Tmp2).addReg(Tmp1).addReg(Tmp1);
    }
    return;

   case ISD::RET:
     switch (N.getNumOperands()) {
     default:
       std::cerr << N.getNumOperands() << "\n";
       for (unsigned i = 0; i < N.getNumOperands(); ++i)
	 std::cerr << N.getOperand(i).getValueType() << "\n";
       assert(0 && "Unknown return instruction!");
     case 2:
       Select(N.getOperand(0));
       Tmp1 = SelectExpr(N.getOperand(1));
       switch (N.getOperand(1).getValueType()) {
       default: assert(0 && "All other types should have been promoted!!");
       case MVT::i32:
       case MVT::i64:
	 BuildMI(BB, Alpha::BIS, 2, Alpha::R0).addReg(Tmp1).addReg(Tmp1);
	 break;
       }
       break;
     case 1:
       Select(N.getOperand(0));
       break;
     }
     //Tmp2 = AlphaLowering.getRetAddr();
     //BuildMI(BB, Alpha::BIS, 2, Alpha::R26).addReg(Tmp2).addReg(Tmp2);
     BuildMI(BB, Alpha::RETURN, 0); // Just emit a 'ret' instruction
     return;

  case ISD::STORE: 
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1)); //value
    if (N.getOperand(2).getOpcode() == ISD::GlobalAddress)
      {
	AlphaLowering.restoreGP(BB);
	BuildMI(BB, Alpha::STORE, 2).addReg(Tmp1).addGlobalAddress(cast<GlobalAddressSDNode>(N.getOperand(2))->getGlobal());
      }
    else
      {
	Tmp2 = SelectExpr(N.getOperand(2)); //address
	BuildMI(BB, Alpha::STQ, 3).addReg(Tmp1).addImm(0).addReg(Tmp2);
      }
    return;

  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::LOAD:
  case ISD::CopyFromReg:
  case ISD::CALL:
//   case ISD::DYNAMIC_STACKALLOC:
    SelectExpr(N);
    return;


  case ISD::TRUNCSTORE: {  // truncstore chain, val, ptr :storety
    MVT::ValueType StoredTy = cast<MVTSDNode>(Node)->getExtraValueType();
    assert(StoredTy != MVT::i64 && "Unsupported TRUNCSTORE for this target!");

    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    Tmp2 = SelectExpr(N.getOperand(2));

    switch (StoredTy) {
    default: assert(0 && "Unhandled Type"); break;
    case MVT::i1: //FIXME: DAG does not promote this load
    case MVT::i8: Opc = Alpha::STB; break;
    case MVT::i16: Opc = Alpha::STW; break;
    case MVT::i32: Opc = Alpha::STL; break;
    }

    BuildMI(BB, Opc, 2).addReg(Tmp1).addImm(0).addReg(Tmp2);
    return;
  }

  case ISD::ADJCALLSTACKDOWN:
  case ISD::ADJCALLSTACKUP:
    Select(N.getOperand(0));
    Tmp1 = cast<ConstantSDNode>(N.getOperand(1))->getValue();
    
    Opc = N.getOpcode() == ISD::ADJCALLSTACKDOWN ? Alpha::ADJUSTSTACKDOWN :
      Alpha::ADJUSTSTACKUP;
    BuildMI(BB, Opc, 1).addImm(Tmp1);
    return;
  }
  assert(0 && "Should not be reached!");
}


/// createAlphaPatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *llvm::createAlphaPatternInstructionSelector(TargetMachine &TM) {
  return new ISel(TM);  
}
