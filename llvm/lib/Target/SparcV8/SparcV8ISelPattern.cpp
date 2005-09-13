#if 0
//===- SparcV8ISelPattern.cpp - A pattern matching isel for SparcV8 -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for SparcV8.
//
//===----------------------------------------------------------------------===//

//Please note that this file is a work in progress, and not a high
//priority for anyone.

#include "SparcV8.h"
#include "SparcV8RegisterInfo.h"
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
#include "llvm/Support/Debug.h"
#include "llvm/Support/CommandLine.h"
#include <set>
#include <algorithm>
using namespace llvm;

//===----------------------------------------------------------------------===//
//  V8TargetLowering - SparcV8 Implementation of the TargetLowering interface
namespace {
  class V8TargetLowering : public TargetLowering {
    int VarArgsFrameIndex;            // FrameIndex for start of varargs area.
  public:
    V8TargetLowering(TargetMachine &TM) : TargetLowering(TM) {
      // Set up the TargetLowering object.
      //I am having problems with shr n ubyte 1
      setShiftAmountType(MVT::i32);
      setSetCCResultType(MVT::i32);
      setSetCCResultContents(ZeroOrOneSetCCResult);

      //FIXME: get these right
      addRegisterClass(MVT::i64, V8::GPRCRegisterClass);
      addRegisterClass(MVT::f64, V8::FPRCRegisterClass);
      addRegisterClass(MVT::f32, V8::FPRCRegisterClass);

      setOperationAction(ISD::BRCONDTWOWAY, MVT::Other, Expand);
      setOperationAction(ISD::BRTWOWAY_CC,  MVT::Other, Expand);
      setOperationAction(ISD::EXTLOAD, MVT::i1,  Promote);
      setOperationAction(ISD::EXTLOAD, MVT::f32, Promote);

      setOperationAction(ISD::ZEXTLOAD, MVT::i1, Expand);
      setOperationAction(ISD::SEXTLOAD, MVT::i1, Expand);

      setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i1,  Expand);
      setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i8,  Expand);
      setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i16, Expand);
      setOperationAction(ISD::SIGN_EXTEND_INREG, MVT::i32, Expand);

      setOperationAction(ISD::UREM, MVT::i32, Expand);
      setOperationAction(ISD::SREM, MVT::i32, Expand);

      setOperationAction(ISD::CTPOP, MVT::i32, Expand);
      setOperationAction(ISD::CTTZ, MVT::i32, Expand);
      setOperationAction(ISD::CTLZ, MVT::i32, Expand);

      setOperationAction(ISD::MEMMOVE, MVT::Other, Expand);
      setOperationAction(ISD::MEMSET,  MVT::Other, Expand);
      setOperationAction(ISD::MEMCPY,  MVT::Other, Expand);

      // We don't support sin/cos/sqrt
      setOperationAction(ISD::FSIN , MVT::f64, Expand);
      setOperationAction(ISD::FCOS , MVT::f64, Expand);
      setOperationAction(ISD::FSQRT, MVT::f64, Expand);
      setOperationAction(ISD::FSIN , MVT::f32, Expand);
      setOperationAction(ISD::FCOS , MVT::f32, Expand);
      setOperationAction(ISD::FSQRT, MVT::f32, Expand);

      computeRegisterProperties();

      addLegalFPImmediate(+0.0); //F31
      addLegalFPImmediate(-0.0); //-F31
    }

    /// LowerArguments - This hook must be implemented to indicate how we should
    /// lower the arguments for the specified function, into the specified DAG.
    virtual std::vector<SDOperand>
    LowerArguments(Function &F, SelectionDAG &DAG);

    /// LowerCallTo - This hook lowers an abstract call to a function into an
    /// actual call.
    virtual std::pair<SDOperand, SDOperand>
    LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg, unsigned CC,
                bool isTailCall, SDOperand Callee, ArgListTy &Args,
                SelectionDAG &DAG);
  };
}

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

std::vector<SDOperand>
V8TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG)
{
  static const unsigned IncomingArgRegs[] =
    { V8::I0, V8::I1, V8::I2, V8::I3, V8::I4, V8::I5 };
  std::vector<SDOperand> ArgValues;

  MachineFunction &MF = DAG.getMachineFunction();
  MachineFrameInfo*MFI = MF.getFrameInfo();

  MachineBasicBlock& BB = MF.front();

  unsigned ArgNo = 0;
  unsigned ArgOffset = 92;
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end();
       I != E; ++I, ++ArgNo) {
    MVT::ValueType VT = getValueType(I->getType());
    SDOperand argt;
    if (ArgNo < 6) {
      switch(VT) {
      default:
        std::cerr << "Unknown Type " << VT << "\n";
        abort();
      case MVT::f64:
      case MVT::i64:
        //FIXME: figure out the build pair thing
        assert(0 && "doubles and longs not supported yet");
      case MVT::f32:
        argt = DAG.getCopyFromReg(AddLiveIn(MF, IncomingArgRegs[ArgNo],
                                            MVT::i32),
                                  VT, DAG.getRoot());
        //copy out of Int reg
        argt = DAG.getNode(ISD::FP_TO_UINT, MVT::f32, argt);
        break;
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
        argt = DAG.getCopyFromReg(AddLiveIn(MF, IncomingArgRegs[ArgNo],
                                            getRegClassFor(MVT::i32)),
                                  VT, DAG.getRoot());
        if (VT != MVT::i32)
          argt = DAG.getNode(ISD::TRUNCATE, VT, argt);
        break;
      }
      DAG.setRoot(argt.getValue(1));
    } else {
      //stack passed
      switch(VT) {
      default:
        std::cerr << "Unknown Type " << VT << "\n";
        abort();
      case MVT::f64:
      case MVT::i64:
        //FIXME: figure out the build pair thing
        assert(0 && "doubles and longs not supported yet");
      case MVT::f32:
      case MVT::i1:
      case MVT::i8:
      case MVT::i16:
      case MVT::i32:
      // Create the frame index object for this incoming parameter...
      int FI = MFI->CreateFixedObject(4, ArgOffset);
      argt = DAG.getLoad(VT,
                         DAG.getEntryNode(),
                         DAG.getFramIndex(FI, MVT::i32),
                         DAG.getSrcValue(NULL));
      ArgOffset += 4;
      break;
      }
      ArgValues.push_back(argt);
    }
  }

  //return the arguments
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
V8TargetLowering::LowerCallTo(SDOperand Chain,
                                 const Type *RetTy, bool isVarArg,
                                 unsigned CallingConv, bool isTailCall,
                                 SDOperand Callee, ArgListTy &Args,
                                 SelectionDAG &DAG) {
  //FIXME
  return std::make_pair(Chain, Chain);
}

namespace {

//===--------------------------------------------------------------------===//
/// ISel - V8 specific code to select V8 machine instructions for
/// SelectionDAG operations.
//===--------------------------------------------------------------------===//
class ISel : public SelectionDAGISel {

  /// V8Lowering - This object fully describes how to lower LLVM code to an
  /// V8-specific SelectionDAG.
  V8TargetLowering V8Lowering;

  SelectionDAG *ISelDAG;  // Hack to support us having a dag->dag transform
                          // for sdiv and udiv until it is put into the future
                          // dag combiner.

  /// ExprMap - As shared expressions are codegen'd, we keep track of which
  /// vreg the value is produced in, so we only emit one copy of each compiled
  /// tree.
  static const unsigned notIn = (unsigned)(-1);
  std::map<SDOperand, unsigned> ExprMap;

public:
  ISel(TargetMachine &TM) : SelectionDAGISel(V8Lowering), V8Lowering(TM)
  {}

  /// InstructionSelectBasicBlock - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG) {
    DEBUG(BB->dump());

    // Codegen the basic block.
    ISelDAG = &DAG;
    max_depth = DAG.getRoot().getNodeDepth();
    Select(DAG.getRoot());

    // Clear state used for selection.
    ExprMap.clear();
  }

  unsigned SelectExpr(SDOperand N);
  void Select(SDOperand N);

};
}

//These describe LDAx
static const int IMM_LOW  = -32768;
static const int IMM_HIGH = 32767;
static const int IMM_MULT = 65536;

static long getUpper16(long l)
{
  long y = l / IMM_MULT;
  if (l % IMM_MULT > IMM_HIGH)
    ++y;
  return y;
}

static long getLower16(long l)
{
  long h = getUpper16(l);
  return l - h * IMM_MULT;
}

unsigned ISel::SelectExpr(SDOperand N) {
  unsigned Result;
  unsigned Tmp1, Tmp2 = 0, Tmp3;
  unsigned Opc = 0;
  unsigned opcode = N.getOpcode();

  SDNode *Node = N.Val;
  MVT::ValueType DestType = N.getValueType();

  unsigned &Reg = ExprMap[N];
  if (Reg) return Reg;

  if (N.getOpcode() != ISD::CALL && N.getOpcode() != ISD::TAILCALL)
    Reg = Result = (N.getValueType() != MVT::Other) ?
      MakeReg(N.getValueType()) : notIn;
  else {
    // If this is a call instruction, make sure to prepare ALL of the result
    // values as well as the chain.
    if (Node->getNumValues() == 1)
      Reg = Result = notIn;  // Void call, just a chain.
    else {
      Result = MakeReg(Node->getValueType(0));
      ExprMap[N.getValue(0)] = Result;
      for (unsigned i = 1, e = N.Val->getNumValues()-1; i != e; ++i)
        ExprMap[N.getValue(i)] = MakeReg(Node->getValueType(i));
      ExprMap[SDOperand(Node, Node->getNumValues()-1)] = notIn;
    }
  }

  switch (opcode) {
  default:
    Node->dump();
    assert(0 && "Node not handled!\n");

  case ISD::EXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::LOAD:
    {
      // Make sure we generate both values.
      if (Result != notIn)
        ExprMap[N.getValue(1)] = notIn;   // Generate the token
      else
        Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

      SDOperand Chain   = N.getOperand(0);
      SDOperand Address = N.getOperand(1);
      Select(Chain);
      unsigned Adr = SelectExpr(Address);
      switch(cast<VTSDNode>(Node->getOperand(3))->getVT()) {
      case MVT::i32: Opc = V8::LD;
      case MVT::i16: Opc = opcode == ISD::ZEXTLOAD ? V8::LDUH : V8::LDSH; break;
      case MVT::i8:  Opc = opcode == ISD::ZEXTLOAD ? V8::LDUB : V8::LDSB; break;
      case MVT::f64: Opc = V8::LDFSRrr;
      case MVT::f32: Opc = V8::LDDFrr;
      default:
        Node->dump();
        assert(0 && "Bad type!");
        break;
      }
      BuildMI(BB, Opc, 1, Result).addReg(Adr);
      return Result;
    }

  case ISD::TAILCALL:
  case ISD::CALL:
    {
      //FIXME:
      abort();
      return Result;
    }

  case ISD::CopyFromReg:
    {
      // Make sure we generate both values.
      if (Result != notIn)
        ExprMap[N.getValue(1)] = notIn;   // Generate the token
      else
        Result = ExprMap[N.getValue(0)] = MakeReg(N.getValue(0).getValueType());

      SDOperand Chain   = N.getOperand(0);
      Select(Chain);
      unsigned r = dyn_cast<RegSDNode>(Node)->getReg();

      BuildMI(BB, V8::ORrr, 2, Result).addReg(r).addReg(V8::G0);
      return Result;
    }

    //Most of the plain arithmetic and logic share the same form, and the same
    //constant immediate test
  case ISD::XOR:
  case ISD::AND:
  case ISD::OR:
  case ISD::SHL:
  case ISD::SRL:
  case ISD::SRA:
  case ISD::ADD:
  case ISD::SUB:
  case ISD::SDIV:
  case ISD::UDIV:
  case ISD::SMUL:
  case ISD::UMUL:
    switch(opcode) {
    case ISD::XOR:  Opc = V8::XORrr; break;
    case ISD::AND:  Opc = V8::ANDrr; break;
    case ISD::OR:   Opc = V8::ORrr; break;
    case ISD::SHL:  Opc = V8::SLLrr; break;
    case ISD::SRL:  Opc = V8::SRLrr; break;
    case ISD::SRA:  Opc = V8::SRArr; break;
    case ISD::ADD:  Opc = V8::ADDrr; break;
    case ISD::SUB:  Opc = V8::SUBrr; break;
    case ISD::SDIV: Opc = V8::SDIVrr; break;
    case ISD::UDIV: Opc = V8::UDIVrr; break;
    case ISD::SMUL: Opc = V8::SMULrr; break;
    case ISD::UMUL: Opc = V8::UMULrr; break;
    }
    Tmp1 = SelectExpr(N.getOperand(0));
    Tmp2 = SelectExpr(N.getOperand(1));
    BuildMI(BB, Opc, 2, Result).addReg(Tmp1).addReg(Tmp2);
    return Result;

  }
  return 0;
}

void ISel::Select(SDOperand N) {
  unsigned Tmp1, Tmp2, Opc;
  unsigned opcode = N.getOpcode();

  if (!ExprMap.insert(std::make_pair(N, notIn)).second)
    return;  // Already selected.

  SDNode *Node = N.Val;

  switch (opcode) {

  default:
    Node->dump(); std::cerr << "\n";
    assert(0 && "Node not handled yet!");

  case ISD::BRCOND: {
    //FIXME
    abort();
    return;
  }

  case ISD::BR: {
    MachineBasicBlock *Dest =
      cast<BasicBlockSDNode>(N.getOperand(1))->getBasicBlock();

    Select(N.getOperand(0));
    BuildMI(BB, V8::BA, 1).addMBB(Dest);
    return;
  }

  case ISD::ImplicitDef:
    Select(N.getOperand(0));
    BuildMI(BB, V8::IMPLICIT_DEF, 0, cast<RegSDNode>(N)->getReg());
    return;

  case ISD::EntryToken: return;  // Noop

  case ISD::TokenFactor:
    for (unsigned i = 0, e = Node->getNumOperands(); i != e; ++i)
      Select(Node->getOperand(i));
    return;

  case ISD::CopyToReg:
    Select(N.getOperand(0));
    Tmp1 = SelectExpr(N.getOperand(1));
    Tmp2 = cast<RegSDNode>(N)->getReg();

    if (Tmp1 != Tmp2) {
      if (N.getOperand(1).getValueType() == MVT::f64 ||
          N.getOperand(1).getValueType() == MVT::f32)
        BuildMI(BB, V8::FMOVS, 2, Tmp2).addReg(Tmp1);
      else
        BuildMI(BB, V8::ORrr, 2, Tmp2).addReg(Tmp1).addReg(V8::G0);
    }
    return;

  case ISD::RET:
    //FIXME:
    abort();
    return;

  case ISD::TRUNCSTORE:
  case ISD::STORE:
    {
      SDOperand Chain   = N.getOperand(0);
      SDOperand Value = N.getOperand(1);
      SDOperand Address = N.getOperand(2);
      Select(Chain);

      Tmp1 = SelectExpr(Value);
      Tmp2 = SelectExpr(Address);

      unsigned VT = opcode == ISD::STORE ?
        Value.getValueType() : cast<VTSDNode>(Node->getOperand(4))->getVT();
      switch(VT) {
      default: assert(0 && "unknown Type in store");
      case MVT::f64: Opc = V8::STDFrr; break;
      case MVT::f32: Opc = V8::STFrr; break;
      case MVT::i1:  //FIXME: DAG does not promote this load
      case MVT::i8:  Opc = V8::STBrr; break;
      case MVT::i16: Opc = V8::STHrr; break;
      case MVT::i32: Opc = V8::STLrr; break;
      case MVT::i64: Opc = V8::STDrr; break;
      }

      BuildMI(BB,Opc,2).addReg(Tmp1).addReg(Tmp2);
      return;
    }

  case ISD::EXTLOAD:
  case ISD::SEXTLOAD:
  case ISD::ZEXTLOAD:
  case ISD::LOAD:
  case ISD::CopyFromReg:
  case ISD::TAILCALL:
  case ISD::CALL:
  case ISD::DYNAMIC_STACKALLOC:
    ExprMap.erase(N);
    SelectExpr(N);
    return;

  case ISD::CALLSEQ_START:
  case ISD::CALLSEQ_END:
    Select(N.getOperand(0));
    Tmp1 = cast<ConstantSDNode>(N.getOperand(1))->getValue();

    Opc = N.getOpcode() == ISD::CALLSEQ_START ? V8::ADJUSTCALLSTACKDOWN :
      V8::ADJUSTCALLSTACKUP;
    BuildMI(BB, Opc, 1).addImm(Tmp1);
    return;
  }
  assert(0 && "Should not be reached!");
}


/// createV8PatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *llvm::createV8PatternInstructionSelector(TargetMachine &TM) {
  return new ISel(TM);
}

#endif
