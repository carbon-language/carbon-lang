//===-- SparcV8ISelDAGToDAG.cpp - A dag to dag inst selector for SparcV8 --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines an instruction selector for the V8 target
//
//===----------------------------------------------------------------------===//

#include "SparcV8.h"
#include "SparcV8TargetMachine.h"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SSARegMap.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Support/Debug.h"
#include <iostream>
using namespace llvm;

//===----------------------------------------------------------------------===//
// TargetLowering Implementation
//===----------------------------------------------------------------------===//

namespace {
  class SparcV8TargetLowering : public TargetLowering {
  public:
    SparcV8TargetLowering(TargetMachine &TM);

    virtual std::vector<SDOperand>
      LowerArguments(Function &F, SelectionDAG &DAG);
    virtual std::pair<SDOperand, SDOperand>
      LowerCallTo(SDOperand Chain, const Type *RetTy, bool isVarArg,
                  unsigned CC,
                  bool isTailCall, SDOperand Callee, ArgListTy &Args,
                  SelectionDAG &DAG);
    
    virtual SDOperand LowerReturnTo(SDOperand Chain, SDOperand Op,
                                    SelectionDAG &DAG);
    virtual SDOperand LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                   Value *VAListV, SelectionDAG &DAG);
    virtual std::pair<SDOperand,SDOperand>
      LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
                 const Type *ArgTy, SelectionDAG &DAG);
    virtual std::pair<SDOperand, SDOperand>
      LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                              SelectionDAG &DAG);
  };
}

SparcV8TargetLowering::SparcV8TargetLowering(TargetMachine &TM)
  : TargetLowering(TM) {
  
  // Set up the register classes.
  addRegisterClass(MVT::i32, V8::IntRegsRegisterClass);
  addRegisterClass(MVT::f32, V8::FPRegsRegisterClass);
  addRegisterClass(MVT::f64, V8::DFPRegsRegisterClass);
  
  computeRegisterProperties();
}

std::vector<SDOperand>
SparcV8TargetLowering::LowerArguments(Function &F, SelectionDAG &DAG) {
  MachineFunction &MF = DAG.getMachineFunction();
  SSARegMap *RegMap = MF.getSSARegMap();
  std::vector<SDOperand> ArgValues;
  
  static const unsigned GPR[] = {
    V8::I0, V8::I1, V8::I2, V8::I3, V8::I4, V8::I5
  };
  unsigned ArgNo = 0;
  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I) {
    MVT::ValueType ObjectVT = getValueType(I->getType());
    assert(ArgNo < 6 && "Only args in regs for now");
    
    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    // TODO: MVT::i64 & FP
    case MVT::i1:
    case MVT::i8:
    case MVT::i16:
    case MVT::i32: {
      unsigned VReg = RegMap->createVirtualRegister(&V8::IntRegsRegClass);
      MF.addLiveIn(GPR[ArgNo++], VReg);
      SDOperand Arg = DAG.getCopyFromReg(DAG.getRoot(), VReg, MVT::i32);
      DAG.setRoot(Arg.getValue(1));
      if (ObjectVT != MVT::i32) {
        unsigned AssertOp = I->getType()->isSigned() ? ISD::AssertSext 
                                                     : ISD::AssertZext;
        Arg = DAG.getNode(AssertOp, MVT::i32, Arg, 
                          DAG.getValueType(ObjectVT));
        Arg = DAG.getNode(ISD::TRUNCATE, ObjectVT, Arg);
      }
      ArgValues.push_back(Arg);
    }
    }
  }
  
  assert(!F.isVarArg() && "Unimp");
  
  // Finally, inform the code generator which regs we return values in.
  switch (getValueType(F.getReturnType())) {
  default: assert(0 && "Unknown type!");
  case MVT::isVoid: break;
  case MVT::i1:
  case MVT::i8:
  case MVT::i16:
  case MVT::i32:
    MF.addLiveOut(V8::I0);
    break;
  case MVT::i64:
    MF.addLiveOut(V8::I0);
    MF.addLiveOut(V8::I1);
    break;
  case MVT::f32:
    MF.addLiveOut(V8::F0);
    break;
  case MVT::f64:
    MF.addLiveOut(V8::D0);
    break;
  }
  
  return ArgValues;
}

std::pair<SDOperand, SDOperand>
SparcV8TargetLowering::LowerCallTo(SDOperand Chain, const Type *RetTy,
                                   bool isVarArg, unsigned CC,
                                   bool isTailCall, SDOperand Callee, 
                                   ArgListTy &Args, SelectionDAG &DAG) {
  assert(0 && "Unimp");
  abort();
}

SDOperand SparcV8TargetLowering::LowerReturnTo(SDOperand Chain, SDOperand Op,
                                               SelectionDAG &DAG) {
  assert(0 && "Unimp");
  abort();
}

SDOperand SparcV8TargetLowering::LowerVAStart(SDOperand Chain, SDOperand VAListP,
                                              Value *VAListV, SelectionDAG &DAG) {
  assert(0 && "Unimp");
  abort();
}

std::pair<SDOperand,SDOperand>
SparcV8TargetLowering::LowerVAArg(SDOperand Chain, SDOperand VAListP, Value *VAListV,
                                  const Type *ArgTy, SelectionDAG &DAG) {
  assert(0 && "Unimp");
  abort();
}

std::pair<SDOperand, SDOperand>
SparcV8TargetLowering::LowerFrameReturnAddress(bool isFrameAddr, SDOperand Chain, unsigned Depth,
                                               SelectionDAG &DAG) {
  assert(0 && "Unimp");
  abort();
}

//===----------------------------------------------------------------------===//
// Instruction Selector Implementation
//===----------------------------------------------------------------------===//

//===--------------------------------------------------------------------===//
/// SparcV8DAGToDAGISel - PPC specific code to select Sparc V8 machine
/// instructions for SelectionDAG operations.
///
namespace {
class SparcV8DAGToDAGISel : public SelectionDAGISel {
  SparcV8TargetLowering V8Lowering;
public:
  SparcV8DAGToDAGISel(TargetMachine &TM)
    : SelectionDAGISel(V8Lowering), V8Lowering(TM) {}

  SDOperand Select(SDOperand Op);

  /// InstructionSelectBasicBlock - This callback is invoked by
  /// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
  virtual void InstructionSelectBasicBlock(SelectionDAG &DAG);
  
  virtual const char *getPassName() const {
    return "PowerPC DAG->DAG Pattern Instruction Selection";
  } 
  
  // Include the pieces autogenerated from the target description.
#include "SparcV8GenDAGISel.inc"
};
}  // end anonymous namespace

/// InstructionSelectBasicBlock - This callback is invoked by
/// SelectionDAGISel when it has created a SelectionDAG for us to codegen.
void SparcV8DAGToDAGISel::InstructionSelectBasicBlock(SelectionDAG &DAG) {
  DEBUG(BB->dump());
  
  // Select target instructions for the DAG.
  DAG.setRoot(Select(DAG.getRoot()));
  CodeGenMap.clear();
  DAG.RemoveDeadNodes();
  
  // Emit machine code to BB. 
  ScheduleAndEmitDAG(DAG);
}


SDOperand SparcV8DAGToDAGISel::Select(SDOperand Op) {
  SDNode *N = Op.Val;
  if (N->getOpcode() >= ISD::BUILTIN_OP_END/* &&
      N->getOpcode() < V8ISD::FIRST_NUMBER*/)
    return Op;   // Already selected.
                 // If this has already been converted, use it.
  std::map<SDOperand, SDOperand>::iterator CGMI = CodeGenMap.find(Op);
  if (CGMI != CodeGenMap.end()) return CGMI->second;
  
  switch (N->getOpcode()) {
  default: break;
  }
  
  return SelectCode(Op);
}


/// createPPCISelDag - This pass converts a legalized DAG into a 
/// PowerPC-specific DAG, ready for instruction scheduling.
///
FunctionPass *llvm::createSparcV8ISelDag(TargetMachine &TM) {
  return new SparcV8DAGToDAGISel(TM);
}
