//===-- InstSelectPattern.cpp - A pattern matching inst selector for X86 --===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file defines a pattern matching instruction selector for X86.
//
//  FIXME: we could allocate one big array of unsigneds to use as the backing
//         store for all of the nodes costs arrays.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/DerivedTypes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/SSARegMap.h"

#include "X86RegisterInfo.h"

// Include the generated instruction selector...
#include "X86GenInstrSelector.inc"
using namespace llvm;

namespace {
  struct ISel : public FunctionPass, SelectionDAGTargetBuilder {
    TargetMachine &TM;
    ISel(TargetMachine &tm) : TM(tm) {}
    int VarArgsFrameIndex;              // FrameIndex for start of varargs area

    bool runOnFunction(Function &Fn) {
      MachineFunction &MF = MachineFunction::construct(&Fn, TM);
      SelectionDAG DAG(MF, TM, *this);

      std::cerr << "\n\n\n=== "
                << DAG.getMachineFunction().getFunction()->getName() << "\n";

      DAG.dump();
      X86ISel(DAG).generateCode();
      std::cerr << "\n\n\n";
      return true;
    }

  public:  // Implementation of the SelectionDAGTargetBuilder class...
    /// expandArguments - Add nodes to the DAG to indicate how to load arguments
    /// off of the X86 stack.
    void expandArguments(SelectionDAG &SD);
    void expandCall(SelectionDAG &SD, CallInst &CI);
  };
}


void ISel::expandArguments(SelectionDAG &SD) {

  // Add DAG nodes to load the arguments...  On entry to a function on the X86,
  // the stack frame looks like this:
  //
  // [ESP] -- return address
  // [ESP + 4] -- first argument (leftmost lexically)
  // [ESP + 8] -- second argument, if first argument is four bytes in size
  //    ... 
  //
  MachineFunction &F = SD.getMachineFunction();
  MachineFrameInfo *MFI = F.getFrameInfo();
  const Function &Fn = *F.getFunction();
  
  unsigned ArgOffset = 0;   // Frame mechanisms handle retaddr slot
  for (Function::const_aiterator I = Fn.abegin(), E = Fn.aend(); I != E; ++I) {
    MVT::ValueType ObjectVT = SD.getValueType(I->getType());
    unsigned ArgIncrement = 4;
    unsigned ObjSize;
    switch (ObjectVT) {
    default: assert(0 && "Unhandled argument type!");
    case MVT::i8:  ObjSize = 1;                break;
    case MVT::i16: ObjSize = 2;                break;
    case MVT::i32: ObjSize = 4;                break;
    case MVT::i64: ObjSize = ArgIncrement = 8; break;
    case MVT::f32: ObjSize = 4;                break;
    case MVT::f64: ObjSize = ArgIncrement = 8; break;
    }
    // Create the frame index object for this incoming parameter...
    int FI = MFI->CreateFixedObject(ObjSize, ArgOffset);
    
    // Create the SelectionDAG nodes corresponding to a load from this parameter
    SelectionDAGNode *FIN = new SelectionDAGNode(ISD::FrameIndex, MVT::i32);
    FIN->addValue(new ReducedValue_FrameIndex_i32(FI));

    SelectionDAGNode *Arg
      = new SelectionDAGNode(ISD::Load, ObjectVT, F.begin(), FIN);

    // Add the SelectionDAGNodes to the SelectionDAG... note that there is no
    // reason to add chain nodes here.  We know that no loads ore stores will
    // ever alias these loads, so we are free to perform the load at any time in
    // the function
    SD.addNode(FIN);
    SD.addNodeForValue(Arg, I);

    ArgOffset += ArgIncrement;   // Move on to the next argument...
  }

  // If the function takes variable number of arguments, make a frame index for
  // the start of the first vararg value... for expansion of llvm.va_start.
  if (Fn.getFunctionType()->isVarArg())
    VarArgsFrameIndex = MFI->CreateFixedObject(1, ArgOffset);
}

void ISel::expandCall(SelectionDAG &SD, CallInst &CI) {
  assert(0 && "ISel::expandCall not implemented!");
}

/// createX86PatternInstructionSelector - This pass converts an LLVM function
/// into a machine code representation using pattern matching and a machine
/// description file.
///
FunctionPass *llvm::createX86PatternInstructionSelector(TargetMachine &TM,
                                                        IntrinsicLowering &IL) {
  return new ISel(TM);  
}
