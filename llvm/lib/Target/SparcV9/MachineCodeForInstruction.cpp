//===-- MachineCodeForInstruction.cpp -------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Container for the sequence of MachineInstrs created for a single
// LLVM Instruction.  MachineCodeForInstruction also tracks temporary values
// (TmpInstruction objects) created during SparcV9 code generation, so that
// they can be deleted when they are no longer needed, and finally, it also
// holds some extra information for 'call' Instructions (using the
// CallArgsDescriptor object, which is also implemented in this file).
//
//===----------------------------------------------------------------------===//

#include "MachineCodeForInstruction.h"
#include "llvm/Function.h"
#include "llvm/Instructions.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "MachineFunctionInfo.h"
#include "MachineInstrAnnot.h"
#include "SparcV9TmpInstr.h"
#include "SparcV9RegisterInfo.h"
using namespace llvm;

MachineCodeForInstruction &MachineCodeForInstruction::get(const Instruction *I){
  MachineFunction &MF = MachineFunction::get(I->getParent()->getParent());
  return MF.getInfo<SparcV9FunctionInfo>()->MCFIEntries[I];
}

void MachineCodeForInstruction::destroy(const Instruction *I) {
  MachineFunction &MF = MachineFunction::get(I->getParent()->getParent());
  MF.getInfo<SparcV9FunctionInfo>()->MCFIEntries.erase(I);
}

void MachineCodeForInstruction::dropAllReferences() {
  for (unsigned i=0, N=tempVec.size(); i < N; i++)
    cast<Instruction>(tempVec[i])->dropAllReferences();
}

MachineCodeForInstruction::~MachineCodeForInstruction() {
  // Let go of all uses in temp. instructions
  dropAllReferences();

  // Free the Value objects created to hold intermediate values
  for (unsigned i=0, N=tempVec.size(); i < N; i++)
    delete tempVec[i];

  // do not free the MachineInstr objects allocated. they are managed
  // by the ilist in MachineBasicBlock

  // Free the CallArgsDescriptor if it exists.
  delete callArgsDesc;
}

CallArgsDescriptor::CallArgsDescriptor(CallInst* _callInstr,
                                       TmpInstruction* _retAddrReg,
                                       bool _isVarArgs, bool _noPrototype)
  : callInstr(_callInstr),
    funcPtr(isa<Function>(_callInstr->getCalledValue())
            ? NULL : _callInstr->getCalledValue()),
    retAddrReg(_retAddrReg),
    isVarArgs(_isVarArgs),
    noPrototype(_noPrototype) {
  unsigned int numArgs = callInstr->getNumOperands();
  argInfoVec.reserve(numArgs);
  assert(callInstr->getOperand(0) == callInstr->getCalledValue()
         && "Operand 0 is ignored in the loop below!");
  for (unsigned int i=1; i < numArgs; ++i)
    argInfoVec.push_back(CallArgInfo(callInstr->getOperand(i)));

  // Enter this object in the MachineCodeForInstr object of the CallInst.
  // This transfers ownership of this object.
  MachineCodeForInstruction::get(callInstr).setCallArgsDescriptor(this);
}

CallInst *CallArgsDescriptor::getReturnValue() const {
  return (callInstr->getType() == Type::VoidTy? NULL : callInstr);
}

/// CallArgsDescriptor::get - Mechanism to get the descriptor for a CALL
/// MachineInstr.  We get the LLVM CallInst from the return-address register
/// argument of the CALL MachineInstr (which is explicit operand #2 for
/// indirect calls or the last implicit operand for direct calls).  We then get
/// the CallArgsDescriptor from the MachineCodeForInstruction object for the
/// CallInstr.  This is roundabout but avoids adding a new map or annotation
/// just to keep track of CallArgsDescriptors.
///
CallArgsDescriptor *CallArgsDescriptor::get(const MachineInstr *MI) {
  const Value *retAddrVal = 0;
  if ((MI->getOperand (0).getType () == MachineOperand::MO_MachineRegister
       && MI->getOperand (0).getReg () == SparcV9::g0)
      || (MI->getOperand (0).getType () == MachineOperand::MO_VirtualRegister
          && !isa<Function> (MI->getOperand (0).getVRegValue ()))) {
    retAddrVal = MI->getOperand (2).getVRegValue ();
  } else {
    retAddrVal = MI->getImplicitRef (MI->getNumImplicitRefs () - 1);
  }

  const TmpInstruction* retAddrReg = cast<TmpInstruction> (retAddrVal);
  assert(retAddrReg->getNumOperands() == 1 &&
         isa<CallInst>(retAddrReg->getOperand(0)) &&
         "Location of callInstr arg for CALL instr. changed? FIX THIS CODE!");

  const CallInst* callInstr = cast<CallInst>(retAddrReg->getOperand(0));

  CallArgsDescriptor* desc =
    MachineCodeForInstruction::get(callInstr).getCallArgsDescriptor();
  assert(desc->getCallInst()==callInstr && "Incorrect call args descriptor?");
  return desc;
}
