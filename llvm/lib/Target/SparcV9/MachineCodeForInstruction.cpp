//===-- MachineCodeForInstruction.cpp -------------------------------------===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Representation of the sequence of machine instructions created for a single
// VM instruction.  Additionally records information about hidden and implicit
// values used by the machine instructions: about hidden values used by the
// machine instructions:
// 
// "Temporary values" are intermediate values used in the machine instruction
// sequence, but not in the VM instruction. Note that such values should be
// treated as pure SSA values with no interpretation of their operands (i.e., as
// a TmpInstruction object which actually represents such a value).
// 
// (2) "Implicit uses" are values used in the VM instruction but not in the
//     machine instruction sequence
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
using namespace llvm;

MachineCodeForInstruction &MachineCodeForInstruction::get(const Instruction *I){
  MachineFunction &MF = MachineFunction::get(I->getParent()->getParent());
  return MF.getInfo()->MCFIEntries[I];
}
void MachineCodeForInstruction::destroy(const Instruction *I) {
  MachineFunction &MF = MachineFunction::get(I->getParent()->getParent());
  MF.getInfo()->MCFIEntries.erase(I);
}

void
MachineCodeForInstruction::dropAllReferences()
{
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

// Mechanism to get the descriptor for a CALL MachineInstr.
// We get the LLVM CallInstr from the ret. addr. register argument
// of the CALL MachineInstr (which is explicit operand #3 for indirect
// calls or the last implicit operand for direct calls).  We then get
// the CallArgsDescriptor from the MachineCodeForInstruction object for
// the CallInstr.
// This is roundabout but avoids adding a new map or annotation just
// to keep track of CallArgsDescriptors.
// 
CallArgsDescriptor *CallArgsDescriptor::get(const MachineInstr* MI) {
  const TmpInstruction* retAddrReg =
    cast<TmpInstruction>(isa<Function>(MI->getOperand(0).getVRegValue())
                         ? MI->getImplicitRef(MI->getNumImplicitRefs()-1)
                         : MI->getOperand(2).getVRegValue());

  assert(retAddrReg->getNumOperands() == 1 &&
         isa<CallInst>(retAddrReg->getOperand(0)) &&
         "Location of callInstr arg for CALL instr. changed? FIX THIS CODE!");

  const CallInst* callInstr = cast<CallInst>(retAddrReg->getOperand(0));

  CallArgsDescriptor* desc =
    MachineCodeForInstruction::get(callInstr).getCallArgsDescriptor(); 
  assert(desc->getCallInst()==callInstr && "Incorrect call args descriptor?");
  return desc;
}
