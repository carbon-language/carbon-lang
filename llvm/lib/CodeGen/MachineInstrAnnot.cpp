//===-- MachineInstrAnnot.cpp ---------------------------------------------===//
// 
//  This file defines Annotations used to pass information between code
//  generation phases.
// 
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstrAnnot.h"
#include "llvm/CodeGen/InstrSelection.h"
#include "llvm/CodeGen/MachineCodeForInstruction.h"
#include "llvm/iOther.h"
#include "llvm/Type.h"


CallArgsDescriptor::CallArgsDescriptor(CallInst* _callInstr,
                                       TmpInstruction* _retAddrReg,
                                       bool _isVarArgs, bool _noPrototype)
  : callInstr(_callInstr),
    funcPtr(isa<Function>(_callInstr->getCalledValue())
            ? NULL : _callInstr->getCalledValue()),
    retAddrReg(_retAddrReg),
    isVarArgs(_isVarArgs),
    noPrototype(_noPrototype)
{
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


CallInst*
CallArgsDescriptor::getReturnValue() const
{
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
CallArgsDescriptor *CallArgsDescriptor::get(const MachineInstr* MI)
{
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
