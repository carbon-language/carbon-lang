//===-- MachineInstrAnnot.cpp ---------------------------------------------===//
// 
//  This file defines Annotations used to pass information between code
//  generation phases.
// 
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineInstrAnnot.h"
#include "llvm/Annotation.h"
#include "llvm/iOther.h"
#include "llvm/Type.h"

AnnotationID CallArgsDescriptor::AID(AnnotationManager::
                                     getID("CodeGen::CallArgsDescriptor"));

CallArgsDescriptor::CallArgsDescriptor(const CallInst* _callInstr,
                                       TmpInstruction* _retAddrReg,
                                       bool _isVarArgs, bool _noPrototype)
  : Annotation(AID),
    callInstr(_callInstr),
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
}


const CallInst*
CallArgsDescriptor::getReturnValue() const
{
  return (callInstr->getType() == Type::VoidTy? NULL : callInstr);
}
