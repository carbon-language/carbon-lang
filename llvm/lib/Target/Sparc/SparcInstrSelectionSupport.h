//===-- llvm/CodeGen/SparcInstrSelectionSupport.h ---------------*- C++ -*-===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef SPARC_INSTR_SELECTION_SUPPORT_h
#define SPARC_INSTR_SELECTION_SUPPORT_h

#include "llvm/DerivedTypes.h"

inline MachineOpCode
ChooseLoadInstruction(const Type *DestTy)
{
  switch (DestTy->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:   return V9::LDUBr;
  case Type::SByteTyID:   return V9::LDSBr;
  case Type::UShortTyID:  return V9::LDUHr;
  case Type::ShortTyID:   return V9::LDSHr;
  case Type::UIntTyID:    return V9::LDUWr;
  case Type::IntTyID:     return V9::LDSWr;
  case Type::PointerTyID:
  case Type::ULongTyID:
  case Type::LongTyID:    return V9::LDXr;
  case Type::FloatTyID:   return V9::LDFr;
  case Type::DoubleTyID:  return V9::LDDFr;
  default: assert(0 && "Invalid type for Load instruction");
  }
  
  return 0;
}

inline MachineOpCode
ChooseStoreInstruction(const Type *DestTy)
{
  switch (DestTy->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:
  case Type::SByteTyID:   return V9::STBr;
  case Type::UShortTyID:
  case Type::ShortTyID:   return V9::STHr;
  case Type::UIntTyID:
  case Type::IntTyID:     return V9::STWr;
  case Type::PointerTyID:
  case Type::ULongTyID:
  case Type::LongTyID:    return V9::STXr;
  case Type::FloatTyID:   return V9::STFr;
  case Type::DoubleTyID:  return V9::STDFr;
  default: assert(0 && "Invalid type for Store instruction");
  }
  
  return 0;
}


inline MachineOpCode 
ChooseAddInstructionByType(const Type* resultType)
{
  MachineOpCode opCode = V9::INVALID_OPCODE;
  
  if (resultType->isIntegral() ||
      isa<PointerType>(resultType) ||
      isa<FunctionType>(resultType) ||
      resultType == Type::LabelTy)
  {
    opCode = V9::ADDr;
  }
  else
    switch(resultType->getPrimitiveID())
    {
    case Type::FloatTyID:  opCode = V9::FADDS; break;
    case Type::DoubleTyID: opCode = V9::FADDD; break;
    default: assert(0 && "Invalid type for ADD instruction"); break; 
    }
  
  return opCode;
}

#endif
