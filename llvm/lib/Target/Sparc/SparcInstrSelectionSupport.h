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
  case Type::UByteTyID:   return V9::LDUB;
  case Type::SByteTyID:   return V9::LDSB;
  case Type::UShortTyID:  return V9::LDUH;
  case Type::ShortTyID:   return V9::LDSH;
  case Type::UIntTyID:    return V9::LDUW;
  case Type::IntTyID:     return V9::LDSW;
  case Type::PointerTyID:
  case Type::ULongTyID:
  case Type::LongTyID:    return V9::LDX;
  case Type::FloatTyID:   return V9::LD;
  case Type::DoubleTyID:  return V9::LDD;
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
  case Type::SByteTyID:   return V9::STB;
  case Type::UShortTyID:
  case Type::ShortTyID:   return V9::STH;
  case Type::UIntTyID:
  case Type::IntTyID:     return V9::STW;
  case Type::PointerTyID:
  case Type::ULongTyID:
  case Type::LongTyID:    return V9::STX;
  case Type::FloatTyID:   return V9::ST;
  case Type::DoubleTyID:  return V9::STD;
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
    opCode = V9::ADD;
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
