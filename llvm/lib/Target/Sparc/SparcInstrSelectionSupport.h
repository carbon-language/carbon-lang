// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	SparcInstrSelectionSupport.h
// 
// Purpose:
// 
// History:
//	10/17/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#ifndef SPARC_INSTR_SELECTION_SUPPORT_h
#define SPARC_INSTR_SELECTION_SUPPORT_h

#include "llvm/DerivedTypes.h"
#include "llvm/Value.h"

inline MachineOpCode
ChooseLoadInstruction(const Type *DestTy)
{
  switch (DestTy->getPrimitiveID()) {
  case Type::BoolTyID:
  case Type::UByteTyID:   return LDUB;
  case Type::SByteTyID:   return LDSB;
  case Type::UShortTyID:  return LDUH;
  case Type::ShortTyID:   return LDSH;
  case Type::UIntTyID:    return LDUW;
  case Type::IntTyID:     return LDSW;
  case Type::PointerTyID:
  case Type::ULongTyID:
  case Type::LongTyID:    return LDX;
  case Type::FloatTyID:   return LD;
  case Type::DoubleTyID:  return LDD;
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
  case Type::SByteTyID:   return STB;
  case Type::UShortTyID:
  case Type::ShortTyID:   return STH;
  case Type::UIntTyID:
  case Type::IntTyID:     return STW;
  case Type::PointerTyID:
  case Type::ULongTyID:
  case Type::LongTyID:    return STX;
  case Type::FloatTyID:   return ST;
  case Type::DoubleTyID:  return STD;
  default: assert(0 && "Invalid type for Store instruction");
  }
  
  return 0;
}


inline MachineOpCode 
ChooseAddInstructionByType(const Type* resultType)
{
  MachineOpCode opCode = INVALID_OPCODE;
  
  if (resultType->isIntegral() ||
      isa<PointerType>(resultType) ||
      isa<FunctionType>(resultType) ||
      resultType == Type::LabelTy ||
      resultType == Type::BoolTy)
    {
      opCode = ADD;
    }
  else
    switch(resultType->getPrimitiveID())
      {
      case Type::FloatTyID:  opCode = FADDS; break;
      case Type::DoubleTyID: opCode = FADDD; break;
      default: assert(0 && "Invalid type for ADD instruction"); break; 
      }
  
  return opCode;
}

#endif
