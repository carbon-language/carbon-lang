//===-- llvm/CodeGen/SparcInstrSelectionSupport.h ---------------*- C++ -*-===//
//
//
//
//===----------------------------------------------------------------------===//

#ifndef SPARC_INSTR_SELECTION_SUPPORT_h
#define SPARC_INSTR_SELECTION_SUPPORT_h

#include "llvm/DerivedTypes.h"
#include "SparcInternals.h"

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


static unsigned
convertOpcodeFromRegToImm(unsigned Opcode) {
  switch (Opcode) {
    /* arithmetic */
  case V9::ADDr:     return V9::ADDi;
  case V9::ADDccr:   return V9::ADDcci;
  case V9::ADDCr:    return V9::ADDCi;
  case V9::ADDCccr:  return V9::ADDCcci;
  case V9::SUBr:     return V9::SUBi;
  case V9::SUBccr:   return V9::SUBcci;
  case V9::SUBCr:    return V9::SUBCi;
  case V9::SUBCccr:  return V9::SUBCcci;
  case V9::MULXr:    return V9::MULXi;
  case V9::SDIVXr:   return V9::SDIVXi;
  case V9::UDIVXr:   return V9::UDIVXi;

    /* logical */
  case V9::ANDr:    return V9::ANDi;
  case V9::ANDccr:  return V9::ANDcci;
  case V9::ANDNr:   return V9::ANDNi;
  case V9::ANDNccr: return V9::ANDNcci;
  case V9::ORr:     return V9::ORi;
  case V9::ORccr:   return V9::ORcci;
  case V9::ORNr:    return V9::ORNi;
  case V9::ORNccr:  return V9::ORNcci;
  case V9::XORr:    return V9::XORi;
  case V9::XORccr:  return V9::XORcci;
  case V9::XNORr:   return V9::XNORi;
  case V9::XNORccr: return V9::XNORcci;

    /* shift */
  case V9::SLLr6:   return V9::SLLi6;
  case V9::SRLr6:   return V9::SRLi6;
  case V9::SRAr6:   return V9::SRAi6;
  case V9::SLLXr6:  return V9::SLLXi6;
  case V9::SRLXr6:  return V9::SRLXi6;
  case V9::SRAXr6:  return V9::SRAXi6;

    /* load */
  case V9::LDSBr:   return V9::LDSBi;
  case V9::LDSHr:   return V9::LDSHi;
  case V9::LDSWr:   return V9::LDSWi;
  case V9::LDUBr:   return V9::LDUBi;
  case V9::LDUHr:   return V9::LDUHi;
  case V9::LDUWr:   return V9::LDUWi;
  case V9::LDXr:    return V9::LDXi;
  case V9::LDFr:    return V9::LDFi;
  case V9::LDDFr:   return V9::LDDFi;
  case V9::LDQFr:   return V9::LDQFi;
  case V9::LDFSRr:  return V9::LDFSRi;
  case V9::LDXFSRr: return V9::LDXFSRi;

    /* store */
  case V9::STBr:    return V9::STBi;
  case V9::STHr:    return V9::STHi;
  case V9::STWr:    return V9::STWi;
  case V9::STXr:    return V9::STXi;
  case V9::STFr:    return V9::STFi;
  case V9::STDFr:   return V9::STDFi;
  case V9::STFSRr:  return V9::STFSRi;
  case V9::STXFSRr: return V9::STXFSRi;

    /* jump & return */
  case V9::JMPLCALLr: return V9::JMPLCALLi;
  case V9::JMPLRETr:  return V9::JMPLRETi;
  case V9::RETURNr:   return V9::RETURNi;

  /* save and restore */
  case V9::SAVEr:     return V9::SAVEi;
  case V9::RESTOREr:  return V9::RESTOREi;

  default:
    // It's already in correct format
    return Opcode;
  }
}

#endif
