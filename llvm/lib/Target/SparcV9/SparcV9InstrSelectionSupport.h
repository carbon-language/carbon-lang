//===-- SparcV9InstrSelectionSupport.h --------------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// More instruction selection support routines for the SparcV9 target.
//
//===----------------------------------------------------------------------===//

#ifndef SPARCV9INSTRSELECTIONSUPPORT_H
#define SPARCV9INSTRSELECTIONSUPPORT_H

#include "llvm/DerivedTypes.h"
#include "SparcV9Internals.h"

namespace llvm {

// Choose load instruction opcode based on type of value
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

// Choose store instruction opcode based on type of value
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


// Because the SparcV9 instruction selector likes to re-write operands to
// instructions, making them change from a Value* (virtual register) to a
// Constant* (making an immediate field), we need to change the opcode from a
// register-based instruction to an immediate-based instruction, hence this
// mapping.
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
  case V9::SLLr5:   return V9::SLLi5;
  case V9::SRLr5:   return V9::SRLi5;
  case V9::SRAr5:   return V9::SRAi5;
  case V9::SLLXr6:  return V9::SLLXi6;
  case V9::SRLXr6:  return V9::SRLXi6;
  case V9::SRAXr6:  return V9::SRAXi6;

    /* Conditional move on int comparison with zero */
  case V9::MOVRZr:   return V9::MOVRZi;
  case V9::MOVRLEZr: return V9::MOVRLEZi;
  case V9::MOVRLZr:  return V9::MOVRLZi;
  case V9::MOVRNZr:  return V9::MOVRNZi;
  case V9::MOVRGZr:  return V9::MOVRGZi;
  case V9::MOVRGEZr: return V9::MOVRGEZi;


    /* Conditional move on int condition code */
  case V9::MOVAr:   return V9::MOVAi;
  case V9::MOVNr:   return V9::MOVNi;
  case V9::MOVNEr:  return V9::MOVNEi;
  case V9::MOVEr:   return V9::MOVEi;
  case V9::MOVGr:   return V9::MOVGi;
  case V9::MOVLEr:  return V9::MOVLEi;
  case V9::MOVGEr:  return V9::MOVGEi;
  case V9::MOVLr:   return V9::MOVLi;
  case V9::MOVGUr:  return V9::MOVGUi;
  case V9::MOVLEUr: return V9::MOVLEUi;
  case V9::MOVCCr:  return V9::MOVCCi;
  case V9::MOVCSr:  return V9::MOVCSi;
  case V9::MOVPOSr: return V9::MOVPOSi;
  case V9::MOVNEGr: return V9::MOVNEGi;
  case V9::MOVVCr:  return V9::MOVVCi;
  case V9::MOVVSr:  return V9::MOVVSi;

    /* Conditional move of int reg on fp condition code */
  case V9::MOVFAr:   return V9::MOVFAi;
  case V9::MOVFNr:   return V9::MOVFNi;
  case V9::MOVFUr:   return V9::MOVFUi;
  case V9::MOVFGr:   return V9::MOVFGi;
  case V9::MOVFUGr:  return V9::MOVFUGi;
  case V9::MOVFLr:   return V9::MOVFLi;
  case V9::MOVFULr:  return V9::MOVFULi;
  case V9::MOVFLGr:  return V9::MOVFLGi;
  case V9::MOVFNEr:  return V9::MOVFNEi;
  case V9::MOVFEr:   return V9::MOVFEi;
  case V9::MOVFUEr:  return V9::MOVFUEi;
  case V9::MOVFGEr:  return V9::MOVFGEi;
  case V9::MOVFUGEr: return V9::MOVFUGEi;
  case V9::MOVFLEr:  return V9::MOVFLEi;
  case V9::MOVFULEr: return V9::MOVFULEi;
  case V9::MOVFOr:   return V9::MOVFOi;

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
    // Or, it's just not handled yet, but an assert() would break LLC
#if 0
    std::cerr << "Unhandled opcode in convertOpcodeFromRegToImm(): " << Opcode 
              << "\n";
#endif
    return Opcode;
  }
}

MachineOperand::MachineOperandType
ChooseRegOrImmed(Value* val, MachineOpCode opCode,
                 const TargetMachine& targetMachine, bool canUseImmed,
                 unsigned& getMachineRegNum, int64_t& getImmedValue);

} // End llvm namespace

#endif
