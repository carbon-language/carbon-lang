//===- X86Disassembler.cpp - Disassembler for x86 and x86_64 ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is part of the X86 Disassembler.
// It contains code to translate the data produced by the decoder into
//  MCInsts.
// Documentation for the disassembler can be found in X86Disassembler.h.
//
//===----------------------------------------------------------------------===//

#include "X86Disassembler.h"
#include "X86DisassemblerDecoder.h"

#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Target/TargetRegistry.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include "X86GenRegisterNames.inc"

using namespace llvm;
using namespace llvm::X86Disassembler;

namespace llvm {  
  
// Fill-ins to make the compiler happy.  These constants are never actually
//   assigned; they are just filler to make an automatically-generated switch
//   statement work.
namespace X86 {
  enum {
    BX_SI = 500,
    BX_DI = 501,
    BP_SI = 502,
    BP_DI = 503,
    sib   = 504,
    sib64 = 505
  };
}

extern Target TheX86_32Target, TheX86_64Target;

}

static void translateInstruction(MCInst &target,
                                 InternalInstruction &source);

X86GenericDisassembler::X86GenericDisassembler(DisassemblerMode mode) :
    MCDisassembler(),
    fMode(mode) {
}

X86GenericDisassembler::~X86GenericDisassembler() {
}

/// regionReader - a callback function that wraps the readByte method from
///   MemoryObject.
///
/// @param arg      - The generic callback parameter.  In this case, this should
///                   be a pointer to a MemoryObject.
/// @param byte     - A pointer to the byte to be read.
/// @param address  - The address to be read.
static int regionReader(void* arg, uint8_t* byte, uint64_t address) {
  MemoryObject* region = static_cast<MemoryObject*>(arg);
  return region->readByte(address, byte);
}

/// logger - a callback function that wraps the operator<< method from
///   raw_ostream.
///
/// @param arg      - The generic callback parameter.  This should be a pointe
///                   to a raw_ostream.
/// @param log      - A string to be logged.  logger() adds a newline.
static void logger(void* arg, const char* log) {
  if (!arg)
    return;
  
  raw_ostream &vStream = *(static_cast<raw_ostream*>(arg));
  vStream << log << "\n";
}  
  
//
// Public interface for the disassembler
//

bool X86GenericDisassembler::getInstruction(MCInst &instr,
                                            uint64_t &size,
                                            const MemoryObject &region,
                                            uint64_t address,
                                            raw_ostream &vStream) const {
  InternalInstruction internalInstr;
  
  int ret = decodeInstruction(&internalInstr,
                              regionReader,
                              (void*)&region,
                              logger,
                              (void*)&vStream,
                              address,
                              fMode);

  if(ret) {
    size = internalInstr.readerCursor - address;
    return false;
  }
  else {
    size = internalInstr.length;
    translateInstruction(instr, internalInstr);
    return true;
  }
}

//
// Private code that translates from struct InternalInstructions to MCInsts.
//

/// translateRegister - Translates an internal register to the appropriate LLVM
///   register, and appends it as an operand to an MCInst.
///
/// @param mcInst     - The MCInst to append to.
/// @param reg        - The Reg to append.
static void translateRegister(MCInst &mcInst, Reg reg) {
#define ENTRY(x) X86::x,
  uint8_t llvmRegnums[] = {
    ALL_REGS
    0
  };
#undef ENTRY

  uint8_t llvmRegnum = llvmRegnums[reg];
  mcInst.addOperand(MCOperand::CreateReg(llvmRegnum));
}

/// translateImmediate  - Appends an immediate operand to an MCInst.
///
/// @param mcInst       - The MCInst to append to.
/// @param immediate    - The immediate value to append.
static void translateImmediate(MCInst &mcInst, uint64_t immediate) {
  mcInst.addOperand(MCOperand::CreateImm(immediate));
}

/// translateRMRegister - Translates a register stored in the R/M field of the
///   ModR/M byte to its LLVM equivalent and appends it to an MCInst.
/// @param mcInst       - The MCInst to append to.
/// @param insn         - The internal instruction to extract the R/M field
///                       from.
static void translateRMRegister(MCInst &mcInst,
                                InternalInstruction &insn) {
  assert(insn.eaBase != EA_BASE_sib && insn.eaBase != EA_BASE_sib64 && 
         "A R/M register operand may not have a SIB byte");
  
  switch (insn.eaBase) {
  case EA_BASE_NONE:
    llvm_unreachable("EA_BASE_NONE for ModR/M base");
    break;
#define ENTRY(x) case EA_BASE_##x:
  ALL_EA_BASES
#undef ENTRY
    llvm_unreachable("A R/M register operand may not have a base; "
                     "the operand must be a register.");
    break;
#define ENTRY(x)                                                        \
  case EA_REG_##x:                                                    \
    mcInst.addOperand(MCOperand::CreateReg(X86::x)); break;
  ALL_REGS
#undef ENTRY
  default:
    llvm_unreachable("Unexpected EA base register");
  }
}

/// translateRMMemory - Translates a memory operand stored in the Mod and R/M
///   fields of an internal instruction (and possibly its SIB byte) to a memory
///   operand in LLVM's format, and appends it to an MCInst.
///
/// @param mcInst       - The MCInst to append to.
/// @param insn         - The instruction to extract Mod, R/M, and SIB fields
///                       from.
/// @param sr           - Whether or not to emit the segment register.  The
///                       LEA instruction does not expect a segment-register
///                       operand.
static void translateRMMemory(MCInst &mcInst,
                              InternalInstruction &insn,
                              bool sr) {
  // Addresses in an MCInst are represented as five operands:
  //   1. basereg       (register)  The R/M base, or (if there is a SIB) the 
  //                                SIB base
  //   2. scaleamount   (immediate) 1, or (if there is a SIB) the specified 
  //                                scale amount
  //   3. indexreg      (register)  x86_registerNONE, or (if there is a SIB)
  //                                the index (which is multiplied by the 
  //                                scale amount)
  //   4. displacement  (immediate) 0, or the displacement if there is one
  //   5. segmentreg    (register)  x86_registerNONE for now, but could be set
  //                                if we have segment overrides
  
  MCOperand baseReg;
  MCOperand scaleAmount;
  MCOperand indexReg;
  MCOperand displacement;
  MCOperand segmentReg;
  
  if (insn.eaBase == EA_BASE_sib || insn.eaBase == EA_BASE_sib64) {
    if (insn.sibBase != SIB_BASE_NONE) {
      switch (insn.sibBase) {
      default:
        llvm_unreachable("Unexpected sibBase");
#define ENTRY(x)                                          \
      case SIB_BASE_##x:                                  \
        baseReg = MCOperand::CreateReg(X86::x); break;
      ALL_SIB_BASES
#undef ENTRY
      }
    } else {
      baseReg = MCOperand::CreateReg(0);
    }
    
    if (insn.sibIndex != SIB_INDEX_NONE) {
      switch (insn.sibIndex) {
      default:
        llvm_unreachable("Unexpected sibIndex");
#define ENTRY(x)                                          \
      case SIB_INDEX_##x:                                 \
        indexReg = MCOperand::CreateReg(X86::x); break;
      EA_BASES_32BIT
      EA_BASES_64BIT
#undef ENTRY
      }
    } else {
      indexReg = MCOperand::CreateReg(0);
    }
    
    scaleAmount = MCOperand::CreateImm(insn.sibScale);
  } else {
    switch (insn.eaBase) {
    case EA_BASE_NONE:
      assert(insn.eaDisplacement != EA_DISP_NONE && 
             "EA_BASE_NONE and EA_DISP_NONE for ModR/M base");
      
      if (insn.mode == MODE_64BIT)
        baseReg = MCOperand::CreateReg(X86::RIP); // Section 2.2.1.6
      else
        baseReg = MCOperand::CreateReg(0);
      
      indexReg = MCOperand::CreateReg(0);
      break;
    case EA_BASE_BX_SI:
      baseReg = MCOperand::CreateReg(X86::BX);
      indexReg = MCOperand::CreateReg(X86::SI);
      break;
    case EA_BASE_BX_DI:
      baseReg = MCOperand::CreateReg(X86::BX);
      indexReg = MCOperand::CreateReg(X86::DI);
      break;
    case EA_BASE_BP_SI:
      baseReg = MCOperand::CreateReg(X86::BP);
      indexReg = MCOperand::CreateReg(X86::SI);
      break;
    case EA_BASE_BP_DI:
      baseReg = MCOperand::CreateReg(X86::BP);
      indexReg = MCOperand::CreateReg(X86::DI);
      break;
    default:
      indexReg = MCOperand::CreateReg(0);
      switch (insn.eaBase) {
      default:
        llvm_unreachable("Unexpected eaBase");
        break;
        // Here, we will use the fill-ins defined above.  However,
        //   BX_SI, BX_DI, BP_SI, and BP_DI are all handled above and
        //   sib and sib64 were handled in the top-level if, so they're only
        //   placeholders to keep the compiler happy.
#define ENTRY(x)                                        \
      case EA_BASE_##x:                                 \
        baseReg = MCOperand::CreateReg(X86::x); break; 
      ALL_EA_BASES
#undef ENTRY
#define ENTRY(x) case EA_REG_##x:
      ALL_REGS
#undef ENTRY
        llvm_unreachable("A R/M memory operand may not be a register; "
                         "the base field must be a base.");
            break;
      }
    }
    
    scaleAmount = MCOperand::CreateImm(1);
  }
  
  displacement = MCOperand::CreateImm(insn.displacement);
  
  static const uint8_t segmentRegnums[SEG_OVERRIDE_max] = {
    0,        // SEG_OVERRIDE_NONE
    X86::CS,
    X86::SS,
    X86::DS,
    X86::ES,
    X86::FS,
    X86::GS
  };
  
  segmentReg = MCOperand::CreateReg(segmentRegnums[insn.segmentOverride]);
  
  mcInst.addOperand(baseReg);
  mcInst.addOperand(scaleAmount);
  mcInst.addOperand(indexReg);
  mcInst.addOperand(displacement);
  
  if (sr)
    mcInst.addOperand(segmentReg);
}

/// translateRM - Translates an operand stored in the R/M (and possibly SIB)
///   byte of an instruction to LLVM form, and appends it to an MCInst.
///
/// @param mcInst       - The MCInst to append to.
/// @param operand      - The operand, as stored in the descriptor table.
/// @param insn         - The instruction to extract Mod, R/M, and SIB fields
///                       from.
static void translateRM(MCInst &mcInst,
                        OperandSpecifier &operand,
                        InternalInstruction &insn) {
  switch (operand.type) {
  default:
    llvm_unreachable("Unexpected type for a R/M operand");
  case TYPE_R8:
  case TYPE_R16:
  case TYPE_R32:
  case TYPE_R64:
  case TYPE_Rv:
  case TYPE_MM:
  case TYPE_MM32:
  case TYPE_MM64:
  case TYPE_XMM:
  case TYPE_XMM32:
  case TYPE_XMM64:
  case TYPE_XMM128:
  case TYPE_DEBUGREG:
  case TYPE_CR32:
  case TYPE_CR64:
    translateRMRegister(mcInst, insn);
    break;
  case TYPE_M:
  case TYPE_M8:
  case TYPE_M16:
  case TYPE_M32:
  case TYPE_M64:
  case TYPE_M128:
  case TYPE_M512:
  case TYPE_Mv:
  case TYPE_M32FP:
  case TYPE_M64FP:
  case TYPE_M80FP:
  case TYPE_M16INT:
  case TYPE_M32INT:
  case TYPE_M64INT:
  case TYPE_M1616:
  case TYPE_M1632:
  case TYPE_M1664:
    translateRMMemory(mcInst, insn, true);
    break;
  case TYPE_LEA:
    translateRMMemory(mcInst, insn, false);
    break;
  }
}
  
/// translateFPRegister - Translates a stack position on the FPU stack to its
///   LLVM form, and appends it to an MCInst.
///
/// @param mcInst       - The MCInst to append to.
/// @param stackPos     - The stack position to translate.
static void translateFPRegister(MCInst &mcInst,
                                uint8_t stackPos) {
  assert(stackPos < 8 && "Invalid FP stack position");
  
  mcInst.addOperand(MCOperand::CreateReg(X86::ST0 + stackPos));
}

/// translateOperand - Translates an operand stored in an internal instruction 
///   to LLVM's format and appends it to an MCInst.
///
/// @param mcInst       - The MCInst to append to.
/// @param operand      - The operand, as stored in the descriptor table.
/// @param insn         - The internal instruction.
static void translateOperand(MCInst &mcInst,
                             OperandSpecifier &operand,
                             InternalInstruction &insn) {
  switch (operand.encoding) {
  default:
    llvm_unreachable("Unhandled operand encoding during translation");
  case ENCODING_REG:
    translateRegister(mcInst, insn.reg);
    break;
  case ENCODING_RM:
    translateRM(mcInst, operand, insn);
    break;
  case ENCODING_CB:
  case ENCODING_CW:
  case ENCODING_CD:
  case ENCODING_CP:
  case ENCODING_CO:
  case ENCODING_CT:
    llvm_unreachable("Translation of code offsets isn't supported.");
  case ENCODING_IB:
  case ENCODING_IW:
  case ENCODING_ID:
  case ENCODING_IO:
  case ENCODING_Iv:
  case ENCODING_Ia:
    translateImmediate(mcInst, 
                       insn.immediates[insn.numImmediatesTranslated++]);
    break;
  case ENCODING_RB:
  case ENCODING_RW:
  case ENCODING_RD:
  case ENCODING_RO:
    translateRegister(mcInst, insn.opcodeRegister);
    break;
  case ENCODING_I:
    translateFPRegister(mcInst, insn.opcodeModifier);
    break;
  case ENCODING_Rv:
    translateRegister(mcInst, insn.opcodeRegister);
    break;
  case ENCODING_DUP:
    translateOperand(mcInst,
                     insn.spec->operands[operand.type - TYPE_DUP0],
                     insn);
    break;
  }
}
  
/// translateInstruction - Translates an internal instruction and all its
///   operands to an MCInst.
///
/// @param mcInst       - The MCInst to populate with the instruction's data.
/// @param insn         - The internal instruction.
static void translateInstruction(MCInst &mcInst,
                                 InternalInstruction &insn) {  
  assert(insn.spec);
  
  mcInst.setOpcode(insn.instructionID);
  
  int index;
  
  insn.numImmediatesTranslated = 0;
  
  for (index = 0; index < X86_MAX_OPERANDS; ++index) {
    if (insn.spec->operands[index].encoding != ENCODING_NONE)                
      translateOperand(mcInst, insn.spec->operands[index], insn);
  }
}

static const MCDisassembler *createX86_32Disassembler(const Target &T) {
  return new X86Disassembler::X86_32Disassembler;
}

static const MCDisassembler *createX86_64Disassembler(const Target &T) {
  return new X86Disassembler::X86_64Disassembler;
}

extern "C" void LLVMInitializeX86Disassembler() { 
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(TheX86_32Target, 
                                         createX86_32Disassembler);
  TargetRegistry::RegisterMCDisassembler(TheX86_64Target,
                                         createX86_64Disassembler);
}
