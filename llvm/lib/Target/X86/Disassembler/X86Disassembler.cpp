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

#include "llvm/MC/EDInstInfo.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCDisassembler.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryObject.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"

#define GET_REGINFO_ENUM
#include "X86GenRegisterInfo.inc"
#define GET_INSTRINFO_ENUM
#include "X86GenInstrInfo.inc"
#include "X86GenEDInfo.inc"

using namespace llvm;
using namespace llvm::X86Disassembler;

void x86DisassemblerDebug(const char *file,
                          unsigned line,
                          const char *s) {
  dbgs() << file << ":" << line << ": " << s;
}

#define debug(s) DEBUG(x86DisassemblerDebug(__FILE__, __LINE__, s));

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

static bool translateInstruction(MCInst &target,
                                InternalInstruction &source);

X86GenericDisassembler::X86GenericDisassembler(const MCSubtargetInfo &STI, DisassemblerMode mode) :
    MCDisassembler(STI),
    fMode(mode) {
}

X86GenericDisassembler::~X86GenericDisassembler() {
}

EDInstInfo *X86GenericDisassembler::getEDInfo() const {
  return instInfoX86;
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

MCDisassembler::DecodeStatus
X86GenericDisassembler::getInstruction(MCInst &instr,
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

  if (ret) {
    size = internalInstr.readerCursor - address;
    return Fail;
  }
  else {
    size = internalInstr.length;
    return (!translateInstruction(instr, internalInstr)) ? Success : Fail;
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
/// @param operand      - The operand, as stored in the descriptor table.
/// @param insn         - The internal instruction.
static void translateImmediate(MCInst &mcInst, uint64_t immediate,
                               const OperandSpecifier &operand,
                               InternalInstruction &insn) {
  // Sign-extend the immediate if necessary.

  OperandType type = operand.type;

  if (type == TYPE_RELv) {
    switch (insn.displacementSize) {
    default:
      break;
    case 1:
      type = TYPE_MOFFS8;
      break;
    case 2:
      type = TYPE_MOFFS16;
      break;
    case 4:
      type = TYPE_MOFFS32;
      break;
    case 8:
      type = TYPE_MOFFS64;
      break;
    }
  }
  // By default sign-extend all X86 immediates based on their encoding.
  else if (type == TYPE_IMM8 || type == TYPE_IMM16 || type == TYPE_IMM32 ||
           type == TYPE_IMM64) {
    uint32_t Opcode = mcInst.getOpcode();
    switch (operand.encoding) {
    default:
      break;
    case ENCODING_IB:
      // Special case those X86 instructions that use the imm8 as a set of
      // bits, bit count, etc. and are not sign-extend.
      if (Opcode != X86::BLENDPSrri && Opcode != X86::BLENDPDrri &&
	  Opcode != X86::PBLENDWrri && Opcode != X86::MPSADBWrri &&
	  Opcode != X86::DPPSrri && Opcode != X86::DPPDrri &&
	  Opcode != X86::INSERTPSrr && Opcode != X86::VBLENDPSYrri &&
	  Opcode != X86::VBLENDPSYrmi && Opcode != X86::VBLENDPDYrri &&
	  Opcode != X86::VBLENDPDYrmi && Opcode != X86::VPBLENDWrri &&
	  Opcode != X86::VMPSADBWrri && Opcode != X86::VDPPSYrri &&
	  Opcode != X86::VDPPSYrmi && Opcode != X86::VDPPDrri &&
	  Opcode != X86::VINSERTPSrr)
	type = TYPE_MOFFS8;
      break;
    case ENCODING_IW:
      type = TYPE_MOFFS16;
      break;
    case ENCODING_ID:
      type = TYPE_MOFFS32;
      break;
    case ENCODING_IO:
      type = TYPE_MOFFS64;
      break;
    }
  }

  switch (type) {
  case TYPE_MOFFS8:
  case TYPE_REL8:
    if(immediate & 0x80)
      immediate |= ~(0xffull);
    break;
  case TYPE_MOFFS16:
    if(immediate & 0x8000)
      immediate |= ~(0xffffull);
    break;
  case TYPE_MOFFS32:
  case TYPE_REL32:
  case TYPE_REL64:
    if(immediate & 0x80000000)
      immediate |= ~(0xffffffffull);
    break;
  case TYPE_MOFFS64:
  default:
    // operand is 64 bits wide.  Do nothing.
    break;
  }
    
  mcInst.addOperand(MCOperand::CreateImm(immediate));
}

/// translateRMRegister - Translates a register stored in the R/M field of the
///   ModR/M byte to its LLVM equivalent and appends it to an MCInst.
/// @param mcInst       - The MCInst to append to.
/// @param insn         - The internal instruction to extract the R/M field
///                       from.
/// @return             - 0 on success; -1 otherwise
static bool translateRMRegister(MCInst &mcInst,
                                InternalInstruction &insn) {
  if (insn.eaBase == EA_BASE_sib || insn.eaBase == EA_BASE_sib64) {
    debug("A R/M register operand may not have a SIB byte");
    return true;
  }
  
  switch (insn.eaBase) {
  default:
    debug("Unexpected EA base register");
    return true;
  case EA_BASE_NONE:
    debug("EA_BASE_NONE for ModR/M base");
    return true;
#define ENTRY(x) case EA_BASE_##x:
  ALL_EA_BASES
#undef ENTRY
    debug("A R/M register operand may not have a base; "
          "the operand must be a register.");
    return true;
#define ENTRY(x)                                                      \
  case EA_REG_##x:                                                    \
    mcInst.addOperand(MCOperand::CreateReg(X86::x)); break;
  ALL_REGS
#undef ENTRY
  }
  
  return false;
}

/// translateRMMemory - Translates a memory operand stored in the Mod and R/M
///   fields of an internal instruction (and possibly its SIB byte) to a memory
///   operand in LLVM's format, and appends it to an MCInst.
///
/// @param mcInst       - The MCInst to append to.
/// @param insn         - The instruction to extract Mod, R/M, and SIB fields
///                       from.
/// @return             - 0 on success; nonzero otherwise
static bool translateRMMemory(MCInst &mcInst, InternalInstruction &insn) {
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
        debug("Unexpected sibBase");
        return true;
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
        debug("Unexpected sibIndex");
        return true;
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
      if (insn.eaDisplacement == EA_DISP_NONE) {
        debug("EA_BASE_NONE and EA_DISP_NONE for ModR/M base");
        return true;
      }
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
        debug("Unexpected eaBase");
        return true;
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
        debug("A R/M memory operand may not be a register; "
              "the base field must be a base.");
        return true;
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
  mcInst.addOperand(segmentReg);
  return false;
}

/// translateRM - Translates an operand stored in the R/M (and possibly SIB)
///   byte of an instruction to LLVM form, and appends it to an MCInst.
///
/// @param mcInst       - The MCInst to append to.
/// @param operand      - The operand, as stored in the descriptor table.
/// @param insn         - The instruction to extract Mod, R/M, and SIB fields
///                       from.
/// @return             - 0 on success; nonzero otherwise
static bool translateRM(MCInst &mcInst, const OperandSpecifier &operand,
                        InternalInstruction &insn) {
  switch (operand.type) {
  default:
    debug("Unexpected type for a R/M operand");
    return true;
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
  case TYPE_XMM256:
  case TYPE_DEBUGREG:
  case TYPE_CONTROLREG:
    return translateRMRegister(mcInst, insn);
  case TYPE_M:
  case TYPE_M8:
  case TYPE_M16:
  case TYPE_M32:
  case TYPE_M64:
  case TYPE_M128:
  case TYPE_M256:
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
  case TYPE_LEA:
    return translateRMMemory(mcInst, insn);
  }
}
  
/// translateFPRegister - Translates a stack position on the FPU stack to its
///   LLVM form, and appends it to an MCInst.
///
/// @param mcInst       - The MCInst to append to.
/// @param stackPos     - The stack position to translate.
/// @return             - 0 on success; nonzero otherwise.
static bool translateFPRegister(MCInst &mcInst,
                               uint8_t stackPos) {
  if (stackPos >= 8) {
    debug("Invalid FP stack position");
    return true;
  }
  
  mcInst.addOperand(MCOperand::CreateReg(X86::ST0 + stackPos));

  return false;
}

/// translateOperand - Translates an operand stored in an internal instruction 
///   to LLVM's format and appends it to an MCInst.
///
/// @param mcInst       - The MCInst to append to.
/// @param operand      - The operand, as stored in the descriptor table.
/// @param insn         - The internal instruction.
/// @return             - false on success; true otherwise.
static bool translateOperand(MCInst &mcInst, const OperandSpecifier &operand,
                             InternalInstruction &insn) {
  switch (operand.encoding) {
  default:
    debug("Unhandled operand encoding during translation");
    return true;
  case ENCODING_REG:
    translateRegister(mcInst, insn.reg);
    return false;
  case ENCODING_RM:
    return translateRM(mcInst, operand, insn);
  case ENCODING_CB:
  case ENCODING_CW:
  case ENCODING_CD:
  case ENCODING_CP:
  case ENCODING_CO:
  case ENCODING_CT:
    debug("Translation of code offsets isn't supported.");
    return true;
  case ENCODING_IB:
  case ENCODING_IW:
  case ENCODING_ID:
  case ENCODING_IO:
  case ENCODING_Iv:
  case ENCODING_Ia:
    translateImmediate(mcInst,
                       insn.immediates[insn.numImmediatesTranslated++],
                       operand,
                       insn);
    return false;
  case ENCODING_RB:
  case ENCODING_RW:
  case ENCODING_RD:
  case ENCODING_RO:
    translateRegister(mcInst, insn.opcodeRegister);
    return false;
  case ENCODING_I:
    return translateFPRegister(mcInst, insn.opcodeModifier);
  case ENCODING_Rv:
    translateRegister(mcInst, insn.opcodeRegister);
    return false;
  case ENCODING_VVVV:
    translateRegister(mcInst, insn.vvvv);
    return false;
  case ENCODING_DUP:
    return translateOperand(mcInst,
                            insn.spec->operands[operand.type - TYPE_DUP0],
                            insn);
  }
}
  
/// translateInstruction - Translates an internal instruction and all its
///   operands to an MCInst.
///
/// @param mcInst       - The MCInst to populate with the instruction's data.
/// @param insn         - The internal instruction.
/// @return             - false on success; true otherwise.
static bool translateInstruction(MCInst &mcInst,
                                InternalInstruction &insn) {  
  if (!insn.spec) {
    debug("Instruction has no specification");
    return true;
  }
  
  mcInst.setOpcode(insn.instructionID);
  
  int index;
  
  insn.numImmediatesTranslated = 0;
  
  for (index = 0; index < X86_MAX_OPERANDS; ++index) {
    if (insn.spec->operands[index].encoding != ENCODING_NONE) {
      if (translateOperand(mcInst, insn.spec->operands[index], insn)) {
        return true;
      }
    }
  }
  
  return false;
}

static MCDisassembler *createX86_32Disassembler(const Target &T, const MCSubtargetInfo &STI) {
  return new X86Disassembler::X86_32Disassembler(STI);
}

static MCDisassembler *createX86_64Disassembler(const Target &T, const MCSubtargetInfo &STI) {
  return new X86Disassembler::X86_64Disassembler(STI);
}

extern "C" void LLVMInitializeX86Disassembler() { 
  // Register the disassembler.
  TargetRegistry::RegisterMCDisassembler(TheX86_32Target, 
                                         createX86_32Disassembler);
  TargetRegistry::RegisterMCDisassembler(TheX86_64Target,
                                         createX86_64Disassembler);
}
