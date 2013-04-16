//===-- ARMUnwindOpAsm.cpp - ARM Unwind Opcodes Assembler -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the unwind opcode assmebler for ARM exception handling
// table.
//
//===----------------------------------------------------------------------===//

#include "ARMUnwindOpAsm.h"

#include "ARMUnwindOp.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;

void UnwindOpcodeAssembler::EmitRegSave(uint32_t RegSave) {
  if (RegSave == 0u)
    return;

  // One byte opcode to save register r14 and r11-r4
  if (RegSave & (1u << 4)) {
    // The one byte opcode will always save r4, thus we can't use the one byte
    // opcode when r4 is not in .save directive.

    // Compute the consecutive registers from r4 to r11.
    uint32_t Range = 0;
    uint32_t Mask = (1u << 4);
    for (uint32_t Bit = (1u << 5); Bit < (1u << 12); Bit <<= 1) {
      if ((RegSave & Bit) == 0u)
        break;
      ++Range;
      Mask |= Bit;
    }

    // Emit this opcode when the mask covers every registers.
    uint32_t UnmaskedReg = RegSave & 0xfff0u & (~Mask);
    if (UnmaskedReg == 0u) {
      // Pop r[4 : (4 + n)]
      Ops.push_back(UNWIND_OPCODE_POP_REG_RANGE_R4 | Range);
      RegSave &= 0x000fu;
    } else if (UnmaskedReg == (1u << 14)) {
      // Pop r[14] + r[4 : (4 + n)]
      Ops.push_back(UNWIND_OPCODE_POP_REG_RANGE_R4_R14 | Range);
      RegSave &= 0x000fu;
    }
  }

  // Two bytes opcode to save register r15-r4
  if ((RegSave & 0xfff0u) != 0) {
    uint32_t Op = UNWIND_OPCODE_POP_REG_MASK_R4 | (RegSave >> 4);
    Ops.push_back(static_cast<uint8_t>(Op >> 8));
    Ops.push_back(static_cast<uint8_t>(Op & 0xff));
  }

  // Opcode to save register r3-r0
  if ((RegSave & 0x000fu) != 0) {
    uint32_t Op = UNWIND_OPCODE_POP_REG_MASK | (RegSave & 0x000fu);
    Ops.push_back(static_cast<uint8_t>(Op >> 8));
    Ops.push_back(static_cast<uint8_t>(Op & 0xff));
  }
}

/// Emit unwind opcodes for .vsave directives
void UnwindOpcodeAssembler::EmitVFPRegSave(uint32_t VFPRegSave) {
  size_t i = 32;

  while (i > 16) {
    uint32_t Bit = 1u << (i - 1);
    if ((VFPRegSave & Bit) == 0u) {
      --i;
      continue;
    }

    uint32_t Range = 0;

    --i;
    Bit >>= 1;

    while (i > 16 && (VFPRegSave & Bit)) {
      --i;
      ++Range;
      Bit >>= 1;
    }

    uint32_t Op =
        UNWIND_OPCODE_POP_VFP_REG_RANGE_FSTMFDD_D16 | ((i - 16) << 4) | Range;
    Ops.push_back(static_cast<uint8_t>(Op >> 8));
    Ops.push_back(static_cast<uint8_t>(Op & 0xff));
  }

  while (i > 0) {
    uint32_t Bit = 1u << (i - 1);
    if ((VFPRegSave & Bit) == 0u) {
      --i;
      continue;
    }

    uint32_t Range = 0;

    --i;
    Bit >>= 1;

    while (i > 0 && (VFPRegSave & Bit)) {
      --i;
      ++Range;
      Bit >>= 1;
    }

    uint32_t Op = UNWIND_OPCODE_POP_VFP_REG_RANGE_FSTMFDD | (i << 4) | Range;
    Ops.push_back(static_cast<uint8_t>(Op >> 8));
    Ops.push_back(static_cast<uint8_t>(Op & 0xff));
  }
}

/// Emit unwind opcodes for .setfp directives
void UnwindOpcodeAssembler::EmitSetFP(uint16_t FPReg) {
  Ops.push_back(UNWIND_OPCODE_SET_VSP | FPReg);
}

/// Emit unwind opcodes to update stack pointer
void UnwindOpcodeAssembler::EmitSPOffset(int64_t Offset) {
  if (Offset > 0x200) {
    uint8_t Buff[10];
    size_t Size = encodeULEB128((Offset - 0x204) >> 2, Buff);
    Ops.push_back(UNWIND_OPCODE_INC_VSP_ULEB128);
    Ops.append(Buff, Buff + Size);
  } else if (Offset > 0) {
    if (Offset > 0x100) {
      Ops.push_back(UNWIND_OPCODE_INC_VSP | 0x3fu);
      Offset -= 0x100;
    }
    Ops.push_back(UNWIND_OPCODE_INC_VSP |
                  static_cast<uint8_t>((Offset - 4) >> 2));
  } else if (Offset < 0) {
    while (Offset < -0x100) {
      Ops.push_back(UNWIND_OPCODE_DEC_VSP | 0x3fu);
      Offset += 0x100;
    }
    Ops.push_back(UNWIND_OPCODE_DEC_VSP |
                  static_cast<uint8_t>(((-Offset) - 4) >> 2));
  }
}

void UnwindOpcodeAssembler::AddOpcodeSizePrefix(size_t Pos) {
  size_t SizeInWords = (size() + 3) / 4;
  assert(SizeInWords <= 0x100u &&
         "Only 256 additional words are allowed for unwind opcodes");
  Ops[Pos] = static_cast<uint8_t>(SizeInWords - 1);
}

void UnwindOpcodeAssembler::AddPersonalityIndexPrefix(size_t Pos, unsigned PI) {
  assert(PI < NUM_PERSONALITY_INDEX && "Invalid personality prefix");
  Ops[Pos] = EHT_COMPACT | PI;
}

void UnwindOpcodeAssembler::EmitFinishOpcodes() {
  for (size_t i = (0x4u - (size() & 0x3u)) & 0x3u; i > 0; --i)
    Ops.push_back(UNWIND_OPCODE_FINISH);
}

void UnwindOpcodeAssembler::Finalize() {
  if (HasPersonality) {
    // Personality specified by .personality directive
    Offset = 1;
    AddOpcodeSizePrefix(1);
  } else {
    if (getOpcodeSize() <= 3) {
      // __aeabi_unwind_cpp_pr0: [ 0x80 , OP1 , OP2 , OP3 ]
      Offset = 1;
      PersonalityIndex = AEABI_UNWIND_CPP_PR0;
      AddPersonalityIndexPrefix(Offset, PersonalityIndex);
    } else {
      // __aeabi_unwind_cpp_pr1: [ 0x81 , SIZE , OP1 , OP2 , ... ]
      Offset = 0;
      PersonalityIndex = AEABI_UNWIND_CPP_PR1;
      AddPersonalityIndexPrefix(Offset, PersonalityIndex);
      AddOpcodeSizePrefix(1);
    }
  }

  // Emit the padding finish opcodes if the size() is not multiple of 4.
  EmitFinishOpcodes();

  // Swap the byte order
  uint8_t *Ptr = Ops.begin() + Offset;
  assert(size() % 4 == 0 && "Final unwind opcodes should align to 4");
  for (size_t i = 0, n = size(); i < n; i += 4) {
    std::swap(Ptr[i], Ptr[i + 3]);
    std::swap(Ptr[i + 1], Ptr[i + 2]);
  }
}
