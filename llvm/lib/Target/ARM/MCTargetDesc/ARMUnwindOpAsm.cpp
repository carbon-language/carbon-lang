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
#include "llvm/Support/ARMEHABI.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"

using namespace llvm;

namespace {
  /// UnwindOpcodeStreamer - The simple wrapper over SmallVector to emit bytes
  /// with MSB to LSB per uint32_t ordering.  For example, the first byte will
  /// be placed in Vec[3], and the following bytes will be placed in 2, 1, 0,
  /// 7, 6, 5, 4, 11, 10, 9, 8, and so on.
  class UnwindOpcodeStreamer {
  private:
    SmallVectorImpl<uint8_t> &Vec;
    size_t Pos;

  public:
    UnwindOpcodeStreamer(SmallVectorImpl<uint8_t> &V) : Vec(V), Pos(3) {
    }

    /// Emit the byte in MSB to LSB per uint32_t order.
    inline void EmitByte(uint8_t elem) {
      Vec[Pos] = elem;
      Pos = (((Pos ^ 0x3u) + 1) ^ 0x3u);
    }

    /// Emit the size prefix.
    inline void EmitSize(size_t Size) {
      size_t SizeInWords = (Size + 3) / 4;
      assert(SizeInWords <= 0x100u &&
             "Only 256 additional words are allowed for unwind opcodes");
      EmitByte(static_cast<uint8_t>(SizeInWords - 1));
    }

    /// Emit the personality index prefix.
    inline void EmitPersonalityIndex(unsigned PI) {
      assert(PI < ARM::EHABI::NUM_PERSONALITY_INDEX &&
             "Invalid personality prefix");
      EmitByte(ARM::EHABI::EHT_COMPACT | PI);
    }

    /// Fill the rest of bytes with FINISH opcode.
    inline void FillFinishOpcode() {
      while (Pos < Vec.size())
        EmitByte(ARM::EHABI::UNWIND_OPCODE_FINISH);
    }
  };
}

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
      EmitInt8(ARM::EHABI::UNWIND_OPCODE_POP_REG_RANGE_R4 | Range);
      RegSave &= 0x000fu;
    } else if (UnmaskedReg == (1u << 14)) {
      // Pop r[14] + r[4 : (4 + n)]
      EmitInt8(ARM::EHABI::UNWIND_OPCODE_POP_REG_RANGE_R4_R14 | Range);
      RegSave &= 0x000fu;
    }
  }

  // Two bytes opcode to save register r15-r4
  if ((RegSave & 0xfff0u) != 0)
    EmitInt16(ARM::EHABI::UNWIND_OPCODE_POP_REG_MASK_R4 | (RegSave >> 4));

  // Opcode to save register r3-r0
  if ((RegSave & 0x000fu) != 0)
    EmitInt16(ARM::EHABI::UNWIND_OPCODE_POP_REG_MASK | (RegSave & 0x000fu));
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

    EmitInt16(ARM::EHABI::UNWIND_OPCODE_POP_VFP_REG_RANGE_FSTMFDD_D16 |
              ((i - 16) << 4) | Range);
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

    EmitInt16(ARM::EHABI::UNWIND_OPCODE_POP_VFP_REG_RANGE_FSTMFDD | (i << 4) |
              Range);
  }
}

/// Emit unwind opcodes to copy address from source register to $sp.
void UnwindOpcodeAssembler::EmitSetSP(uint16_t Reg) {
  EmitInt8(ARM::EHABI::UNWIND_OPCODE_SET_VSP | Reg);
}

/// Emit unwind opcodes to add $sp with an offset.
void UnwindOpcodeAssembler::EmitSPOffset(int64_t Offset) {
  if (Offset > 0x200) {
    uint8_t Buff[16];
    Buff[0] = ARM::EHABI::UNWIND_OPCODE_INC_VSP_ULEB128;
    size_t ULEBSize = encodeULEB128((Offset - 0x204) >> 2, Buff + 1);
    EmitBytes(Buff, ULEBSize + 1);
  } else if (Offset > 0) {
    if (Offset > 0x100) {
      EmitInt8(ARM::EHABI::UNWIND_OPCODE_INC_VSP | 0x3fu);
      Offset -= 0x100;
    }
    EmitInt8(ARM::EHABI::UNWIND_OPCODE_INC_VSP |
             static_cast<uint8_t>((Offset - 4) >> 2));
  } else if (Offset < 0) {
    while (Offset < -0x100) {
      EmitInt8(ARM::EHABI::UNWIND_OPCODE_DEC_VSP | 0x3fu);
      Offset += 0x100;
    }
    EmitInt8(ARM::EHABI::UNWIND_OPCODE_DEC_VSP |
             static_cast<uint8_t>(((-Offset) - 4) >> 2));
  }
}

void UnwindOpcodeAssembler::Finalize(unsigned &PersonalityIndex,
                                     SmallVectorImpl<uint8_t> &Result) {

  UnwindOpcodeStreamer OpStreamer(Result);

  if (HasPersonality) {
    // User-specifed personality routine: [ SIZE , OP1 , OP2 , ... ]
    PersonalityIndex = ARM::EHABI::NUM_PERSONALITY_INDEX;
    size_t TotalSize = Ops.size() + 1;
    size_t RoundUpSize = (TotalSize + 3) / 4 * 4;
    Result.resize(RoundUpSize);
    OpStreamer.EmitSize(RoundUpSize);
  } else {
    if (Ops.size() <= 3) {
      // __aeabi_unwind_cpp_pr0: [ 0x80 , OP1 , OP2 , OP3 ]
      PersonalityIndex = ARM::EHABI::AEABI_UNWIND_CPP_PR0;
      Result.resize(4);
      OpStreamer.EmitPersonalityIndex(PersonalityIndex);
    } else {
      // __aeabi_unwind_cpp_pr1: [ 0x81 , SIZE , OP1 , OP2 , ... ]
      PersonalityIndex = ARM::EHABI::AEABI_UNWIND_CPP_PR1;
      size_t TotalSize = Ops.size() + 2;
      size_t RoundUpSize = (TotalSize + 3) / 4 * 4;
      Result.resize(RoundUpSize);
      OpStreamer.EmitPersonalityIndex(PersonalityIndex);
      OpStreamer.EmitSize(RoundUpSize);
    }
  }

  // Copy the unwind opcodes
  for (size_t i = OpBegins.size() - 1; i > 0; --i)
    for (size_t j = OpBegins[i - 1], end = OpBegins[i]; j < end; ++j)
      OpStreamer.EmitByte(Ops[j]);

  // Emit the padding finish opcodes if the size is not multiple of 4.
  OpStreamer.FillFinishOpcode();

  // Reset the assembler state
  Reset();
}
