//===- RISCVMatInt.cpp - Immediate materialisation -------------*- C++ -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "RISCVMatInt.h"
#include "MCTargetDesc/RISCVMCTargetDesc.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

static int getInstSeqCost(RISCVMatInt::InstSeq &Res, bool HasRVC) {
  if (!HasRVC)
    return Res.size();

  int Cost = 0;
  for (auto Instr : Res) {
    bool Compressed;
    switch (Instr.Opc) {
    default: llvm_unreachable("Unexpected opcode");
    case RISCV::SLLI:
    case RISCV::SRLI:
      Compressed = true;
      break;
    case RISCV::ADDI:
    case RISCV::ADDIW:
    case RISCV::LUI:
      Compressed = isInt<6>(Instr.Imm);
      break;
    case RISCV::ADDUW:
      Compressed = false;
      break;
    }
    // Two RVC instructions take the same space as one RVI instruction, but
    // can take longer to execute than the single RVI instruction. Thus, we
    // consider that two RVC instruction are slightly more costly than one
    // RVI instruction. For longer sequences of RVC instructions the space
    // savings can be worth it, though. The costs below try to model that.
    if (!Compressed)
      Cost += 100; // Baseline cost of one RVI instruction: 100%.
    else
      Cost += 70; // 70% cost of baseline.
  }
  return Cost;
}

// Recursively generate a sequence for materializing an integer.
static void generateInstSeqImpl(int64_t Val,
                                const FeatureBitset &ActiveFeatures,
                                RISCVMatInt::InstSeq &Res) {
  bool IsRV64 = ActiveFeatures[RISCV::Feature64Bit];

  if (isInt<32>(Val)) {
    // Depending on the active bits in the immediate Value v, the following
    // instruction sequences are emitted:
    //
    // v == 0                        : ADDI
    // v[0,12) != 0 && v[12,32) == 0 : ADDI
    // v[0,12) == 0 && v[12,32) != 0 : LUI
    // v[0,32) != 0                  : LUI+ADDI(W)
    int64_t Hi20 = ((Val + 0x800) >> 12) & 0xFFFFF;
    int64_t Lo12 = SignExtend64<12>(Val);

    if (Hi20)
      Res.push_back(RISCVMatInt::Inst(RISCV::LUI, Hi20));

    if (Lo12 || Hi20 == 0) {
      unsigned AddiOpc = (IsRV64 && Hi20) ? RISCV::ADDIW : RISCV::ADDI;
      Res.push_back(RISCVMatInt::Inst(AddiOpc, Lo12));
    }
    return;
  }

  assert(IsRV64 && "Can't emit >32-bit imm for non-RV64 target");

  // In the worst case, for a full 64-bit constant, a sequence of 8 instructions
  // (i.e., LUI+ADDIW+SLLI+ADDI+SLLI+ADDI+SLLI+ADDI) has to be emmitted. Note
  // that the first two instructions (LUI+ADDIW) can contribute up to 32 bits
  // while the following ADDI instructions contribute up to 12 bits each.
  //
  // On the first glance, implementing this seems to be possible by simply
  // emitting the most significant 32 bits (LUI+ADDIW) followed by as many left
  // shift (SLLI) and immediate additions (ADDI) as needed. However, due to the
  // fact that ADDI performs a sign extended addition, doing it like that would
  // only be possible when at most 11 bits of the ADDI instructions are used.
  // Using all 12 bits of the ADDI instructions, like done by GAS, actually
  // requires that the constant is processed starting with the least significant
  // bit.
  //
  // In the following, constants are processed from LSB to MSB but instruction
  // emission is performed from MSB to LSB by recursively calling
  // generateInstSeq. In each recursion, first the lowest 12 bits are removed
  // from the constant and the optimal shift amount, which can be greater than
  // 12 bits if the constant is sparse, is determined. Then, the shifted
  // remaining constant is processed recursively and gets emitted as soon as it
  // fits into 32 bits. The emission of the shifts and additions is subsequently
  // performed when the recursion returns.

  int64_t Lo12 = SignExtend64<12>(Val);
  int64_t Hi52 = ((uint64_t)Val + 0x800ull) >> 12;
  int ShiftAmount = 12 + findFirstSet((uint64_t)Hi52);
  Hi52 = SignExtend64(Hi52 >> (ShiftAmount - 12), 64 - ShiftAmount);

  // If the remaining bits don't fit in 12 bits, we might be able to reduce the
  // shift amount in order to use LUI which will zero the lower 12 bits.
  if (ShiftAmount > 12 && !isInt<12>(Hi52) && isInt<32>((uint64_t)Hi52 << 12)) {
    // Reduce the shift amount and add zeros to the LSBs so it will match LUI.
    ShiftAmount -= 12;
    Hi52 = (uint64_t)Hi52 << 12;
  }

  generateInstSeqImpl(Hi52, ActiveFeatures, Res);

  Res.push_back(RISCVMatInt::Inst(RISCV::SLLI, ShiftAmount));
  if (Lo12)
    Res.push_back(RISCVMatInt::Inst(RISCV::ADDI, Lo12));
}

namespace llvm {
namespace RISCVMatInt {
InstSeq generateInstSeq(int64_t Val, const FeatureBitset &ActiveFeatures) {
  RISCVMatInt::InstSeq Res;
  generateInstSeqImpl(Val, ActiveFeatures, Res);

  // If the constant is positive we might be able to generate a shifted constant
  // with no leading zeros and use a final SRLI to restore them.
  if (Val > 0 && Res.size() > 2) {
    assert(ActiveFeatures[RISCV::Feature64Bit] &&
           "Expected RV32 to only need 2 instructions");
    unsigned LeadingZeros = countLeadingZeros((uint64_t)Val);
    uint64_t ShiftedVal = (uint64_t)Val << LeadingZeros;
    // Fill in the bits that will be shifted out with 1s. An example where this
    // helps is trailing one masks with 32 or more ones. This will generate
    // ADDI -1 and an SRLI.
    ShiftedVal |= maskTrailingOnes<uint64_t>(LeadingZeros);

    RISCVMatInt::InstSeq TmpSeq;
    generateInstSeqImpl(ShiftedVal, ActiveFeatures, TmpSeq);
    TmpSeq.push_back(RISCVMatInt::Inst(RISCV::SRLI, LeadingZeros));

    // Keep the new sequence if it is an improvement.
    if (TmpSeq.size() < Res.size()) {
      Res = TmpSeq;
      // A 2 instruction sequence is the best we can do.
      if (Res.size() <= 2)
        return Res;
    }

    // Some cases can benefit from filling the lower bits with zeros instead.
    ShiftedVal &= maskTrailingZeros<uint64_t>(LeadingZeros);
    TmpSeq.clear();
    generateInstSeqImpl(ShiftedVal, ActiveFeatures, TmpSeq);
    TmpSeq.push_back(RISCVMatInt::Inst(RISCV::SRLI, LeadingZeros));

    // Keep the new sequence if it is an improvement.
    if (TmpSeq.size() < Res.size()) {
      Res = TmpSeq;
      // A 2 instruction sequence is the best we can do.
      if (Res.size() <= 2)
        return Res;
    }

    // If we have exactly 32 leading zeros and Zba, we can try using zext.w at
    // the end of the sequence.
    if (LeadingZeros == 32 && ActiveFeatures[RISCV::FeatureExtZba]) {
      // Try replacing upper bits with 1.
      uint64_t LeadingOnesVal = Val | maskLeadingOnes<uint64_t>(LeadingZeros);
      TmpSeq.clear();
      generateInstSeqImpl(LeadingOnesVal, ActiveFeatures, TmpSeq);
      TmpSeq.push_back(RISCVMatInt::Inst(RISCV::ADDUW, 0));

      // Keep the new sequence if it is an improvement.
      if (TmpSeq.size() < Res.size()) {
        Res = TmpSeq;
        // A 2 instruction sequence is the best we can do.
        if (Res.size() <= 2)
          return Res;
      }
    }
  }

  return Res;
}

int getIntMatCost(const APInt &Val, unsigned Size,
                  const FeatureBitset &ActiveFeatures,
                  bool CompressionCost) {
  bool IsRV64 = ActiveFeatures[RISCV::Feature64Bit];
  bool HasRVC = CompressionCost && ActiveFeatures[RISCV::FeatureStdExtC];
  int PlatRegSize = IsRV64 ? 64 : 32;

  // Split the constant into platform register sized chunks, and calculate cost
  // of each chunk.
  int Cost = 0;
  for (unsigned ShiftVal = 0; ShiftVal < Size; ShiftVal += PlatRegSize) {
    APInt Chunk = Val.ashr(ShiftVal).sextOrTrunc(PlatRegSize);
    InstSeq MatSeq = generateInstSeq(Chunk.getSExtValue(), ActiveFeatures);
    Cost += getInstSeqCost(MatSeq, HasRVC);
  }
  return std::max(1, Cost);
}
} // namespace RISCVMatInt
} // namespace llvm
