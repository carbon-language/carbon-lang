//===-- ARM64ExpandPseudoInsts.cpp - Expand pseudo instructions ---*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that expands pseudo instructions into target
// instructions to allow proper scheduling and other late optimizations.  This
// pass should be run after register allocation but before the post-regalloc
// scheduling pass.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARM64AddressingModes.h"
#include "ARM64InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/MathExtras.h"
using namespace llvm;

namespace {
class ARM64ExpandPseudo : public MachineFunctionPass {
public:
  static char ID;
  ARM64ExpandPseudo() : MachineFunctionPass(ID) {}

  const ARM64InstrInfo *TII;

  virtual bool runOnMachineFunction(MachineFunction &Fn);

  virtual const char *getPassName() const {
    return "ARM64 pseudo instruction expansion pass";
  }

private:
  bool expandMBB(MachineBasicBlock &MBB);
  bool expandMI(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI);
  bool expandMOVImm(MachineBasicBlock &MBB, MachineBasicBlock::iterator MBBI,
                    unsigned BitSize);
};
char ARM64ExpandPseudo::ID = 0;
}

/// \brief Transfer implicit operands on the pseudo instruction to the
/// instructions created from the expansion.
static void transferImpOps(MachineInstr &OldMI, MachineInstrBuilder &UseMI,
                           MachineInstrBuilder &DefMI) {
  const MCInstrDesc &Desc = OldMI.getDesc();
  for (unsigned i = Desc.getNumOperands(), e = OldMI.getNumOperands(); i != e;
       ++i) {
    const MachineOperand &MO = OldMI.getOperand(i);
    assert(MO.isReg() && MO.getReg());
    if (MO.isUse())
      UseMI.addOperand(MO);
    else
      DefMI.addOperand(MO);
  }
}

/// \brief Helper function which extracts the specified 16-bit chunk from a
/// 64-bit value.
static uint64_t getChunk(uint64_t Imm, unsigned ChunkIdx) {
  assert(ChunkIdx < 4 && "Out of range chunk index specified!");

  return (Imm >> (ChunkIdx * 16)) & 0xFFFF;
}

/// \brief Helper function which replicates a 16-bit chunk within a 64-bit
/// value. Indices correspond to element numbers in a v4i16.
static uint64_t replicateChunk(uint64_t Imm, unsigned FromIdx, unsigned ToIdx) {
  assert((FromIdx < 4) && (ToIdx < 4) && "Out of range chunk index specified!");
  const unsigned ShiftAmt = ToIdx * 16;

  // Replicate the source chunk to the destination position.
  const uint64_t Chunk = getChunk(Imm, FromIdx) << ShiftAmt;
  // Clear the destination chunk.
  Imm &= ~(0xFFFFLL << ShiftAmt);
  // Insert the replicated chunk.
  return Imm | Chunk;
}

/// \brief Helper function which tries to materialize a 64-bit value with an
/// ORR + MOVK instruction sequence.
static bool tryOrrMovk(uint64_t UImm, uint64_t OrrImm, MachineInstr &MI,
                       MachineBasicBlock &MBB,
                       MachineBasicBlock::iterator &MBBI,
                       const ARM64InstrInfo *TII, unsigned ChunkIdx) {
  assert(ChunkIdx < 4 && "Out of range chunk index specified!");
  const unsigned ShiftAmt = ChunkIdx * 16;

  uint64_t Encoding;
  if (ARM64_AM::processLogicalImmediate(OrrImm, 64, Encoding)) {
    // Create the ORR-immediate instruction.
    MachineInstrBuilder MIB =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::ORRXri))
            .addOperand(MI.getOperand(0))
            .addReg(ARM64::XZR)
            .addImm(Encoding);

    // Create the MOVK instruction.
    const unsigned Imm16 = getChunk(UImm, ChunkIdx);
    const unsigned DstReg = MI.getOperand(0).getReg();
    const bool DstIsDead = MI.getOperand(0).isDead();
    MachineInstrBuilder MIB1 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::MOVKXi))
            .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead))
            .addReg(DstReg)
            .addImm(Imm16)
            .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, ShiftAmt));

    transferImpOps(MI, MIB, MIB1);
    MI.eraseFromParent();
    return true;
  }

  return false;
}

/// \brief Check whether the given 16-bit chunk replicated to full 64-bit width
/// can be materialized with an ORR instruction.
static bool canUseOrr(uint64_t Chunk, uint64_t &Encoding) {
  Chunk = (Chunk << 48) | (Chunk << 32) | (Chunk << 16) | Chunk;

  return ARM64_AM::processLogicalImmediate(Chunk, 64, Encoding);
}

/// \brief Check for identical 16-bit chunks within the constant and if so
/// materialize them with a single ORR instruction. The remaining one or two
/// 16-bit chunks will be materialized with MOVK instructions.
///
/// This allows us to materialize constants like |A|B|A|A| or |A|B|C|A| (order
/// of the chunks doesn't matter), assuming |A|A|A|A| can be materialized with
/// an ORR instruction.
///
static bool tryToreplicateChunks(uint64_t UImm, MachineInstr &MI,
                                 MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator &MBBI,
                                 const ARM64InstrInfo *TII) {
  typedef DenseMap<uint64_t, unsigned> CountMap;
  CountMap Counts;

  // Scan the constant and count how often every chunk occurs.
  for (unsigned Idx = 0; Idx < 4; ++Idx)
    ++Counts[getChunk(UImm, Idx)];

  // Traverse the chunks to find one which occurs more than once.
  for (CountMap::const_iterator Chunk = Counts.begin(), End = Counts.end();
       Chunk != End; ++Chunk) {
    const uint64_t ChunkVal = Chunk->first;
    const unsigned Count = Chunk->second;

    uint64_t Encoding = 0;

    // We are looking for chunks which have two or three instances and can be
    // materialized with an ORR instruction.
    if ((Count != 2 && Count != 3) || !canUseOrr(ChunkVal, Encoding))
      continue;

    const bool CountThree = Count == 3;
    // Create the ORR-immediate instruction.
    MachineInstrBuilder MIB =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::ORRXri))
            .addOperand(MI.getOperand(0))
            .addReg(ARM64::XZR)
            .addImm(Encoding);

    const unsigned DstReg = MI.getOperand(0).getReg();
    const bool DstIsDead = MI.getOperand(0).isDead();

    unsigned ShiftAmt = 0;
    uint64_t Imm16 = 0;
    // Find the first chunk not materialized with the ORR instruction.
    for (; ShiftAmt < 64; ShiftAmt += 16) {
      Imm16 = (UImm >> ShiftAmt) & 0xFFFF;

      if (Imm16 != ChunkVal)
        break;
    }

    // Create the first MOVK instruction.
    MachineInstrBuilder MIB1 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::MOVKXi))
            .addReg(DstReg,
                    RegState::Define | getDeadRegState(DstIsDead && CountThree))
            .addReg(DstReg)
            .addImm(Imm16)
            .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, ShiftAmt));

    // In case we have three instances the whole constant is now materialized
    // and we can exit.
    if (CountThree) {
      transferImpOps(MI, MIB, MIB1);
      MI.eraseFromParent();
      return true;
    }

    // Find the remaining chunk which needs to be materialized.
    for (ShiftAmt += 16; ShiftAmt < 64; ShiftAmt += 16) {
      Imm16 = (UImm >> ShiftAmt) & 0xFFFF;

      if (Imm16 != ChunkVal)
        break;
    }

    // Create the second MOVK instruction.
    MachineInstrBuilder MIB2 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::MOVKXi))
            .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead))
            .addReg(DstReg)
            .addImm(Imm16)
            .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, ShiftAmt));

    transferImpOps(MI, MIB, MIB2);
    MI.eraseFromParent();
    return true;
  }

  return false;
}

/// \brief Check whether this chunk matches the pattern '1...0...'. This pattern
/// starts a contiguous sequence of ones if we look at the bits from the LSB
/// towards the MSB.
static bool isStartChunk(uint64_t Chunk) {
  if (Chunk == 0 || Chunk == UINT64_MAX)
    return false;

  return (CountLeadingOnes_64(Chunk) + countTrailingZeros(Chunk)) == 64;
}

/// \brief Check whether this chunk matches the pattern '0...1...' This pattern
/// ends a contiguous sequence of ones if we look at the bits from the LSB
/// towards the MSB.
static bool isEndChunk(uint64_t Chunk) {
  if (Chunk == 0 || Chunk == UINT64_MAX)
    return false;

  return (countLeadingZeros(Chunk) + CountTrailingOnes_64(Chunk)) == 64;
}

/// \brief Clear or set all bits in the chunk at the given index.
static uint64_t updateImm(uint64_t Imm, unsigned Idx, bool Clear) {
  const uint64_t Mask = 0xFFFF;

  if (Clear)
    // Clear chunk in the immediate.
    Imm &= ~(Mask << (Idx * 16));
  else
    // Set all bits in the immediate for the particular chunk.
    Imm |= Mask << (Idx * 16);

  return Imm;
}

/// \brief Check whether the constant contains a sequence of contiguous ones,
/// which might be interrupted by one or two chunks. If so, materialize the
/// sequence of contiguous ones with an ORR instruction.
/// Materialize the chunks which are either interrupting the sequence or outside
/// of the sequence with a MOVK instruction.
///
/// Assuming S is a chunk which starts the sequence (1...0...), E is a chunk
/// which ends the sequence (0...1...). Then we are looking for constants which
/// contain at least one S and E chunk.
/// E.g. |E|A|B|S|, |A|E|B|S| or |A|B|E|S|.
///
/// We are also looking for constants like |S|A|B|E| where the contiguous
/// sequence of ones wraps around the MSB into the LSB.
///
static bool trySequenceOfOnes(uint64_t UImm, MachineInstr &MI,
                              MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator &MBBI,
                              const ARM64InstrInfo *TII) {
  const int NotSet = -1;
  const uint64_t Mask = 0xFFFF;

  int StartIdx = NotSet;
  int EndIdx = NotSet;
  // Try to find the chunks which start/end a contiguous sequence of ones.
  for (int Idx = 0; Idx < 4; ++Idx) {
    int64_t Chunk = getChunk(UImm, Idx);
    // Sign extend the 16-bit chunk to 64-bit.
    Chunk = (Chunk << 48) >> 48;

    if (isStartChunk(Chunk))
      StartIdx = Idx;
    else if (isEndChunk(Chunk))
      EndIdx = Idx;
  }

  // Early exit in case we can't find a start/end chunk.
  if (StartIdx == NotSet || EndIdx == NotSet)
    return false;

  // Outside of the contiguous sequence of ones everything needs to be zero.
  uint64_t Outside = 0;
  // Chunks between the start and end chunk need to have all their bits set.
  uint64_t Inside = Mask;

  // If our contiguous sequence of ones wraps around from the MSB into the LSB,
  // just swap indices and pretend we are materializing a contiguous sequence
  // of zeros surrounded by a contiguous sequence of ones.
  if (StartIdx > EndIdx) {
    std::swap(StartIdx, EndIdx);
    std::swap(Outside, Inside);
  }

  uint64_t OrrImm = UImm;
  int FirstMovkIdx = NotSet;
  int SecondMovkIdx = NotSet;

  // Find out which chunks we need to patch up to obtain a contiguous sequence
  // of ones.
  for (int Idx = 0; Idx < 4; ++Idx) {
    const uint64_t Chunk = getChunk(UImm, Idx);

    // Check whether we are looking at a chunk which is not part of the
    // contiguous sequence of ones.
    if ((Idx < StartIdx || EndIdx < Idx) && Chunk != Outside) {
      OrrImm = updateImm(OrrImm, Idx, Outside == 0);

      // Remember the index we need to patch.
      if (FirstMovkIdx == NotSet)
        FirstMovkIdx = Idx;
      else
        SecondMovkIdx = Idx;

      // Check whether we are looking a chunk which is part of the contiguous
      // sequence of ones.
    } else if (Idx > StartIdx && Idx < EndIdx && Chunk != Inside) {
      OrrImm = updateImm(OrrImm, Idx, Inside != Mask);

      // Remember the index we need to patch.
      if (FirstMovkIdx == NotSet)
        FirstMovkIdx = Idx;
      else
        SecondMovkIdx = Idx;
    }
  }
  assert(FirstMovkIdx != NotSet && "Constant materializable with single ORR!");

  // Create the ORR-immediate instruction.
  uint64_t Encoding = 0;
  ARM64_AM::processLogicalImmediate(OrrImm, 64, Encoding);
  MachineInstrBuilder MIB =
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::ORRXri))
          .addOperand(MI.getOperand(0))
          .addReg(ARM64::XZR)
          .addImm(Encoding);

  const unsigned DstReg = MI.getOperand(0).getReg();
  const bool DstIsDead = MI.getOperand(0).isDead();

  const bool SingleMovk = SecondMovkIdx == NotSet;
  // Create the first MOVK instruction.
  MachineInstrBuilder MIB1 =
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::MOVKXi))
          .addReg(DstReg,
                  RegState::Define | getDeadRegState(DstIsDead && SingleMovk))
          .addReg(DstReg)
          .addImm(getChunk(UImm, FirstMovkIdx))
          .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, FirstMovkIdx * 16));

  // Early exit in case we only need to emit a single MOVK instruction.
  if (SingleMovk) {
    transferImpOps(MI, MIB, MIB1);
    MI.eraseFromParent();
    return true;
  }

  // Create the second MOVK instruction.
  MachineInstrBuilder MIB2 =
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::MOVKXi))
          .addReg(DstReg, RegState::Define | getDeadRegState(DstIsDead))
          .addReg(DstReg)
          .addImm(getChunk(UImm, SecondMovkIdx))
          .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, SecondMovkIdx * 16));

  transferImpOps(MI, MIB, MIB2);
  MI.eraseFromParent();
  return true;
}

/// \brief Expand a MOVi32imm or MOVi64imm pseudo instruction to one or more
/// real move-immediate instructions to synthesize the immediate.
bool ARM64ExpandPseudo::expandMOVImm(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     unsigned BitSize) {
  MachineInstr &MI = *MBBI;
  uint64_t Imm = MI.getOperand(1).getImm();
  const unsigned Mask = 0xFFFF;

  // Try a MOVI instruction (aka ORR-immediate with the zero register).
  uint64_t UImm = Imm << (64 - BitSize) >> (64 - BitSize);
  uint64_t Encoding;
  if (ARM64_AM::processLogicalImmediate(UImm, BitSize, Encoding)) {
    unsigned Opc = (BitSize == 32 ? ARM64::ORRWri : ARM64::ORRXri);
    MachineInstrBuilder MIB =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(Opc))
            .addOperand(MI.getOperand(0))
            .addReg(BitSize == 32 ? ARM64::WZR : ARM64::XZR)
            .addImm(Encoding);
    transferImpOps(MI, MIB, MIB);
    MI.eraseFromParent();
    return true;
  }

  // Scan the immediate and count the number of 16-bit chunks which are either
  // all ones or all zeros.
  unsigned OneChunks = 0;
  unsigned ZeroChunks = 0;
  for (unsigned Shift = 0; Shift < BitSize; Shift += 16) {
    const unsigned Chunk = (Imm >> Shift) & Mask;
    if (Chunk == Mask)
      OneChunks++;
    else if (Chunk == 0)
      ZeroChunks++;
  }

  // Since we can't materialize the constant with a single ORR instruction,
  // let's see whether we can materialize 3/4 of the constant with an ORR
  // instruction and use an additional MOVK instruction to materialize the
  // remaining 1/4.
  //
  // We are looking for constants with a pattern like: |A|X|B|X| or |X|A|X|B|.
  //
  // E.g. assuming |A|X|A|X| is a pattern which can be materialized with ORR,
  // we would create the following instruction sequence:
  //
  // ORR x0, xzr, |A|X|A|X|
  // MOVK x0, |B|, LSL #16
  //
  // Only look at 64-bit constants which can't be materialized with a single
  // instruction e.g. which have less than either three all zero or all one
  // chunks.
  //
  // Ignore 32-bit constants here, they always can be materialized with a
  // MOVZ/MOVN + MOVK pair. Since the 32-bit constant can't be materialized
  // with a single ORR, the best sequence we can achieve is a ORR + MOVK pair.
  // Thus we fall back to the default code below which in the best case creates
  // a single MOVZ/MOVN instruction (in case one chunk is all zero or all one).
  //
  if (BitSize == 64 && OneChunks < 3 && ZeroChunks < 3) {
    // If we interpret the 64-bit constant as a v4i16, are elements 0 and 2
    // identical?
    if (getChunk(UImm, 0) == getChunk(UImm, 2)) {
      // See if we can come up with a constant which can be materialized with
      // ORR-immediate by replicating element 3 into element 1.
      uint64_t OrrImm = replicateChunk(UImm, 3, 1);
      if (tryOrrMovk(UImm, OrrImm, MI, MBB, MBBI, TII, 1))
        return true;

      // See if we can come up with a constant which can be materialized with
      // ORR-immediate by replicating element 1 into element 3.
      OrrImm = replicateChunk(UImm, 1, 3);
      if (tryOrrMovk(UImm, OrrImm, MI, MBB, MBBI, TII, 3))
        return true;

      // If we interpret the 64-bit constant as a v4i16, are elements 1 and 3
      // identical?
    } else if (getChunk(UImm, 1) == getChunk(UImm, 3)) {
      // See if we can come up with a constant which can be materialized with
      // ORR-immediate by replicating element 2 into element 0.
      uint64_t OrrImm = replicateChunk(UImm, 2, 0);
      if (tryOrrMovk(UImm, OrrImm, MI, MBB, MBBI, TII, 0))
        return true;

      // See if we can come up with a constant which can be materialized with
      // ORR-immediate by replicating element 1 into element 3.
      OrrImm = replicateChunk(UImm, 0, 2);
      if (tryOrrMovk(UImm, OrrImm, MI, MBB, MBBI, TII, 2))
        return true;
    }
  }

  // Check for identical 16-bit chunks within the constant and if so materialize
  // them with a single ORR instruction. The remaining one or two 16-bit chunks
  // will be materialized with MOVK instructions.
  if (BitSize == 64 && tryToreplicateChunks(UImm, MI, MBB, MBBI, TII))
    return true;

  // Check whether the constant contains a sequence of contiguous ones, which
  // might be interrupted by one or two chunks. If so, materialize the sequence
  // of contiguous ones with an ORR instruction. Materialize the chunks which
  // are either interrupting the sequence or outside of the sequence with a
  // MOVK instruction.
  if (BitSize == 64 && trySequenceOfOnes(UImm, MI, MBB, MBBI, TII))
    return true;

  // Use a MOVZ or MOVN instruction to set the high bits, followed by one or
  // more MOVK instructions to insert additional 16-bit portions into the
  // lower bits.
  bool isNeg = false;

  // Use MOVN to materialize the high bits if we have more all one chunks
  // than all zero chunks.
  if (OneChunks > ZeroChunks) {
    isNeg = true;
    Imm = ~Imm;
  }

  unsigned FirstOpc;
  if (BitSize == 32) {
    Imm &= (1LL << 32) - 1;
    FirstOpc = (isNeg ? ARM64::MOVNWi : ARM64::MOVZWi);
  } else {
    FirstOpc = (isNeg ? ARM64::MOVNXi : ARM64::MOVZXi);
  }
  unsigned Shift = 0;     // LSL amount for high bits with MOVZ/MOVN
  unsigned LastShift = 0; // LSL amount for last MOVK
  if (Imm != 0) {
    unsigned LZ = countLeadingZeros(Imm);
    unsigned TZ = countTrailingZeros(Imm);
    Shift = ((63 - LZ) / 16) * 16;
    LastShift = (TZ / 16) * 16;
  }
  unsigned Imm16 = (Imm >> Shift) & Mask;
  unsigned DstReg = MI.getOperand(0).getReg();
  bool DstIsDead = MI.getOperand(0).isDead();
  MachineInstrBuilder MIB1 =
      BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(FirstOpc))
          .addReg(DstReg, RegState::Define |
                              getDeadRegState(DstIsDead && Shift == LastShift))
          .addImm(Imm16)
          .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, Shift));

  // If a MOVN was used for the high bits of a negative value, flip the rest
  // of the bits back for use with MOVK.
  if (isNeg)
    Imm = ~Imm;

  if (Shift == LastShift) {
    transferImpOps(MI, MIB1, MIB1);
    MI.eraseFromParent();
    return true;
  }

  MachineInstrBuilder MIB2;
  unsigned Opc = (BitSize == 32 ? ARM64::MOVKWi : ARM64::MOVKXi);
  while (Shift != LastShift) {
    Shift -= 16;
    Imm16 = (Imm >> Shift) & Mask;
    if (Imm16 == (isNeg ? Mask : 0))
      continue; // This 16-bit portion is already set correctly.
    MIB2 = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(Opc))
               .addReg(DstReg,
                       RegState::Define |
                           getDeadRegState(DstIsDead && Shift == LastShift))
               .addReg(DstReg)
               .addImm(Imm16)
               .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, Shift));
  }

  transferImpOps(MI, MIB1, MIB2);
  MI.eraseFromParent();
  return true;
}

/// \brief If MBBI references a pseudo instruction that should be expanded here,
/// do the expansion and return true.  Otherwise return false.
bool ARM64ExpandPseudo::expandMI(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MBBI) {
  MachineInstr &MI = *MBBI;
  unsigned Opcode = MI.getOpcode();
  switch (Opcode) {
  default:
    break;

  case ARM64::ADDWrr:
  case ARM64::SUBWrr:
  case ARM64::ADDXrr:
  case ARM64::SUBXrr:
  case ARM64::ADDSWrr:
  case ARM64::SUBSWrr:
  case ARM64::ADDSXrr:
  case ARM64::SUBSXrr:
  case ARM64::ANDWrr:
  case ARM64::ANDXrr:
  case ARM64::BICWrr:
  case ARM64::BICXrr:
  case ARM64::EONWrr:
  case ARM64::EONXrr:
  case ARM64::EORWrr:
  case ARM64::EORXrr:
  case ARM64::ORNWrr:
  case ARM64::ORNXrr:
  case ARM64::ORRWrr:
  case ARM64::ORRXrr: {
    unsigned Opcode;
    switch (MI.getOpcode()) {
    default:
      return false;
    case ARM64::ADDWrr:      Opcode = ARM64::ADDWrs; break;
    case ARM64::SUBWrr:      Opcode = ARM64::SUBWrs; break;
    case ARM64::ADDXrr:      Opcode = ARM64::ADDXrs; break;
    case ARM64::SUBXrr:      Opcode = ARM64::SUBXrs; break;
    case ARM64::ADDSWrr:     Opcode = ARM64::ADDSWrs; break;
    case ARM64::SUBSWrr:     Opcode = ARM64::SUBSWrs; break;
    case ARM64::ADDSXrr:     Opcode = ARM64::ADDSXrs; break;
    case ARM64::SUBSXrr:     Opcode = ARM64::SUBSXrs; break;
    case ARM64::ANDWrr:      Opcode = ARM64::ANDWrs; break;
    case ARM64::ANDXrr:      Opcode = ARM64::ANDXrs; break;
    case ARM64::BICWrr:      Opcode = ARM64::BICWrs; break;
    case ARM64::BICXrr:      Opcode = ARM64::BICXrs; break;
    case ARM64::EONWrr:      Opcode = ARM64::EONWrs; break;
    case ARM64::EONXrr:      Opcode = ARM64::EONXrs; break;
    case ARM64::EORWrr:      Opcode = ARM64::EORWrs; break;
    case ARM64::EORXrr:      Opcode = ARM64::EORXrs; break;
    case ARM64::ORNWrr:      Opcode = ARM64::ORNWrs; break;
    case ARM64::ORNXrr:      Opcode = ARM64::ORNXrs; break;
    case ARM64::ORRWrr:      Opcode = ARM64::ORRWrs; break;
    case ARM64::ORRXrr:      Opcode = ARM64::ORRXrs; break;
    }
    MachineInstrBuilder MIB1 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(Opcode),
                MI.getOperand(0).getReg())
            .addOperand(MI.getOperand(1))
            .addOperand(MI.getOperand(2))
            .addImm(ARM64_AM::getShifterImm(ARM64_AM::LSL, 0));
    transferImpOps(MI, MIB1, MIB1);
    MI.eraseFromParent();
    return true;
  }

  case ARM64::FCVTSHpseudo: {
    MachineOperand Src = MI.getOperand(1);
    Src.setImplicit();
    unsigned SrcH = TII->getRegisterInfo().getSubReg(Src.getReg(), ARM64::hsub);
    auto MIB = BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::FCVTSHr))
                   .addOperand(MI.getOperand(0))
                   .addReg(SrcH, RegState::Undef)
                   .addOperand(Src);
    transferImpOps(MI, MIB, MIB);
    MI.eraseFromParent();
    return true;
  }
  case ARM64::LOADgot: {
    // Expand into ADRP + LDR.
    unsigned DstReg = MI.getOperand(0).getReg();
    const MachineOperand &MO1 = MI.getOperand(1);
    unsigned Flags = MO1.getTargetFlags();
    MachineInstrBuilder MIB1 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::ADRP), DstReg);
    MachineInstrBuilder MIB2 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::LDRXui))
            .addOperand(MI.getOperand(0))
            .addReg(DstReg);

    if (MO1.isGlobal()) {
      MIB1.addGlobalAddress(MO1.getGlobal(), 0, Flags | ARM64II::MO_PAGE);
      MIB2.addGlobalAddress(MO1.getGlobal(), 0,
                            Flags | ARM64II::MO_PAGEOFF | ARM64II::MO_NC);
    } else if (MO1.isSymbol()) {
      MIB1.addExternalSymbol(MO1.getSymbolName(), Flags | ARM64II::MO_PAGE);
      MIB2.addExternalSymbol(MO1.getSymbolName(),
                             Flags | ARM64II::MO_PAGEOFF | ARM64II::MO_NC);
    } else {
      assert(MO1.isCPI() &&
             "Only expect globals, externalsymbols, or constant pools");
      MIB1.addConstantPoolIndex(MO1.getIndex(), MO1.getOffset(),
                                Flags | ARM64II::MO_PAGE);
      MIB2.addConstantPoolIndex(MO1.getIndex(), MO1.getOffset(),
                                Flags | ARM64II::MO_PAGEOFF | ARM64II::MO_NC);
    }

    transferImpOps(MI, MIB1, MIB2);
    MI.eraseFromParent();
    return true;
  }

  case ARM64::MOVaddr:
  case ARM64::MOVaddrJT:
  case ARM64::MOVaddrCP:
  case ARM64::MOVaddrBA:
  case ARM64::MOVaddrTLS:
  case ARM64::MOVaddrEXT: {
    // Expand into ADRP + ADD.
    unsigned DstReg = MI.getOperand(0).getReg();
    MachineInstrBuilder MIB1 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::ADRP), DstReg)
            .addOperand(MI.getOperand(1));

    MachineInstrBuilder MIB2 =
        BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::ADDXri))
            .addOperand(MI.getOperand(0))
            .addReg(DstReg)
            .addOperand(MI.getOperand(2))
            .addImm(0);

    transferImpOps(MI, MIB1, MIB2);
    MI.eraseFromParent();
    return true;
  }

  case ARM64::MOVi32imm:
    return expandMOVImm(MBB, MBBI, 32);
  case ARM64::MOVi64imm:
    return expandMOVImm(MBB, MBBI, 64);
  case ARM64::RET_ReallyLR:
    BuildMI(MBB, MBBI, MI.getDebugLoc(), TII->get(ARM64::RET))
        .addReg(ARM64::LR);
    MI.eraseFromParent();
    return true;
  }
  return false;
}

/// \brief Iterate over the instructions in basic block MBB and expand any
/// pseudo instructions.  Return true if anything was modified.
bool ARM64ExpandPseudo::expandMBB(MachineBasicBlock &MBB) {
  bool Modified = false;

  MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
  while (MBBI != E) {
    MachineBasicBlock::iterator NMBBI = std::next(MBBI);
    Modified |= expandMI(MBB, MBBI);
    MBBI = NMBBI;
  }

  return Modified;
}

bool ARM64ExpandPseudo::runOnMachineFunction(MachineFunction &MF) {
  TII = static_cast<const ARM64InstrInfo *>(MF.getTarget().getInstrInfo());

  bool Modified = false;
  for (MachineFunction::iterator MFI = MF.begin(), E = MF.end(); MFI != E;
       ++MFI)
    Modified |= expandMBB(*MFI);
  return Modified;
}

/// \brief Returns an instance of the pseudo instruction expansion pass.
FunctionPass *llvm::createARM64ExpandPseudoPass() {
  return new ARM64ExpandPseudo();
}
