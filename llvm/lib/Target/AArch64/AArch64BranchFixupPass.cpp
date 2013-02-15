//===-- AArch64BranchFixupPass.cpp - AArch64 branch fixup -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that fixes AArch64 branches which have ended up out
// of range for their immediate operands.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "aarch64-branch-fixup"
#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

STATISTIC(NumSplit,      "Number of uncond branches inserted");
STATISTIC(NumCBrFixed,   "Number of cond branches fixed");

/// Return the worst case padding that could result from unknown offset bits.
/// This does not include alignment padding caused by known offset bits.
///
/// @param LogAlign log2(alignment)
/// @param KnownBits Number of known low offset bits.
static inline unsigned UnknownPadding(unsigned LogAlign, unsigned KnownBits) {
  if (KnownBits < LogAlign)
    return (1u << LogAlign) - (1u << KnownBits);
  return 0;
}

namespace {
  /// Due to limited PC-relative displacements, conditional branches to distant
  /// blocks may need converting into an unconditional equivalent. For example:
  ///     tbz w1, #0, far_away
  /// becomes
  ///     tbnz w1, #0, skip
  ///     b far_away
  ///   skip:
  class AArch64BranchFixup : public MachineFunctionPass {
    /// Information about the offset and size of a single basic block.
    struct BasicBlockInfo {
      /// Distance from the beginning of the function to the beginning of this
      /// basic block.
      ///
      /// Offsets are computed assuming worst case padding before an aligned
      /// block. This means that subtracting basic block offsets always gives a
      /// conservative estimate of the real distance which may be smaller.
      ///
      /// Because worst case padding is used, the computed offset of an aligned
      /// block may not actually be aligned.
      unsigned Offset;

      /// Size of the basic block in bytes.  If the block contains inline
      /// assembly, this is a worst case estimate.
      ///
      /// The size does not include any alignment padding whether from the
      /// beginning of the block, or from an aligned jump table at the end.
      unsigned Size;

      /// The number of low bits in Offset that are known to be exact.  The
      /// remaining bits of Offset are an upper bound.
      uint8_t KnownBits;

      /// When non-zero, the block contains instructions (inline asm) of unknown
      /// size.  The real size may be smaller than Size bytes by a multiple of 1
      /// << Unalign.
      uint8_t Unalign;

      BasicBlockInfo() : Offset(0), Size(0), KnownBits(0), Unalign(0) {}

      /// Compute the number of known offset bits internally to this block.
      /// This number should be used to predict worst case padding when
      /// splitting the block.
      unsigned internalKnownBits() const {
        unsigned Bits = Unalign ? Unalign : KnownBits;
        // If the block size isn't a multiple of the known bits, assume the
        // worst case padding.
        if (Size & ((1u << Bits) - 1))
          Bits = CountTrailingZeros_32(Size);
        return Bits;
      }

      /// Compute the offset immediately following this block.  If LogAlign is
      /// specified, return the offset the successor block will get if it has
      /// this alignment.
      unsigned postOffset(unsigned LogAlign = 0) const {
        unsigned PO = Offset + Size;
        if (!LogAlign)
          return PO;
        // Add alignment padding from the terminator.
        return PO + UnknownPadding(LogAlign, internalKnownBits());
      }

      /// Compute the number of known low bits of postOffset.  If this block
      /// contains inline asm, the number of known bits drops to the
      /// instruction alignment.  An aligned terminator may increase the number
      /// of know bits.
      /// If LogAlign is given, also consider the alignment of the next block.
      unsigned postKnownBits(unsigned LogAlign = 0) const {
        return std::max(LogAlign, internalKnownBits());
      }
    };

    std::vector<BasicBlockInfo> BBInfo;

    /// One per immediate branch, keeping the machine instruction pointer,
    /// conditional or unconditional, the max displacement, and (if IsCond is
    /// true) the corresponding inverted branch opcode.
    struct ImmBranch {
      MachineInstr *MI;
      unsigned OffsetBits : 31;
      bool IsCond : 1;
      ImmBranch(MachineInstr *mi, unsigned offsetbits, bool cond)
        : MI(mi), OffsetBits(offsetbits), IsCond(cond) {}
    };

    /// Keep track of all the immediate branch instructions.
    ///
    std::vector<ImmBranch> ImmBranches;

    MachineFunction *MF;
    const AArch64InstrInfo *TII;
  public:
    static char ID;
    AArch64BranchFixup() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "AArch64 branch fixup pass";
    }

  private:
    void initializeFunctionInfo();
    MachineBasicBlock *splitBlockBeforeInstr(MachineInstr *MI);
    void adjustBBOffsetsAfter(MachineBasicBlock *BB);
    bool isBBInRange(MachineInstr *MI, MachineBasicBlock *BB,
                     unsigned OffsetBits);
    bool fixupImmediateBr(ImmBranch &Br);
    bool fixupConditionalBr(ImmBranch &Br);

    void computeBlockSize(MachineBasicBlock *MBB);
    unsigned getOffsetOf(MachineInstr *MI) const;
    void dumpBBs();
    void verify();
  };
  char AArch64BranchFixup::ID = 0;
}

/// check BBOffsets
void AArch64BranchFixup::verify() {
#ifndef NDEBUG
  for (MachineFunction::iterator MBBI = MF->begin(), E = MF->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    unsigned MBBId = MBB->getNumber();
    assert(!MBBId || BBInfo[MBBId - 1].postOffset() <= BBInfo[MBBId].Offset);
  }
#endif
}

/// print block size and offset information - debugging
void AArch64BranchFixup::dumpBBs() {
  DEBUG({
    for (unsigned J = 0, E = BBInfo.size(); J !=E; ++J) {
      const BasicBlockInfo &BBI = BBInfo[J];
      dbgs() << format("%08x BB#%u\t", BBI.Offset, J)
             << " kb=" << unsigned(BBI.KnownBits)
             << " ua=" << unsigned(BBI.Unalign)
             << format(" size=%#x\n", BBInfo[J].Size);
    }
  });
}

/// Returns an instance of the branch fixup pass.
FunctionPass *llvm::createAArch64BranchFixupPass() {
  return new AArch64BranchFixup();
}

bool AArch64BranchFixup::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  DEBUG(dbgs() << "***** AArch64BranchFixup ******");
  TII = (const AArch64InstrInfo*)MF->getTarget().getInstrInfo();

  // This pass invalidates liveness information when it splits basic blocks.
  MF->getRegInfo().invalidateLiveness();

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  MF->RenumberBlocks();

  // Do the initial scan of the function, building up information about the
  // sizes of each block and location of each immediate branch.
  initializeFunctionInfo();

  // Iteratively fix up branches until there is no change.
  unsigned NoBRIters = 0;
  bool MadeChange = false;
  while (true) {
    DEBUG(dbgs() << "Beginning iteration #" << NoBRIters << '\n');
    bool BRChange = false;
    for (unsigned i = 0, e = ImmBranches.size(); i != e; ++i)
      BRChange |= fixupImmediateBr(ImmBranches[i]);
    if (BRChange && ++NoBRIters > 30)
      report_fatal_error("Branch Fix Up pass failed to converge!");
    DEBUG(dumpBBs());

    if (!BRChange)
      break;
    MadeChange = true;
  }

  // After a while, this might be made debug-only, but it is not expensive.
  verify();

  DEBUG(dbgs() << '\n'; dumpBBs());

  BBInfo.clear();
  ImmBranches.clear();

  return MadeChange;
}

/// Return true if the specified basic block can fallthrough into the block
/// immediately after it.
static bool BBHasFallthrough(MachineBasicBlock *MBB) {
  // Get the next machine basic block in the function.
  MachineFunction::iterator MBBI = MBB;
  // Can't fall off end of function.
  if (llvm::next(MBBI) == MBB->getParent()->end())
    return false;

  MachineBasicBlock *NextBB = llvm::next(MBBI);
  for (MachineBasicBlock::succ_iterator I = MBB->succ_begin(),
       E = MBB->succ_end(); I != E; ++I)
    if (*I == NextBB)
      return true;

  return false;
}

/// Do the initial scan of the function, building up information about the sizes
/// of each block, and each immediate branch.
void AArch64BranchFixup::initializeFunctionInfo() {
  BBInfo.clear();
  BBInfo.resize(MF->getNumBlockIDs());

  // First thing, compute the size of all basic blocks, and see if the function
  // has any inline assembly in it. If so, we have to be conservative about
  // alignment assumptions, as we don't know for sure the size of any
  // instructions in the inline assembly.
  for (MachineFunction::iterator I = MF->begin(), E = MF->end(); I != E; ++I)
    computeBlockSize(I);

  // The known bits of the entry block offset are determined by the function
  // alignment.
  BBInfo.front().KnownBits = MF->getAlignment();

  // Compute block offsets and known bits.
  adjustBBOffsetsAfter(MF->begin());

  // Now go back through the instructions and build up our data structures.
  for (MachineFunction::iterator MBBI = MF->begin(), E = MF->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock &MBB = *MBBI;

    for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
         I != E; ++I) {
      if (I->isDebugValue())
        continue;

      int Opc = I->getOpcode();
      if (I->isBranch()) {
        bool IsCond = false;

        // The offsets encoded in instructions here scale by the instruction
        // size (4 bytes), effectively increasing their range by 2 bits.
        unsigned Bits = 0;
        switch (Opc) {
        default:
          continue;  // Ignore other JT branches
        case AArch64::TBZxii:
        case AArch64::TBZwii:
        case AArch64::TBNZxii:
        case AArch64::TBNZwii:
          IsCond = true;
          Bits = 14 + 2;
          break;
        case AArch64::Bcc:
        case AArch64::CBZx:
        case AArch64::CBZw:
        case AArch64::CBNZx:
        case AArch64::CBNZw:
          IsCond = true;
          Bits = 19 + 2;
          break;
        case AArch64::Bimm:
          Bits = 26 + 2;
          break;
        }

        // Record this immediate branch.
        ImmBranches.push_back(ImmBranch(I, Bits, IsCond));
      }
    }
  }
}

/// Compute the size and some alignment information for MBB.  This function
/// updates BBInfo directly.
void AArch64BranchFixup::computeBlockSize(MachineBasicBlock *MBB) {
  BasicBlockInfo &BBI = BBInfo[MBB->getNumber()];
  BBI.Size = 0;
  BBI.Unalign = 0;

  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
       ++I) {
    BBI.Size += TII->getInstSizeInBytes(*I);
    // For inline asm, GetInstSizeInBytes returns a conservative estimate.
    // The actual size may be smaller, but still a multiple of the instr size.
    if (I->isInlineAsm())
      BBI.Unalign = 2;
  }
}

/// Return the current offset of the specified machine instruction from the
/// start of the function.  This offset changes as stuff is moved around inside
/// the function.
unsigned AArch64BranchFixup::getOffsetOf(MachineInstr *MI) const {
  MachineBasicBlock *MBB = MI->getParent();

  // The offset is composed of two things: the sum of the sizes of all MBB's
  // before this instruction's block, and the offset from the start of the block
  // it is in.
  unsigned Offset = BBInfo[MBB->getNumber()].Offset;

  // Sum instructions before MI in MBB.
  for (MachineBasicBlock::iterator I = MBB->begin(); &*I != MI; ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    Offset += TII->getInstSizeInBytes(*I);
  }
  return Offset;
}

/// Split the basic block containing MI into two blocks, which are joined by
/// an unconditional branch.  Update data structures and renumber blocks to
/// account for this change and returns the newly created block.
MachineBasicBlock *
AArch64BranchFixup::splitBlockBeforeInstr(MachineInstr *MI) {
  MachineBasicBlock *OrigBB = MI->getParent();

  // Create a new MBB for the code after the OrigBB.
  MachineBasicBlock *NewBB =
    MF->CreateMachineBasicBlock(OrigBB->getBasicBlock());
  MachineFunction::iterator MBBI = OrigBB; ++MBBI;
  MF->insert(MBBI, NewBB);

  // Splice the instructions starting with MI over to NewBB.
  NewBB->splice(NewBB->end(), OrigBB, MI, OrigBB->end());

  // Add an unconditional branch from OrigBB to NewBB.
  // Note the new unconditional branch is not being recorded.
  // There doesn't seem to be meaningful DebugInfo available; this doesn't
  // correspond to anything in the source.
  BuildMI(OrigBB, DebugLoc(), TII->get(AArch64::Bimm)).addMBB(NewBB);
  ++NumSplit;

  // Update the CFG.  All succs of OrigBB are now succs of NewBB.
  NewBB->transferSuccessors(OrigBB);

  // OrigBB branches to NewBB.
  OrigBB->addSuccessor(NewBB);

  // Update internal data structures to account for the newly inserted MBB.
  MF->RenumberBlocks(NewBB);

  // Insert an entry into BBInfo to align it properly with the (newly
  // renumbered) block numbers.
  BBInfo.insert(BBInfo.begin() + NewBB->getNumber(), BasicBlockInfo());

  // Figure out how large the OrigBB is.  As the first half of the original
  // block, it cannot contain a tablejump.  The size includes
  // the new jump we added.  (It should be possible to do this without
  // recounting everything, but it's very confusing, and this is rarely
  // executed.)
  computeBlockSize(OrigBB);

  // Figure out how large the NewMBB is.  As the second half of the original
  // block, it may contain a tablejump.
  computeBlockSize(NewBB);

  // All BBOffsets following these blocks must be modified.
  adjustBBOffsetsAfter(OrigBB);

  return NewBB;
}

void AArch64BranchFixup::adjustBBOffsetsAfter(MachineBasicBlock *BB) {
  unsigned BBNum = BB->getNumber();
  for(unsigned i = BBNum + 1, e = MF->getNumBlockIDs(); i < e; ++i) {
    // Get the offset and known bits at the end of the layout predecessor.
    // Include the alignment of the current block.
    unsigned LogAlign = MF->getBlockNumbered(i)->getAlignment();
    unsigned Offset = BBInfo[i - 1].postOffset(LogAlign);
    unsigned KnownBits = BBInfo[i - 1].postKnownBits(LogAlign);

    // This is where block i begins.  Stop if the offset is already correct,
    // and we have updated 2 blocks.  This is the maximum number of blocks
    // changed before calling this function.
    if (i > BBNum + 2 &&
        BBInfo[i].Offset == Offset &&
        BBInfo[i].KnownBits == KnownBits)
      break;

    BBInfo[i].Offset = Offset;
    BBInfo[i].KnownBits = KnownBits;
  }
}

/// Returns true if the distance between specific MI and specific BB can fit in
/// MI's displacement field.
bool AArch64BranchFixup::isBBInRange(MachineInstr *MI,
                                     MachineBasicBlock *DestBB,
                                     unsigned OffsetBits) {
  int64_t BrOffset   = getOffsetOf(MI);
  int64_t DestOffset = BBInfo[DestBB->getNumber()].Offset;

  DEBUG(dbgs() << "Branch of destination BB#" << DestBB->getNumber()
               << " from BB#" << MI->getParent()->getNumber()
               << " bits available=" << OffsetBits
               << " from " << getOffsetOf(MI) << " to " << DestOffset
               << " offset " << int(DestOffset-BrOffset) << "\t" << *MI);

  return isIntN(OffsetBits, DestOffset - BrOffset);
}

/// Fix up an immediate branch whose destination is too far away to fit in its
/// displacement field.
bool AArch64BranchFixup::fixupImmediateBr(ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *DestBB = 0;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    if (MI->getOperand(i).isMBB()) {
      DestBB = MI->getOperand(i).getMBB();
      break;
    }
  }
  assert(DestBB && "Branch with no destination BB?");

  // Check to see if the DestBB is already in-range.
  if (isBBInRange(MI, DestBB, Br.OffsetBits))
    return false;

  assert(Br.IsCond && "Only conditional branches should need fixup");
  return fixupConditionalBr(Br);
}

/// Fix up a conditional branch whose destination is too far away to fit in its
/// displacement field. It is converted to an inverse conditional branch + an
/// unconditional branch to the destination.
bool
AArch64BranchFixup::fixupConditionalBr(ImmBranch &Br) {
  MachineInstr *MI = Br.MI;
  MachineBasicBlock *MBB = MI->getParent();
  unsigned CondBrMBBOperand = 0;

  // The general idea is to add an unconditional branch to the destination and
  // invert the conditional branch to jump over it. Complications occur around
  // fallthrough and unreachable ends to the block.
  //   b.lt L1
  //   =>
  //   b.ge L2
  //   b   L1
  // L2:

  // First we invert the conditional branch, by creating a replacement if
  // necessary. This if statement contains all the special handling of different
  // branch types.
  if (MI->getOpcode() == AArch64::Bcc) {
    // The basic block is operand number 1 for Bcc
    CondBrMBBOperand = 1;

    A64CC::CondCodes CC = (A64CC::CondCodes)MI->getOperand(0).getImm();
    CC = A64InvertCondCode(CC);
    MI->getOperand(0).setImm(CC);
  } else {
    MachineInstrBuilder InvertedMI;
    int InvertedOpcode;
    switch (MI->getOpcode()) {
    default: llvm_unreachable("Unknown branch type");
    case AArch64::TBZxii: InvertedOpcode = AArch64::TBNZxii; break;
    case AArch64::TBZwii: InvertedOpcode = AArch64::TBNZwii; break;
    case AArch64::TBNZxii: InvertedOpcode = AArch64::TBZxii; break;
    case AArch64::TBNZwii: InvertedOpcode = AArch64::TBZwii; break;
    case AArch64::CBZx: InvertedOpcode = AArch64::CBNZx; break;
    case AArch64::CBZw: InvertedOpcode = AArch64::CBNZw; break;
    case AArch64::CBNZx: InvertedOpcode = AArch64::CBZx; break;
    case AArch64::CBNZw: InvertedOpcode = AArch64::CBZw; break;
    }

    InvertedMI = BuildMI(*MBB, MI, MI->getDebugLoc(), TII->get(InvertedOpcode));
    for (unsigned i = 0, e= MI->getNumOperands(); i != e; ++i) {
      InvertedMI.addOperand(MI->getOperand(i));
      if (MI->getOperand(i).isMBB())
        CondBrMBBOperand = i;
    }

    MI->eraseFromParent();
    MI = Br.MI = InvertedMI;
  }

  // If the branch is at the end of its MBB and that has a fall-through block,
  // direct the updated conditional branch to the fall-through
  // block. Otherwise, split the MBB before the next instruction.
  MachineInstr *BMI = &MBB->back();
  bool NeedSplit = (BMI != MI) || !BBHasFallthrough(MBB);

  ++NumCBrFixed;
  if (BMI != MI) {
    if (llvm::next(MachineBasicBlock::iterator(MI)) == prior(MBB->end()) &&
        BMI->getOpcode() == AArch64::Bimm) {
      // Last MI in the BB is an unconditional branch. We can swap destinations:
      // b.eq L1 (temporarily b.ne L1 after first change)
      // b   L2
      // =>
      // b.ne L2
      // b   L1
      MachineBasicBlock *NewDest = BMI->getOperand(0).getMBB();
      if (isBBInRange(MI, NewDest, Br.OffsetBits)) {
        DEBUG(dbgs() << "  Invert Bcc condition and swap its destination with "
                     << *BMI);
        MachineBasicBlock *DestBB = MI->getOperand(CondBrMBBOperand).getMBB();
        BMI->getOperand(0).setMBB(DestBB);
        MI->getOperand(CondBrMBBOperand).setMBB(NewDest);
        return true;
      }
    }
  }

  if (NeedSplit) {
    MachineBasicBlock::iterator MBBI = MI; ++MBBI;
    splitBlockBeforeInstr(MBBI);
    // No need for the branch to the next block. We're adding an unconditional
    // branch to the destination.
    int delta = TII->getInstSizeInBytes(MBB->back());
    BBInfo[MBB->getNumber()].Size -= delta;
    MBB->back().eraseFromParent();
    // BBInfo[SplitBB].Offset is wrong temporarily, fixed below
  }

  // After splitting and removing the unconditional branch from the original BB,
  // the structure is now:
  // oldbb:
  //   [things]
  //   b.invertedCC L1
  // splitbb/fallthroughbb:
  //   [old b L2/real continuation]
  //
  // We now have to change the conditional branch to point to splitbb and add an
  // unconditional branch after it to L1, giving the final structure:
  // oldbb:
  //   [things]
  //   b.invertedCC splitbb
  //   b L1
  // splitbb/fallthroughbb:
  //   [old b L2/real continuation]
  MachineBasicBlock *NextBB = llvm::next(MachineFunction::iterator(MBB));

  DEBUG(dbgs() << "  Insert B to BB#"
               << MI->getOperand(CondBrMBBOperand).getMBB()->getNumber()
               << " also invert condition and change dest. to BB#"
               << NextBB->getNumber() << "\n");

  // Insert a new unconditional branch and fixup the destination of the
  // conditional one.  Also update the ImmBranch as well as adding a new entry
  // for the new branch.
  BuildMI(MBB, DebugLoc(), TII->get(AArch64::Bimm))
    .addMBB(MI->getOperand(CondBrMBBOperand).getMBB());
  MI->getOperand(CondBrMBBOperand).setMBB(NextBB);

  BBInfo[MBB->getNumber()].Size += TII->getInstSizeInBytes(MBB->back());

  // 26 bits written down in Bimm, specifying a multiple of 4.
  unsigned OffsetBits = 26 + 2;
  ImmBranches.push_back(ImmBranch(&MBB->back(), OffsetBits, false));

  adjustBBOffsetsAfter(MBB);
  return true;
}
