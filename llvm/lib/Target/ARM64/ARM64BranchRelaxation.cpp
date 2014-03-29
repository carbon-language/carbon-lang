//===-- ARM64BranchRelaxation.cpp - ARM64 branch relaxation ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "arm64-branch-relax"
#include "ARM64.h"
#include "ARM64InstrInfo.h"
#include "ARM64MachineFunctionInfo.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

static cl::opt<bool>
BranchRelaxation("arm64-branch-relax", cl::Hidden, cl::init(true),
                 cl::desc("Relax out of range conditional branches"));

static cl::opt<unsigned>
TBZDisplacementBits("arm64-tbz-offset-bits", cl::Hidden, cl::init(14),
                    cl::desc("Restrict range of TB[N]Z instructions (DEBUG)"));

static cl::opt<unsigned>
CBZDisplacementBits("arm64-cbz-offset-bits", cl::Hidden, cl::init(19),
                    cl::desc("Restrict range of CB[N]Z instructions (DEBUG)"));

static cl::opt<unsigned>
BCCDisplacementBits("arm64-bcc-offset-bits", cl::Hidden, cl::init(19),
                    cl::desc("Restrict range of Bcc instructions (DEBUG)"));

STATISTIC(NumSplit, "Number of basic blocks split");
STATISTIC(NumRelaxed, "Number of conditional branches relaxed");

namespace {
class ARM64BranchRelaxation : public MachineFunctionPass {
  /// BasicBlockInfo - Information about the offset and size of a single
  /// basic block.
  struct BasicBlockInfo {
    /// Offset - Distance from the beginning of the function to the beginning
    /// of this basic block.
    ///
    /// The offset is always aligned as required by the basic block.
    unsigned Offset;

    /// Size - Size of the basic block in bytes.  If the block contains
    /// inline assembly, this is a worst case estimate.
    ///
    /// The size does not include any alignment padding whether from the
    /// beginning of the block, or from an aligned jump table at the end.
    unsigned Size;

    BasicBlockInfo() : Offset(0), Size(0) {}

    /// Compute the offset immediately following this block.  If LogAlign is
    /// specified, return the offset the successor block will get if it has
    /// this alignment.
    unsigned postOffset(unsigned LogAlign = 0) const {
      unsigned PO = Offset + Size;
      unsigned Align = 1 << LogAlign;
      return (PO + Align - 1) / Align * Align;
    }
  };

  SmallVector<BasicBlockInfo, 16> BlockInfo;

  MachineFunction *MF;
  const ARM64InstrInfo *TII;

  bool relaxBranchInstructions();
  void scanFunction();
  MachineBasicBlock *splitBlockBeforeInstr(MachineInstr *MI);
  void adjustBlockOffsets(MachineBasicBlock *BB);
  bool isBlockInRange(MachineInstr *MI, MachineBasicBlock *BB, unsigned Disp);
  bool fixupConditionalBranch(MachineInstr *MI);
  void computeBlockSize(MachineBasicBlock *MBB);
  unsigned getInstrOffset(MachineInstr *MI) const;
  void dumpBBs();
  void verify();

public:
  static char ID;
  ARM64BranchRelaxation() : MachineFunctionPass(ID) {}

  virtual bool runOnMachineFunction(MachineFunction &MF);

  virtual const char *getPassName() const {
    return "ARM64 branch relaxation pass";
  }
};
char ARM64BranchRelaxation::ID = 0;
}

/// verify - check BBOffsets, BBSizes, alignment of islands
void ARM64BranchRelaxation::verify() {
#ifndef NDEBUG
  unsigned PrevNum = MF->begin()->getNumber();
  for (MachineFunction::iterator MBBI = MF->begin(), E = MF->end(); MBBI != E;
       ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    unsigned Align = MBB->getAlignment();
    unsigned Num = MBB->getNumber();
    assert(BlockInfo[Num].Offset % (1u << Align) == 0);
    assert(!Num || BlockInfo[PrevNum].postOffset() <= BlockInfo[Num].Offset);
    PrevNum = Num;
  }
#endif
}

/// print block size and offset information - debugging
void ARM64BranchRelaxation::dumpBBs() {
  for (MachineFunction::iterator MBBI = MF->begin(), E = MF->end(); MBBI != E;
       ++MBBI) {
    const BasicBlockInfo &BBI = BlockInfo[MBBI->getNumber()];
    dbgs() << format("BB#%u\toffset=%08x\t", MBBI->getNumber(), BBI.Offset)
           << format("size=%#x\n", BBI.Size);
  }
}

/// BBHasFallthrough - Return true if the specified basic block can fallthrough
/// into the block immediately after it.
static bool BBHasFallthrough(MachineBasicBlock *MBB) {
  // Get the next machine basic block in the function.
  MachineFunction::iterator MBBI = MBB;
  // Can't fall off end of function.
  if (std::next(MBBI) == MBB->getParent()->end())
    return false;

  MachineBasicBlock *NextBB = std::next(MBBI);
  for (MachineBasicBlock::succ_iterator I = MBB->succ_begin(),
                                        E = MBB->succ_end();
       I != E; ++I)
    if (*I == NextBB)
      return true;

  return false;
}

/// scanFunction - Do the initial scan of the function, building up
/// information about each block.
void ARM64BranchRelaxation::scanFunction() {
  BlockInfo.clear();
  BlockInfo.resize(MF->getNumBlockIDs());

  // First thing, compute the size of all basic blocks, and see if the function
  // has any inline assembly in it. If so, we have to be conservative about
  // alignment assumptions, as we don't know for sure the size of any
  // instructions in the inline assembly.
  for (MachineFunction::iterator I = MF->begin(), E = MF->end(); I != E; ++I)
    computeBlockSize(I);

  // Compute block offsets and known bits.
  adjustBlockOffsets(MF->begin());
}

/// computeBlockSize - Compute the size for MBB.
/// This function updates BlockInfo directly.
void ARM64BranchRelaxation::computeBlockSize(MachineBasicBlock *MBB) {
  unsigned Size = 0;
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
       ++I)
    Size += TII->GetInstSizeInBytes(I);
  BlockInfo[MBB->getNumber()].Size = Size;
}

/// getInstrOffset - Return the current offset of the specified machine
/// instruction from the start of the function.  This offset changes as stuff is
/// moved around inside the function.
unsigned ARM64BranchRelaxation::getInstrOffset(MachineInstr *MI) const {
  MachineBasicBlock *MBB = MI->getParent();

  // The offset is composed of two things: the sum of the sizes of all MBB's
  // before this instruction's block, and the offset from the start of the block
  // it is in.
  unsigned Offset = BlockInfo[MBB->getNumber()].Offset;

  // Sum instructions before MI in MBB.
  for (MachineBasicBlock::iterator I = MBB->begin(); &*I != MI; ++I) {
    assert(I != MBB->end() && "Didn't find MI in its own basic block?");
    Offset += TII->GetInstSizeInBytes(I);
  }
  return Offset;
}

void ARM64BranchRelaxation::adjustBlockOffsets(MachineBasicBlock *Start) {
  unsigned PrevNum = Start->getNumber();
  MachineFunction::iterator MBBI = Start, E = MF->end();
  for (++MBBI; MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    unsigned Num = MBB->getNumber();
    if (!Num) // block zero is never changed from offset zero.
      continue;
    // Get the offset and known bits at the end of the layout predecessor.
    // Include the alignment of the current block.
    unsigned LogAlign = MBBI->getAlignment();
    BlockInfo[Num].Offset = BlockInfo[PrevNum].postOffset(LogAlign);
    PrevNum = Num;
  }
}

/// Split the basic block containing MI into two blocks, which are joined by
/// an unconditional branch.  Update data structures and renumber blocks to
/// account for this change and returns the newly created block.
/// NOTE: Successor list of the original BB is out of date after this function,
/// and must be updated by the caller! Other transforms follow using this
/// utility function, so no point updating now rather than waiting.
MachineBasicBlock *
ARM64BranchRelaxation::splitBlockBeforeInstr(MachineInstr *MI) {
  MachineBasicBlock *OrigBB = MI->getParent();

  // Create a new MBB for the code after the OrigBB.
  MachineBasicBlock *NewBB =
      MF->CreateMachineBasicBlock(OrigBB->getBasicBlock());
  MachineFunction::iterator MBBI = OrigBB;
  ++MBBI;
  MF->insert(MBBI, NewBB);

  // Splice the instructions starting with MI over to NewBB.
  NewBB->splice(NewBB->end(), OrigBB, MI, OrigBB->end());

  // Add an unconditional branch from OrigBB to NewBB.
  // Note the new unconditional branch is not being recorded.
  // There doesn't seem to be meaningful DebugInfo available; this doesn't
  // correspond to anything in the source.
  BuildMI(OrigBB, DebugLoc(), TII->get(ARM64::B)).addMBB(NewBB);

  // Insert an entry into BlockInfo to align it properly with the block numbers.
  BlockInfo.insert(BlockInfo.begin() + NewBB->getNumber(), BasicBlockInfo());

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
  adjustBlockOffsets(OrigBB);

  ++NumSplit;

  return NewBB;
}

/// isBlockInRange - Returns true if the distance between specific MI and
/// specific BB can fit in MI's displacement field.
bool ARM64BranchRelaxation::isBlockInRange(MachineInstr *MI,
                                           MachineBasicBlock *DestBB,
                                           unsigned Bits) {
  unsigned MaxOffs = ((1 << (Bits - 1)) - 1) << 2;
  unsigned BrOffset = getInstrOffset(MI);
  unsigned DestOffset = BlockInfo[DestBB->getNumber()].Offset;

  DEBUG(dbgs() << "Branch of destination BB#" << DestBB->getNumber()
               << " from BB#" << MI->getParent()->getNumber()
               << " max delta=" << MaxOffs << " from " << getInstrOffset(MI)
               << " to " << DestOffset << " offset "
               << int(DestOffset - BrOffset) << "\t" << *MI);

  // Branch before the Dest.
  if (BrOffset <= DestOffset)
    return (DestOffset - BrOffset <= MaxOffs);
  return (BrOffset - DestOffset <= MaxOffs);
}

static bool isConditionalBranch(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case ARM64::TBZ:
  case ARM64::TBNZ:
  case ARM64::CBZW:
  case ARM64::CBNZW:
  case ARM64::CBZX:
  case ARM64::CBNZX:
  case ARM64::Bcc:
    return true;
  }
}

static MachineBasicBlock *getDestBlock(MachineInstr *MI) {
  switch (MI->getOpcode()) {
  default:
    assert(0 && "unexpected opcode!");
  case ARM64::TBZ:
  case ARM64::TBNZ:
    return MI->getOperand(2).getMBB();
  case ARM64::CBZW:
  case ARM64::CBNZW:
  case ARM64::CBZX:
  case ARM64::CBNZX:
  case ARM64::Bcc:
    return MI->getOperand(1).getMBB();
  }
}

static unsigned getOppositeConditionOpcode(unsigned Opc) {
  switch (Opc) {
  default:
    assert(0 && "unexpected opcode!");
  case ARM64::TBNZ:    return ARM64::TBZ;
  case ARM64::TBZ:     return ARM64::TBNZ;
  case ARM64::CBNZW:   return ARM64::CBZW;
  case ARM64::CBNZX:   return ARM64::CBZX;
  case ARM64::CBZW:    return ARM64::CBNZW;
  case ARM64::CBZX:    return ARM64::CBNZX;
  case ARM64::Bcc:     return ARM64::Bcc; // Condition is an operand for Bcc.
  }
}

static unsigned getBranchDisplacementBits(unsigned Opc) {
  switch (Opc) {
  default:
    assert(0 && "unexpected opcode!");
  case ARM64::TBNZ:
  case ARM64::TBZ:
    return TBZDisplacementBits;
  case ARM64::CBNZW:
  case ARM64::CBZW:
  case ARM64::CBNZX:
  case ARM64::CBZX:
    return CBZDisplacementBits;
  case ARM64::Bcc:
    return BCCDisplacementBits;
  }
}

static inline void invertBccCondition(MachineInstr *MI) {
  assert(MI->getOpcode() == ARM64::Bcc && "Unexpected opcode!");
  ARM64CC::CondCode CC = (ARM64CC::CondCode)MI->getOperand(0).getImm();
  CC = ARM64CC::getInvertedCondCode(CC);
  MI->getOperand(0).setImm((int64_t)CC);
}

/// fixupConditionalBranch - Fix up a conditional branch whose destination is
/// too far away to fit in its displacement field. It is converted to an inverse
/// conditional branch + an unconditional branch to the destination.
bool ARM64BranchRelaxation::fixupConditionalBranch(MachineInstr *MI) {
  MachineBasicBlock *DestBB = getDestBlock(MI);

  // Add an unconditional branch to the destination and invert the branch
  // condition to jump over it:
  // tbz L1
  // =>
  // tbnz L2
  // b   L1
  // L2:

  // If the branch is at the end of its MBB and that has a fall-through block,
  // direct the updated conditional branch to the fall-through block. Otherwise,
  // split the MBB before the next instruction.
  MachineBasicBlock *MBB = MI->getParent();
  MachineInstr *BMI = &MBB->back();
  bool NeedSplit = (BMI != MI) || !BBHasFallthrough(MBB);

  if (BMI != MI) {
    if (std::next(MachineBasicBlock::iterator(MI)) ==
            std::prev(MBB->getLastNonDebugInstr()) &&
        BMI->getOpcode() == ARM64::B) {
      // Last MI in the BB is an unconditional branch. Can we simply invert the
      // condition and swap destinations:
      // beq L1
      // b   L2
      // =>
      // bne L2
      // b   L1
      MachineBasicBlock *NewDest = BMI->getOperand(0).getMBB();
      if (isBlockInRange(MI, NewDest,
                         getBranchDisplacementBits(MI->getOpcode()))) {
        DEBUG(dbgs() << "  Invert condition and swap its destination with "
                     << *BMI);
        BMI->getOperand(0).setMBB(DestBB);
        unsigned OpNum =
            (MI->getOpcode() == ARM64::TBZ || MI->getOpcode() == ARM64::TBNZ)
                ? 2
                : 1;
        MI->getOperand(OpNum).setMBB(NewDest);
        MI->setDesc(TII->get(getOppositeConditionOpcode(MI->getOpcode())));
        if (MI->getOpcode() == ARM64::Bcc)
          invertBccCondition(MI);
        return true;
      }
    }
  }

  if (NeedSplit) {
    // Analyze the branch so we know how to update the successor lists.
    MachineBasicBlock *TBB, *FBB;
    SmallVector<MachineOperand, 2> Cond;
    TII->AnalyzeBranch(*MBB, TBB, FBB, Cond, false);

    MachineBasicBlock *NewBB = splitBlockBeforeInstr(MI);
    // No need for the branch to the next block. We're adding an unconditional
    // branch to the destination.
    int delta = TII->GetInstSizeInBytes(&MBB->back());
    BlockInfo[MBB->getNumber()].Size -= delta;
    MBB->back().eraseFromParent();
    // BlockInfo[SplitBB].Offset is wrong temporarily, fixed below

    // Update the successor lists according to the transformation to follow.
    // Do it here since if there's no split, no update is needed.
    MBB->replaceSuccessor(FBB, NewBB);
    NewBB->addSuccessor(FBB);
  }
  MachineBasicBlock *NextBB = std::next(MachineFunction::iterator(MBB));

  DEBUG(dbgs() << "  Insert B to BB#" << DestBB->getNumber()
               << ", invert condition and change dest. to BB#"
               << NextBB->getNumber() << "\n");

  // Insert a new conditional branch and a new unconditional branch.
  MachineInstrBuilder MIB = BuildMI(
      MBB, DebugLoc(), TII->get(getOppositeConditionOpcode(MI->getOpcode())))
                                .addOperand(MI->getOperand(0));
  if (MI->getOpcode() == ARM64::TBZ || MI->getOpcode() == ARM64::TBNZ)
    MIB.addOperand(MI->getOperand(1));
  if (MI->getOpcode() == ARM64::Bcc)
    invertBccCondition(MIB);
  MIB.addMBB(NextBB);
  BlockInfo[MBB->getNumber()].Size += TII->GetInstSizeInBytes(&MBB->back());
  BuildMI(MBB, DebugLoc(), TII->get(ARM64::B)).addMBB(DestBB);
  BlockInfo[MBB->getNumber()].Size += TII->GetInstSizeInBytes(&MBB->back());

  // Remove the old conditional branch.  It may or may not still be in MBB.
  BlockInfo[MI->getParent()->getNumber()].Size -= TII->GetInstSizeInBytes(MI);
  MI->eraseFromParent();

  // Finally, keep the block offsets up to date.
  adjustBlockOffsets(MBB);
  return true;
}

bool ARM64BranchRelaxation::relaxBranchInstructions() {
  bool Changed = false;
  // Relaxing branches involves creating new basic blocks, so re-eval
  // end() for termination.
  for (MachineFunction::iterator I = MF->begin(); I != MF->end(); ++I) {
    MachineInstr *MI = I->getFirstTerminator();
    if (isConditionalBranch(MI->getOpcode()) &&
        !isBlockInRange(MI, getDestBlock(MI),
                        getBranchDisplacementBits(MI->getOpcode()))) {
      fixupConditionalBranch(MI);
      ++NumRelaxed;
      Changed = true;
    }
  }
  return Changed;
}

bool ARM64BranchRelaxation::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;

  // If the pass is disabled, just bail early.
  if (!BranchRelaxation)
    return false;

  DEBUG(dbgs() << "***** ARM64BranchRelaxation *****\n");

  TII = (const ARM64InstrInfo *)MF->getTarget().getInstrInfo();

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  MF->RenumberBlocks();

  // Do the initial scan of the function, building up information about the
  // sizes of each block.
  scanFunction();

  DEBUG(dbgs() << "  Basic blocks before relaxation\n");
  DEBUG(dumpBBs());

  bool MadeChange = false;
  while (relaxBranchInstructions())
    MadeChange = true;

  // After a while, this might be made debug-only, but it is not expensive.
  verify();

  DEBUG(dbgs() << "  Basic blocks after relaxation\n");
  DEBUG(dbgs() << '\n'; dumpBBs());

  BlockInfo.clear();

  return MadeChange;
}

/// createARM64BranchRelaxation - returns an instance of the constpool
/// island pass.
FunctionPass *llvm::createARM64BranchRelaxation() {
  return new ARM64BranchRelaxation();
}
