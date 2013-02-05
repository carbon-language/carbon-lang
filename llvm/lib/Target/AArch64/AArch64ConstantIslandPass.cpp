//===-- AArch64ConstantIslandPass.cpp - AArch64 constant islands ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that splits the constant pool up into 'islands'
// which are scattered through-out the function.  This is required due to the
// limited pc-relative displacements that AArch64 has.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "aarch64-cp-islands"
#include "AArch64.h"
#include "AArch64InstrInfo.h"
#include "AArch64MachineFunctionInfo.h"
#include "AArch64Subtarget.h"
#include "AArch64MachineFunctionInfo.h"
#include "Utils/AArch64BaseInfo.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include <algorithm>
using namespace llvm;

STATISTIC(NumCPEs,       "Number of constpool entries");
STATISTIC(NumSplit,      "Number of uncond branches inserted");
STATISTIC(NumCBrFixed,   "Number of cond branches fixed");

// FIXME: This option should be removed once it has received sufficient testing.
static cl::opt<bool>
AlignConstantIslands("aarch64-align-constant-islands", cl::Hidden,
                     cl::init(true),
                     cl::desc("Align constant islands in code"));

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
  /// Due to limited PC-relative displacements, AArch64 requires constant pool
  /// entries to be scattered among the instructions inside a function.  To do
  /// this, it completely ignores the normal LLVM constant pool; instead, it
  /// places constants wherever it feels like with special instructions.
  ///
  /// The terminology used in this pass includes:
  ///   Islands - Clumps of constants placed in the function.
  ///   Water   - Potential places where an island could be formed.
  ///   CPE     - A constant pool entry that has been placed somewhere, which
  ///             tracks a list of users.
  class AArch64ConstantIslands : public MachineFunctionPass {
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

    /// A sorted list of basic blocks where islands could be placed (i.e. blocks
    /// that don't fall through to the following block, due to a return,
    /// unreachable, or unconditional branch).
    std::vector<MachineBasicBlock*> WaterList;

    /// The subset of WaterList that was created since the previous iteration by
    /// inserting unconditional branches.
    SmallSet<MachineBasicBlock*, 4> NewWaterList;

    typedef std::vector<MachineBasicBlock*>::iterator water_iterator;

    /// One user of a constant pool, keeping the machine instruction pointer,
    /// the constant pool being referenced, and the number of bits used by the
    /// instruction for displacement.  The HighWaterMark records the highest
    /// basic block where a new CPEntry can be placed.  To ensure this pass
    /// terminates, the CP entries are initially placed at the end of the
    /// function and then move monotonically to lower addresses.  The exception
    /// to this rule is when the current CP entry for a particular CPUser is out
    /// of range, but there is another CP entry for the same constant value in
    /// range.  We want to use the existing in-range CP entry, but if it later
    /// moves out of range, the search for new water should resume where it left
    /// off.  The HighWaterMark is used to record that point.
    struct CPUser {
      MachineInstr *MI;
      MachineInstr *CPEMI;
      MachineBasicBlock *HighWaterMark;
    private:
      unsigned OffsetBits;
    public:
      CPUser(MachineInstr *mi, MachineInstr *cpemi, unsigned offsetbits)
        : MI(mi), CPEMI(cpemi), OffsetBits(offsetbits) {
        HighWaterMark = CPEMI->getParent();
      }
      /// Returns the number of bits used to specify the offset.
      unsigned getOffsetBits() const {
        return OffsetBits;
      }

      /// Returns the maximum positive displacement possible from this CPUser
      /// (essentially INT<N>_MAX * 4).
      unsigned getMaxPosDisp() const {
        return (1 << (OffsetBits - 1)) - 1;
      }
    };

    /// Keep track of all of the machine instructions that use various constant
    /// pools and their max displacement.
    std::vector<CPUser> CPUsers;

    /// One per constant pool entry, keeping the machine instruction pointer,
    /// the constpool index, and the number of CPUser's which reference this
    /// entry.
    struct CPEntry {
      MachineInstr *CPEMI;
      unsigned CPI;
      unsigned RefCount;
      CPEntry(MachineInstr *cpemi, unsigned cpi, unsigned rc = 0)
        : CPEMI(cpemi), CPI(cpi), RefCount(rc) {}
    };

    ///  Keep track of all of the constant pool entry machine instructions. For
    /// each original constpool index (i.e. those that existed upon entry to
    /// this pass), it keeps a vector of entries.  Original elements are cloned
    /// as we go along; the clones are put in the vector of the original
    /// element, but have distinct CPIs.
    std::vector<std::vector<CPEntry> > CPEntries;

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
    MachineConstantPool *MCP;
    const AArch64InstrInfo *TII;
    const AArch64Subtarget *STI;
    AArch64MachineFunctionInfo *AFI;
  public:
    static char ID;
    AArch64ConstantIslands() : MachineFunctionPass(ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual const char *getPassName() const {
      return "AArch64 constant island placement pass";
    }

  private:
    void doInitialPlacement(std::vector<MachineInstr*> &CPEMIs);
    CPEntry *findConstPoolEntry(unsigned CPI, const MachineInstr *CPEMI);
    unsigned getCPELogAlign(const MachineInstr *CPEMI);
    void scanFunctionJumpTables();
    void initializeFunctionInfo(const std::vector<MachineInstr*> &CPEMIs);
    MachineBasicBlock *splitBlockBeforeInstr(MachineInstr *MI);
    void updateForInsertedWaterBlock(MachineBasicBlock *NewBB);
    void adjustBBOffsetsAfter(MachineBasicBlock *BB);
    bool decrementCPEReferenceCount(unsigned CPI, MachineInstr* CPEMI);
    int findInRangeCPEntry(CPUser& U, unsigned UserOffset);
    bool findAvailableWater(CPUser&U, unsigned UserOffset,
                            water_iterator &WaterIter);
    void createNewWater(unsigned CPUserIndex, unsigned UserOffset,
                        MachineBasicBlock *&NewMBB);
    bool handleConstantPoolUser(unsigned CPUserIndex);
    void removeDeadCPEMI(MachineInstr *CPEMI);
    bool removeUnusedCPEntries();
    bool isCPEntryInRange(MachineInstr *MI, unsigned UserOffset,
                          MachineInstr *CPEMI, unsigned OffsetBits,
                          bool DoDump = false);
    bool isWaterInRange(unsigned UserOffset, MachineBasicBlock *Water,
                        CPUser &U, unsigned &Growth);
    bool isBBInRange(MachineInstr *MI, MachineBasicBlock *BB,
                     unsigned OffsetBits);
    bool fixupImmediateBr(ImmBranch &Br);
    bool fixupConditionalBr(ImmBranch &Br);

    void computeBlockSize(MachineBasicBlock *MBB);
    unsigned getOffsetOf(MachineInstr *MI) const;
    unsigned getUserOffset(CPUser&) const;
    void dumpBBs();
    void verify();

    bool isOffsetInRange(unsigned UserOffset, unsigned TrialOffset,
                         unsigned BitsAvailable);
    bool isOffsetInRange(unsigned UserOffset, unsigned TrialOffset,
                         const CPUser &U) {
      return isOffsetInRange(UserOffset, TrialOffset, U.getOffsetBits());
    }
  };
  char AArch64ConstantIslands::ID = 0;
}

/// check BBOffsets, BBSizes, alignment of islands
void AArch64ConstantIslands::verify() {
#ifndef NDEBUG
  for (MachineFunction::iterator MBBI = MF->begin(), E = MF->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    unsigned MBBId = MBB->getNumber();
    assert(!MBBId || BBInfo[MBBId - 1].postOffset() <= BBInfo[MBBId].Offset);
  }
  DEBUG(dbgs() << "Verifying " << CPUsers.size() << " CP users.\n");
  for (unsigned i = 0, e = CPUsers.size(); i != e; ++i) {
    CPUser &U = CPUsers[i];
    unsigned UserOffset = getUserOffset(U);
    // Verify offset using the real max displacement without the safety
    // adjustment.
    if (isCPEntryInRange(U.MI, UserOffset, U.CPEMI, U.getOffsetBits(),
                         /* DoDump = */ true)) {
      DEBUG(dbgs() << "OK\n");
      continue;
    }
    DEBUG(dbgs() << "Out of range.\n");
    dumpBBs();
    DEBUG(MF->dump());
    llvm_unreachable("Constant pool entry out of range!");
  }
#endif
}

/// print block size and offset information - debugging
void AArch64ConstantIslands::dumpBBs() {
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

/// Returns an instance of the constpool island pass.
FunctionPass *llvm::createAArch64ConstantIslandPass() {
  return new AArch64ConstantIslands();
}

bool AArch64ConstantIslands::runOnMachineFunction(MachineFunction &mf) {
  MF = &mf;
  MCP = mf.getConstantPool();

  DEBUG(dbgs() << "***** AArch64ConstantIslands: "
               << MCP->getConstants().size() << " CP entries, aligned to "
               << MCP->getConstantPoolAlignment() << " bytes *****\n");

  TII = (const AArch64InstrInfo*)MF->getTarget().getInstrInfo();
  AFI = MF->getInfo<AArch64MachineFunctionInfo>();
  STI = &MF->getTarget().getSubtarget<AArch64Subtarget>();

  // This pass invalidates liveness information when it splits basic blocks.
  MF->getRegInfo().invalidateLiveness();

  // Renumber all of the machine basic blocks in the function, guaranteeing that
  // the numbers agree with the position of the block in the function.
  MF->RenumberBlocks();

  // Perform the initial placement of the constant pool entries.  To start with,
  // we put them all at the end of the function.
  std::vector<MachineInstr*> CPEMIs;
  if (!MCP->isEmpty())
    doInitialPlacement(CPEMIs);

  /// The next UID to take is the first unused one.
  AFI->initPICLabelUId(CPEMIs.size());

  // Do the initial scan of the function, building up information about the
  // sizes of each block, the location of all the water, and finding all of the
  // constant pool users.
  initializeFunctionInfo(CPEMIs);
  CPEMIs.clear();
  DEBUG(dumpBBs());


  /// Remove dead constant pool entries.
  bool MadeChange = removeUnusedCPEntries();

  // Iteratively place constant pool entries and fix up branches until there
  // is no change.
  unsigned NoCPIters = 0, NoBRIters = 0;
  while (true) {
    DEBUG(dbgs() << "Beginning CP iteration #" << NoCPIters << '\n');
    bool CPChange = false;
    for (unsigned i = 0, e = CPUsers.size(); i != e; ++i)
      CPChange |= handleConstantPoolUser(i);
    if (CPChange && ++NoCPIters > 30)
      report_fatal_error("Constant Island pass failed to converge!");
    DEBUG(dumpBBs());

    // Clear NewWaterList now.  If we split a block for branches, it should
    // appear as "new water" for the next iteration of constant pool placement.
    NewWaterList.clear();

    DEBUG(dbgs() << "Beginning BR iteration #" << NoBRIters << '\n');
    bool BRChange = false;
    for (unsigned i = 0, e = ImmBranches.size(); i != e; ++i)
      BRChange |= fixupImmediateBr(ImmBranches[i]);
    if (BRChange && ++NoBRIters > 30)
      report_fatal_error("Branch Fix Up pass failed to converge!");
    DEBUG(dumpBBs());

    if (!CPChange && !BRChange)
      break;
    MadeChange = true;
  }

  // After a while, this might be made debug-only, but it is not expensive.
  verify();

  DEBUG(dbgs() << '\n'; dumpBBs());

  BBInfo.clear();
  WaterList.clear();
  CPUsers.clear();
  CPEntries.clear();
  ImmBranches.clear();

  return MadeChange;
}

/// Perform the initial placement of the constant pool entries.  To start with,
/// we put them all at the end of the function.
void
AArch64ConstantIslands::doInitialPlacement(std::vector<MachineInstr*> &CPEMIs) {
  // Create the basic block to hold the CPE's.
  MachineBasicBlock *BB = MF->CreateMachineBasicBlock();
  MF->push_back(BB);

  // MachineConstantPool measures alignment in bytes. We measure in log2(bytes).
  unsigned MaxAlign = Log2_32(MCP->getConstantPoolAlignment());

  // Mark the basic block as required by the const-pool.
  // If AlignConstantIslands isn't set, use 4-byte alignment for everything.
  BB->setAlignment(AlignConstantIslands ? MaxAlign : 2);

  // The function needs to be as aligned as the basic blocks. The linker may
  // move functions around based on their alignment.
  MF->ensureAlignment(BB->getAlignment());

  // Order the entries in BB by descending alignment.  That ensures correct
  // alignment of all entries as long as BB is sufficiently aligned.  Keep
  // track of the insertion point for each alignment.  We are going to bucket
  // sort the entries as they are created.
  SmallVector<MachineBasicBlock::iterator, 8> InsPoint(MaxAlign + 1, BB->end());

  // Add all of the constants from the constant pool to the end block, use an
  // identity mapping of CPI's to CPE's.
  const std::vector<MachineConstantPoolEntry> &CPs = MCP->getConstants();

  const DataLayout &TD = *MF->getTarget().getDataLayout();
  for (unsigned i = 0, e = CPs.size(); i != e; ++i) {
    unsigned Size = TD.getTypeAllocSize(CPs[i].getType());
    assert(Size >= 4 && "Too small constant pool entry");
    unsigned Align = CPs[i].getAlignment();
    assert(isPowerOf2_32(Align) && "Invalid alignment");
    // Verify that all constant pool entries are a multiple of their alignment.
    // If not, we would have to pad them out so that instructions stay aligned.
    assert((Size % Align) == 0 && "CP Entry not multiple of 4 bytes!");

    // Insert CONSTPOOL_ENTRY before entries with a smaller alignment.
    unsigned LogAlign = Log2_32(Align);
    MachineBasicBlock::iterator InsAt = InsPoint[LogAlign];
    MachineInstr *CPEMI =
      BuildMI(*BB, InsAt, DebugLoc(), TII->get(AArch64::CONSTPOOL_ENTRY))
        .addImm(i).addConstantPoolIndex(i).addImm(Size);
    CPEMIs.push_back(CPEMI);

    // Ensure that future entries with higher alignment get inserted before
    // CPEMI. This is bucket sort with iterators.
    for (unsigned a = LogAlign + 1; a <= MaxAlign; ++a)
      if (InsPoint[a] == InsAt)
        InsPoint[a] = CPEMI;

    // Add a new CPEntry, but no corresponding CPUser yet.
    std::vector<CPEntry> CPEs;
    CPEs.push_back(CPEntry(CPEMI, i));
    CPEntries.push_back(CPEs);
    ++NumCPEs;
    DEBUG(dbgs() << "Moved CPI#" << i << " to end of function, size = "
                 << Size << ", align = " << Align <<'\n');
  }
  DEBUG(BB->dump());
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

/// Given the constpool index and CONSTPOOL_ENTRY MI, look up the corresponding
/// CPEntry.
AArch64ConstantIslands::CPEntry
*AArch64ConstantIslands::findConstPoolEntry(unsigned CPI,
                                            const MachineInstr *CPEMI) {
  std::vector<CPEntry> &CPEs = CPEntries[CPI];
  // Number of entries per constpool index should be small, just do a
  // linear search.
  for (unsigned i = 0, e = CPEs.size(); i != e; ++i) {
    if (CPEs[i].CPEMI == CPEMI)
      return &CPEs[i];
  }
  return NULL;
}

/// Returns the required alignment of the constant pool entry represented by
/// CPEMI.  Alignment is measured in log2(bytes) units.
unsigned AArch64ConstantIslands::getCPELogAlign(const MachineInstr *CPEMI) {
  assert(CPEMI && CPEMI->getOpcode() == AArch64::CONSTPOOL_ENTRY);

  // Everything is 4-byte aligned unless AlignConstantIslands is set.
  if (!AlignConstantIslands)
    return 2;

  unsigned CPI = CPEMI->getOperand(1).getIndex();
  assert(CPI < MCP->getConstants().size() && "Invalid constant pool index.");
  unsigned Align = MCP->getConstants()[CPI].getAlignment();
  assert(isPowerOf2_32(Align) && "Invalid CPE alignment");
  return Log2_32(Align);
}

/// Do the initial scan of the function, building up information about the sizes
/// of each block, the location of all the water, and finding all of the
/// constant pool users.
void AArch64ConstantIslands::
initializeFunctionInfo(const std::vector<MachineInstr*> &CPEMIs) {
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

    // If this block doesn't fall through into the next MBB, then this is
    // 'water' that a constant pool island could be placed.
    if (!BBHasFallthrough(&MBB))
      WaterList.push_back(&MBB);

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

      if (Opc == AArch64::CONSTPOOL_ENTRY)
        continue;

      // Scan the instructions for constant pool operands.
      for (unsigned op = 0, e = I->getNumOperands(); op != e; ++op)
        if (I->getOperand(op).isCPI()) {
          // We found one.  The addressing mode tells us the max displacement
          // from the PC that this instruction permits.

          // The offsets encoded in instructions here scale by the instruction
          // size (4 bytes), effectively increasing their range by 2 bits.
          unsigned Bits = 0;

          switch (Opc) {
          default:
            llvm_unreachable("Unknown addressing mode for CP reference!");

          case AArch64::LDRw_lit:
          case AArch64::LDRx_lit:
          case AArch64::LDRs_lit:
          case AArch64::LDRd_lit:
          case AArch64::LDRq_lit:
          case AArch64::LDRSWx_lit:
          case AArch64::PRFM_lit:
            Bits = 19 + 2;
          }

          // Remember that this is a user of a CP entry.
          unsigned CPI = I->getOperand(op).getIndex();
          MachineInstr *CPEMI = CPEMIs[CPI];
          CPUsers.push_back(CPUser(I, CPEMI, Bits));

          // Increment corresponding CPEntry reference count.
          CPEntry *CPE = findConstPoolEntry(CPI, CPEMI);
          assert(CPE && "Cannot find a corresponding CPEntry!");
          CPE->RefCount++;

          // Instructions can only use one CP entry, don't bother scanning the
          // rest of the operands.
          break;
        }
    }
  }
}

/// Compute the size and some alignment information for MBB.  This function
/// updates BBInfo directly.
void AArch64ConstantIslands::computeBlockSize(MachineBasicBlock *MBB) {
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
unsigned AArch64ConstantIslands::getOffsetOf(MachineInstr *MI) const {
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

/// Little predicate function to sort the WaterList by MBB ID.
static bool CompareMBBNumbers(const MachineBasicBlock *LHS,
                              const MachineBasicBlock *RHS) {
  return LHS->getNumber() < RHS->getNumber();
}

/// When a block is newly inserted into the machine function, it upsets all of
/// the block numbers.  Renumber the blocks and update the arrays that parallel
/// this numbering.
void AArch64ConstantIslands::
updateForInsertedWaterBlock(MachineBasicBlock *NewBB) {
  // Renumber the MBB's to keep them consecutive.
  NewBB->getParent()->RenumberBlocks(NewBB);

  // Insert an entry into BBInfo to align it properly with the (newly
  // renumbered) block numbers.
  BBInfo.insert(BBInfo.begin() + NewBB->getNumber(), BasicBlockInfo());

  // Next, update WaterList.  Specifically, we need to add NewMBB as having
  // available water after it.
  water_iterator IP =
    std::lower_bound(WaterList.begin(), WaterList.end(), NewBB,
                     CompareMBBNumbers);
  WaterList.insert(IP, NewBB);
}


/// Split the basic block containing MI into two blocks, which are joined by
/// an unconditional branch.  Update data structures and renumber blocks to
/// account for this change and returns the newly created block.
MachineBasicBlock *
AArch64ConstantIslands::splitBlockBeforeInstr(MachineInstr *MI) {
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
  // This is almost the same as updateForInsertedWaterBlock, except that
  // the Water goes after OrigBB, not NewBB.
  MF->RenumberBlocks(NewBB);

  // Insert an entry into BBInfo to align it properly with the (newly
  // renumbered) block numbers.
  BBInfo.insert(BBInfo.begin() + NewBB->getNumber(), BasicBlockInfo());

  // Next, update WaterList.  Specifically, we need to add OrigMBB as having
  // available water after it (but not if it's already there, which happens
  // when splitting before a conditional branch that is followed by an
  // unconditional branch - in that case we want to insert NewBB).
  water_iterator IP =
    std::lower_bound(WaterList.begin(), WaterList.end(), OrigBB,
                     CompareMBBNumbers);
  MachineBasicBlock* WaterBB = *IP;
  if (WaterBB == OrigBB)
    WaterList.insert(llvm::next(IP), NewBB);
  else
    WaterList.insert(IP, OrigBB);
  NewWaterList.insert(OrigBB);

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

/// Compute the offset of U.MI as seen by the hardware displacement computation.
unsigned AArch64ConstantIslands::getUserOffset(CPUser &U) const {
  return getOffsetOf(U.MI);
}

/// Checks whether UserOffset (the location of a constant pool reference) is
/// within OffsetBits of TrialOffset (a proposed location of a constant pool
/// entry).
bool AArch64ConstantIslands::isOffsetInRange(unsigned UserOffset,
                                             unsigned TrialOffset,
                                             unsigned OffsetBits) {
  return isIntN(OffsetBits, static_cast<int64_t>(TrialOffset) - UserOffset);
}

/// Returns true if a CPE placed after the specified Water (a basic block) will
/// be in range for the specific MI.
///
/// Compute how much the function will grow by inserting a CPE after Water.
bool AArch64ConstantIslands::isWaterInRange(unsigned UserOffset,
                                            MachineBasicBlock* Water, CPUser &U,
                                            unsigned &Growth) {
  unsigned CPELogAlign = getCPELogAlign(U.CPEMI);
  unsigned CPEOffset = BBInfo[Water->getNumber()].postOffset(CPELogAlign);
  unsigned NextBlockOffset, NextBlockAlignment;
  MachineFunction::const_iterator NextBlock = Water;
  if (++NextBlock == MF->end()) {
    NextBlockOffset = BBInfo[Water->getNumber()].postOffset();
    NextBlockAlignment = 0;
  } else {
    NextBlockOffset = BBInfo[NextBlock->getNumber()].Offset;
    NextBlockAlignment = NextBlock->getAlignment();
  }
  unsigned Size = U.CPEMI->getOperand(2).getImm();
  unsigned CPEEnd = CPEOffset + Size;

  // The CPE may be able to hide in the alignment padding before the next
  // block. It may also cause more padding to be required if it is more aligned
  // that the next block.
  if (CPEEnd > NextBlockOffset) {
    Growth = CPEEnd - NextBlockOffset;
    // Compute the padding that would go at the end of the CPE to align the next
    // block.
    Growth += OffsetToAlignment(CPEEnd, 1u << NextBlockAlignment);

    // If the CPE is to be inserted before the instruction, that will raise
    // the offset of the instruction. Also account for unknown alignment padding
    // in blocks between CPE and the user.
    if (CPEOffset < UserOffset)
      UserOffset += Growth + UnknownPadding(MF->getAlignment(), CPELogAlign);
  } else
    // CPE fits in existing padding.
    Growth = 0;

  return isOffsetInRange(UserOffset, CPEOffset, U);
}

/// Returns true if the distance between specific MI and specific ConstPool
/// entry instruction can fit in MI's displacement field.
bool AArch64ConstantIslands::isCPEntryInRange(MachineInstr *MI,
                                              unsigned UserOffset,
                                              MachineInstr *CPEMI,
                                              unsigned OffsetBits,
                                              bool DoDump) {
  unsigned CPEOffset  = getOffsetOf(CPEMI);

  if (DoDump) {
    DEBUG({
      unsigned Block = MI->getParent()->getNumber();
      const BasicBlockInfo &BBI = BBInfo[Block];
      dbgs() << "User of CPE#" << CPEMI->getOperand(0).getImm()
             << " bits available=" << OffsetBits
             << format(" insn address=%#x", UserOffset)
             << " in BB#" << Block << ": "
             << format("%#x-%x\t", BBI.Offset, BBI.postOffset()) << *MI
             << format("CPE address=%#x offset=%+d: ", CPEOffset,
                       int(CPEOffset-UserOffset));
    });
  }

  return isOffsetInRange(UserOffset, CPEOffset, OffsetBits);
}

#ifndef NDEBUG
/// Return true of the specified basic block's only predecessor unconditionally
/// branches to its only successor.
static bool BBIsJumpedOver(MachineBasicBlock *MBB) {
  if (MBB->pred_size() != 1 || MBB->succ_size() != 1)
    return false;

  MachineBasicBlock *Succ = *MBB->succ_begin();
  MachineBasicBlock *Pred = *MBB->pred_begin();
  MachineInstr *PredMI = &Pred->back();
  if (PredMI->getOpcode() == AArch64::Bimm)
    return PredMI->getOperand(0).getMBB() == Succ;
  return false;
}
#endif // NDEBUG

void AArch64ConstantIslands::adjustBBOffsetsAfter(MachineBasicBlock *BB) {
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

/// Find the constant pool entry with index CPI and instruction CPEMI, and
/// decrement its refcount.  If the refcount becomes 0 remove the entry and
/// instruction.  Returns true if we removed the entry, false if we didn't.
bool AArch64ConstantIslands::decrementCPEReferenceCount(unsigned CPI,
                                                        MachineInstr *CPEMI) {
  // Find the old entry. Eliminate it if it is no longer used.
  CPEntry *CPE = findConstPoolEntry(CPI, CPEMI);
  assert(CPE && "Unexpected!");
  if (--CPE->RefCount == 0) {
    removeDeadCPEMI(CPEMI);
    CPE->CPEMI = NULL;
    --NumCPEs;
    return true;
  }
  return false;
}

/// See if the currently referenced CPE is in range; if not, see if an in-range
/// clone of the CPE is in range, and if so, change the data structures so the
/// user references the clone.  Returns:
/// 0 = no existing entry found
/// 1 = entry found, and there were no code insertions or deletions
/// 2 = entry found, and there were code insertions or deletions
int AArch64ConstantIslands::findInRangeCPEntry(CPUser& U, unsigned UserOffset)
{
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI  = U.CPEMI;

  // Check to see if the CPE is already in-range.
  if (isCPEntryInRange(UserMI, UserOffset, CPEMI, U.getOffsetBits(), true)) {
    DEBUG(dbgs() << "In range\n");
    return 1;
  }

  // No.  Look for previously created clones of the CPE that are in range.
  unsigned CPI = CPEMI->getOperand(1).getIndex();
  std::vector<CPEntry> &CPEs = CPEntries[CPI];
  for (unsigned i = 0, e = CPEs.size(); i != e; ++i) {
    // We already tried this one
    if (CPEs[i].CPEMI == CPEMI)
      continue;
    // Removing CPEs can leave empty entries, skip
    if (CPEs[i].CPEMI == NULL)
      continue;
    if (isCPEntryInRange(UserMI, UserOffset, CPEs[i].CPEMI,
                         U.getOffsetBits())) {
      DEBUG(dbgs() << "Replacing CPE#" << CPI << " with CPE#"
                   << CPEs[i].CPI << "\n");
      // Point the CPUser node to the replacement
      U.CPEMI = CPEs[i].CPEMI;
      // Change the CPI in the instruction operand to refer to the clone.
      for (unsigned j = 0, e = UserMI->getNumOperands(); j != e; ++j)
        if (UserMI->getOperand(j).isCPI()) {
          UserMI->getOperand(j).setIndex(CPEs[i].CPI);
          break;
        }
      // Adjust the refcount of the clone...
      CPEs[i].RefCount++;
      // ...and the original.  If we didn't remove the old entry, none of the
      // addresses changed, so we don't need another pass.
      return decrementCPEReferenceCount(CPI, CPEMI) ? 2 : 1;
    }
  }
  return 0;
}

/// Look for an existing entry in the WaterList in which we can place the CPE
/// referenced from U so it's within range of U's MI.  Returns true if found,
/// false if not.  If it returns true, WaterIter is set to the WaterList
/// entry. To ensure that this pass terminates, the CPE location for a
/// particular CPUser is only allowed to move to a lower address, so search
/// backward from the end of the list and prefer the first water that is in
/// range.
bool AArch64ConstantIslands::findAvailableWater(CPUser &U, unsigned UserOffset,
                                                water_iterator &WaterIter) {
  if (WaterList.empty())
    return false;

  unsigned BestGrowth = ~0u;
  for (water_iterator IP = prior(WaterList.end()), B = WaterList.begin();;
       --IP) {
    MachineBasicBlock* WaterBB = *IP;
    // Check if water is in range and is either at a lower address than the
    // current "high water mark" or a new water block that was created since
    // the previous iteration by inserting an unconditional branch.  In the
    // latter case, we want to allow resetting the high water mark back to
    // this new water since we haven't seen it before.  Inserting branches
    // should be relatively uncommon and when it does happen, we want to be
    // sure to take advantage of it for all the CPEs near that block, so that
    // we don't insert more branches than necessary.
    unsigned Growth;
    if (isWaterInRange(UserOffset, WaterBB, U, Growth) &&
        (WaterBB->getNumber() < U.HighWaterMark->getNumber() ||
         NewWaterList.count(WaterBB)) && Growth < BestGrowth) {
      // This is the least amount of required padding seen so far.
      BestGrowth = Growth;
      WaterIter = IP;
      DEBUG(dbgs() << "Found water after BB#" << WaterBB->getNumber()
                   << " Growth=" << Growth << '\n');

      // Keep looking unless it is perfect.
      if (BestGrowth == 0)
        return true;
    }
    if (IP == B)
      break;
  }
  return BestGrowth != ~0u;
}

/// No existing WaterList entry will work for CPUsers[CPUserIndex], so create a
/// place to put the CPE.  The end of the block is used if in range, and the
/// conditional branch munged so control flow is correct.  Otherwise the block
/// is split to create a hole with an unconditional branch around it.  In either
/// case NewMBB is set to a block following which the new island can be inserted
/// (the WaterList is not adjusted).
void AArch64ConstantIslands::createNewWater(unsigned CPUserIndex,
                                            unsigned UserOffset,
                                            MachineBasicBlock *&NewMBB) {
  CPUser &U = CPUsers[CPUserIndex];
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI  = U.CPEMI;
  unsigned CPELogAlign = getCPELogAlign(CPEMI);
  MachineBasicBlock *UserMBB = UserMI->getParent();
  const BasicBlockInfo &UserBBI = BBInfo[UserMBB->getNumber()];

  // If the block does not end in an unconditional branch already, and if the
  // end of the block is within range, make new water there.
  if (BBHasFallthrough(UserMBB)) {
    // Size of branch to insert.
    unsigned InstrSize = 4;
    // Compute the offset where the CPE will begin.
    unsigned CPEOffset = UserBBI.postOffset(CPELogAlign) + InstrSize;

    if (isOffsetInRange(UserOffset, CPEOffset, U)) {
      DEBUG(dbgs() << "Split at end of BB#" << UserMBB->getNumber()
            << format(", expected CPE offset %#x\n", CPEOffset));
      NewMBB = llvm::next(MachineFunction::iterator(UserMBB));
      // Add an unconditional branch from UserMBB to fallthrough block.  Record
      // it for branch lengthening; this new branch will not get out of range,
      // but if the preceding conditional branch is out of range, the targets
      // will be exchanged, and the altered branch may be out of range, so the
      // machinery has to know about it.
      BuildMI(UserMBB, DebugLoc(), TII->get(AArch64::Bimm)).addMBB(NewMBB);

      // 26 bits written down, specifying a multiple of 4.
      unsigned OffsetBits = 26 + 2;
      ImmBranches.push_back(ImmBranch(&UserMBB->back(), OffsetBits, false));
      BBInfo[UserMBB->getNumber()].Size += InstrSize;
      adjustBBOffsetsAfter(UserMBB);
      return;
    }
  }

  // What a big block.  Find a place within the block to split it.  We make a
  // first guess, then walk through the instructions between the one currently
  // being looked at and the possible insertion point, and make sure any other
  // instructions that reference CPEs will be able to use the same island area;
  // if not, we back up the insertion point.

  // Try to split the block so it's fully aligned.  Compute the latest split
  // point where we can add a 4-byte branch instruction, and then align to
  // LogAlign which is the largest possible alignment in the function.
  unsigned LogAlign = MF->getAlignment();
  assert(LogAlign >= CPELogAlign && "Over-aligned constant pool entry");
  unsigned KnownBits = UserBBI.internalKnownBits();
  unsigned UPad = UnknownPadding(LogAlign, KnownBits);
  unsigned BaseInsertOffset = UserOffset + U.getMaxPosDisp() - UPad;
  DEBUG(dbgs() << format("Split in middle of big block before %#x",
                         BaseInsertOffset));

  // The 4 in the following is for the unconditional branch we'll be inserting
  // Alignment of the island is handled inside isOffsetInRange.
  BaseInsertOffset -= 4;

  DEBUG(dbgs() << format(", adjusted to %#x", BaseInsertOffset)
               << " la=" << LogAlign
               << " kb=" << KnownBits
               << " up=" << UPad << '\n');

  // This could point off the end of the block if we've already got constant
  // pool entries following this block; only the last one is in the water list.
  // Back past any possible branches (allow for a conditional and a maximally
  // long unconditional).
  if (BaseInsertOffset + 8 >= UserBBI.postOffset()) {
    BaseInsertOffset = UserBBI.postOffset() - UPad - 8;
    DEBUG(dbgs() << format("Move inside block: %#x\n", BaseInsertOffset));
  }
  unsigned EndInsertOffset = BaseInsertOffset + 4 + UPad +
    CPEMI->getOperand(2).getImm();
  MachineBasicBlock::iterator MI = UserMI;
  ++MI;
  unsigned CPUIndex = CPUserIndex+1;
  unsigned NumCPUsers = CPUsers.size();
  for (unsigned Offset = UserOffset+TII->getInstSizeInBytes(*UserMI);
       Offset < BaseInsertOffset;
       Offset += TII->getInstSizeInBytes(*MI),
       MI = llvm::next(MI)) {
    assert(MI != UserMBB->end() && "Fell off end of block");
    if (CPUIndex < NumCPUsers && CPUsers[CPUIndex].MI == MI) {
      CPUser &U = CPUsers[CPUIndex];
      if (!isOffsetInRange(Offset, EndInsertOffset, U)) {
        // Shift intertion point by one unit of alignment so it is within reach.
        BaseInsertOffset -= 1u << LogAlign;
        EndInsertOffset  -= 1u << LogAlign;
      }
      // This is overly conservative, as we don't account for CPEMIs being
      // reused within the block, but it doesn't matter much.  Also assume CPEs
      // are added in order with alignment padding.  We may eventually be able
      // to pack the aligned CPEs better.
      EndInsertOffset += U.CPEMI->getOperand(2).getImm();
      CPUIndex++;
    }
  }

  --MI;
  NewMBB = splitBlockBeforeInstr(MI);
}

/// Analyze the specified user, checking to see if it is out-of-range.  If so,
/// pick up the constant pool value and move it some place in-range.  Return
/// true if we changed any addresses, false otherwise.
bool AArch64ConstantIslands::handleConstantPoolUser(unsigned CPUserIndex) {
  CPUser &U = CPUsers[CPUserIndex];
  MachineInstr *UserMI = U.MI;
  MachineInstr *CPEMI  = U.CPEMI;
  unsigned CPI = CPEMI->getOperand(1).getIndex();
  unsigned Size = CPEMI->getOperand(2).getImm();
  // Compute this only once, it's expensive.
  unsigned UserOffset = getUserOffset(U);

  // See if the current entry is within range, or there is a clone of it
  // in range.
  int result = findInRangeCPEntry(U, UserOffset);
  if (result==1) return false;
  else if (result==2) return true;

  // No existing clone of this CPE is within range.
  // We will be generating a new clone.  Get a UID for it.
  unsigned ID = AFI->createPICLabelUId();

  // Look for water where we can place this CPE.
  MachineBasicBlock *NewIsland = MF->CreateMachineBasicBlock();
  MachineBasicBlock *NewMBB;
  water_iterator IP;
  if (findAvailableWater(U, UserOffset, IP)) {
    DEBUG(dbgs() << "Found water in range\n");
    MachineBasicBlock *WaterBB = *IP;

    // If the original WaterList entry was "new water" on this iteration,
    // propagate that to the new island.  This is just keeping NewWaterList
    // updated to match the WaterList, which will be updated below.
    if (NewWaterList.count(WaterBB)) {
      NewWaterList.erase(WaterBB);
      NewWaterList.insert(NewIsland);
    }
    // The new CPE goes before the following block (NewMBB).
    NewMBB = llvm::next(MachineFunction::iterator(WaterBB));

  } else {
    // No water found.
    DEBUG(dbgs() << "No water found\n");
    createNewWater(CPUserIndex, UserOffset, NewMBB);

    // splitBlockBeforeInstr adds to WaterList, which is important when it is
    // called while handling branches so that the water will be seen on the
    // next iteration for constant pools, but in this context, we don't want
    // it.  Check for this so it will be removed from the WaterList.
    // Also remove any entry from NewWaterList.
    MachineBasicBlock *WaterBB = prior(MachineFunction::iterator(NewMBB));
    IP = std::find(WaterList.begin(), WaterList.end(), WaterBB);
    if (IP != WaterList.end())
      NewWaterList.erase(WaterBB);

    // We are adding new water.  Update NewWaterList.
    NewWaterList.insert(NewIsland);
  }

  // Remove the original WaterList entry; we want subsequent insertions in
  // this vicinity to go after the one we're about to insert.  This
  // considerably reduces the number of times we have to move the same CPE
  // more than once and is also important to ensure the algorithm terminates.
  if (IP != WaterList.end())
    WaterList.erase(IP);

  // Okay, we know we can put an island before NewMBB now, do it!
  MF->insert(NewMBB, NewIsland);

  // Update internal data structures to account for the newly inserted MBB.
  updateForInsertedWaterBlock(NewIsland);

  // Decrement the old entry, and remove it if refcount becomes 0.
  decrementCPEReferenceCount(CPI, CPEMI);

  // Now that we have an island to add the CPE to, clone the original CPE and
  // add it to the island.
  U.HighWaterMark = NewIsland;
  U.CPEMI = BuildMI(NewIsland, DebugLoc(), TII->get(AArch64::CONSTPOOL_ENTRY))
                .addImm(ID).addConstantPoolIndex(CPI).addImm(Size);
  CPEntries[CPI].push_back(CPEntry(U.CPEMI, ID, 1));
  ++NumCPEs;

  // Mark the basic block as aligned as required by the const-pool entry.
  NewIsland->setAlignment(getCPELogAlign(U.CPEMI));

  // Increase the size of the island block to account for the new entry.
  BBInfo[NewIsland->getNumber()].Size += Size;
  adjustBBOffsetsAfter(llvm::prior(MachineFunction::iterator(NewIsland)));

  // Finally, change the CPI in the instruction operand to be ID.
  for (unsigned i = 0, e = UserMI->getNumOperands(); i != e; ++i)
    if (UserMI->getOperand(i).isCPI()) {
      UserMI->getOperand(i).setIndex(ID);
      break;
    }

  DEBUG(dbgs() << "  Moved CPE to #" << ID << " CPI=" << CPI
        << format(" offset=%#x\n", BBInfo[NewIsland->getNumber()].Offset));

  return true;
}

/// Remove a dead constant pool entry instruction. Update sizes and offsets of
/// impacted basic blocks.
void AArch64ConstantIslands::removeDeadCPEMI(MachineInstr *CPEMI) {
  MachineBasicBlock *CPEBB = CPEMI->getParent();
  unsigned Size = CPEMI->getOperand(2).getImm();
  CPEMI->eraseFromParent();
  BBInfo[CPEBB->getNumber()].Size -= Size;
  // All succeeding offsets have the current size value added in, fix this.
  if (CPEBB->empty()) {
    BBInfo[CPEBB->getNumber()].Size = 0;

    // This block no longer needs to be aligned. <rdar://problem/10534709>.
    CPEBB->setAlignment(0);
  } else
    // Entries are sorted by descending alignment, so realign from the front.
    CPEBB->setAlignment(getCPELogAlign(CPEBB->begin()));

  adjustBBOffsetsAfter(CPEBB);
  // An island has only one predecessor BB and one successor BB. Check if
  // this BB's predecessor jumps directly to this BB's successor. This
  // shouldn't happen currently.
  assert(!BBIsJumpedOver(CPEBB) && "How did this happen?");
  // FIXME: remove the empty blocks after all the work is done?
}

/// Remove constant pool entries whose refcounts are zero.
bool AArch64ConstantIslands::removeUnusedCPEntries() {
  unsigned MadeChange = false;
  for (unsigned i = 0, e = CPEntries.size(); i != e; ++i) {
      std::vector<CPEntry> &CPEs = CPEntries[i];
      for (unsigned j = 0, ee = CPEs.size(); j != ee; ++j) {
        if (CPEs[j].RefCount == 0 && CPEs[j].CPEMI) {
          removeDeadCPEMI(CPEs[j].CPEMI);
          CPEs[j].CPEMI = NULL;
          MadeChange = true;
        }
      }
  }
  return MadeChange;
}

/// Returns true if the distance between specific MI and specific BB can fit in
/// MI's displacement field.
bool AArch64ConstantIslands::isBBInRange(MachineInstr *MI,
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
bool AArch64ConstantIslands::fixupImmediateBr(ImmBranch &Br) {
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
AArch64ConstantIslands::fixupConditionalBr(ImmBranch &Br) {
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
