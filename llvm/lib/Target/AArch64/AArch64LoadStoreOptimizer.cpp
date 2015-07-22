//=- AArch64LoadStoreOptimizer.cpp - AArch64 load/store opt. pass -*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a pass that performs load / store related peephole
// optimizations. This pass should be run after register allocation.
//
//===----------------------------------------------------------------------===//

#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"
#include "MCTargetDesc/AArch64AddressingModes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "aarch64-ldst-opt"

/// AArch64AllocLoadStoreOpt - Post-register allocation pass to combine
/// load / store instructions to form ldp / stp instructions.

STATISTIC(NumPairCreated, "Number of load/store pair instructions generated");
STATISTIC(NumPostFolded, "Number of post-index updates folded");
STATISTIC(NumPreFolded, "Number of pre-index updates folded");
STATISTIC(NumUnscaledPairCreated,
          "Number of load/store from unscaled generated");

static cl::opt<unsigned> ScanLimit("aarch64-load-store-scan-limit",
                                   cl::init(20), cl::Hidden);

// Place holder while testing unscaled load/store combining
static cl::opt<bool> EnableAArch64UnscaledMemOp(
    "aarch64-unscaled-mem-op", cl::Hidden,
    cl::desc("Allow AArch64 unscaled load/store combining"), cl::init(true));

namespace {

typedef struct LdStPairFlags {
  // If a matching instruction is found, MergeForward is set to true if the
  // merge is to remove the first instruction and replace the second with
  // a pair-wise insn, and false if the reverse is true.
  bool MergeForward;

  // SExtIdx gives the index of the result of the load pair that must be
  // extended. The value of SExtIdx assumes that the paired load produces the
  // value in this order: (I, returned iterator), i.e., -1 means no value has
  // to be extended, 0 means I, and 1 means the returned iterator.
  int SExtIdx;

  LdStPairFlags() : MergeForward(false), SExtIdx(-1) {}

  void setMergeForward(bool V = true) { MergeForward = V; }
  bool getMergeForward() const { return MergeForward; }

  void setSExtIdx(int V) { SExtIdx = V; }
  int getSExtIdx() const { return SExtIdx; }

} LdStPairFlags;

struct AArch64LoadStoreOpt : public MachineFunctionPass {
  static char ID;
  AArch64LoadStoreOpt() : MachineFunctionPass(ID) {}

  const AArch64InstrInfo *TII;
  const TargetRegisterInfo *TRI;

  // Scan the instructions looking for a load/store that can be combined
  // with the current instruction into a load/store pair.
  // Return the matching instruction if one is found, else MBB->end().
  MachineBasicBlock::iterator findMatchingInsn(MachineBasicBlock::iterator I,
                                               LdStPairFlags &Flags,
                                               unsigned Limit);
  // Merge the two instructions indicated into a single pair-wise instruction.
  // If MergeForward is true, erase the first instruction and fold its
  // operation into the second. If false, the reverse. Return the instruction
  // following the first instruction (which may change during processing).
  MachineBasicBlock::iterator
  mergePairedInsns(MachineBasicBlock::iterator I,
                   MachineBasicBlock::iterator Paired,
                   const LdStPairFlags &Flags);

  // Scan the instruction list to find a base register update that can
  // be combined with the current instruction (a load or store) using
  // pre or post indexed addressing with writeback. Scan forwards.
  MachineBasicBlock::iterator
  findMatchingUpdateInsnForward(MachineBasicBlock::iterator I, unsigned Limit,
                                int Value);

  // Scan the instruction list to find a base register update that can
  // be combined with the current instruction (a load or store) using
  // pre or post indexed addressing with writeback. Scan backwards.
  MachineBasicBlock::iterator
  findMatchingUpdateInsnBackward(MachineBasicBlock::iterator I, unsigned Limit);

  // Merge a pre-index base register update into a ld/st instruction.
  MachineBasicBlock::iterator
  mergePreIdxUpdateInsn(MachineBasicBlock::iterator I,
                        MachineBasicBlock::iterator Update);

  // Merge a post-index base register update into a ld/st instruction.
  MachineBasicBlock::iterator
  mergePostIdxUpdateInsn(MachineBasicBlock::iterator I,
                         MachineBasicBlock::iterator Update);

  bool optimizeBlock(MachineBasicBlock &MBB);

  bool runOnMachineFunction(MachineFunction &Fn) override;

  const char *getPassName() const override {
    return "AArch64 load / store optimization pass";
  }

private:
  int getMemSize(MachineInstr *MemMI);
};
char AArch64LoadStoreOpt::ID = 0;
} // namespace

static bool isUnscaledLdst(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case AArch64::STURSi:
  case AArch64::STURDi:
  case AArch64::STURQi:
  case AArch64::STURWi:
  case AArch64::STURXi:
  case AArch64::LDURSi:
  case AArch64::LDURDi:
  case AArch64::LDURQi:
  case AArch64::LDURWi:
  case AArch64::LDURXi:
  case AArch64::LDURSWi:
    return true;
  }
}

// Size in bytes of the data moved by an unscaled load or store
int AArch64LoadStoreOpt::getMemSize(MachineInstr *MemMI) {
  switch (MemMI->getOpcode()) {
  default:
    llvm_unreachable("Opcode has unknown size!");
  case AArch64::STRSui:
  case AArch64::STURSi:
    return 4;
  case AArch64::STRDui:
  case AArch64::STURDi:
    return 8;
  case AArch64::STRQui:
  case AArch64::STURQi:
    return 16;
  case AArch64::STRWui:
  case AArch64::STURWi:
    return 4;
  case AArch64::STRXui:
  case AArch64::STURXi:
    return 8;
  case AArch64::LDRSui:
  case AArch64::LDURSi:
    return 4;
  case AArch64::LDRDui:
  case AArch64::LDURDi:
    return 8;
  case AArch64::LDRQui:
  case AArch64::LDURQi:
    return 16;
  case AArch64::LDRWui:
  case AArch64::LDURWi:
    return 4;
  case AArch64::LDRXui:
  case AArch64::LDURXi:
    return 8;
  case AArch64::LDRSWui:
  case AArch64::LDURSWi:
    return 4;
  }
}

static unsigned getMatchingNonSExtOpcode(unsigned Opc,
                                         bool *IsValidLdStrOpc = nullptr) {
  if (IsValidLdStrOpc)
    *IsValidLdStrOpc = true;
  switch (Opc) {
  default:
    if (IsValidLdStrOpc)
      *IsValidLdStrOpc = false;
    return UINT_MAX;
  case AArch64::STRDui:
  case AArch64::STURDi:
  case AArch64::STRQui:
  case AArch64::STURQi:
  case AArch64::STRWui:
  case AArch64::STURWi:
  case AArch64::STRXui:
  case AArch64::STURXi:
  case AArch64::LDRDui:
  case AArch64::LDURDi:
  case AArch64::LDRQui:
  case AArch64::LDURQi:
  case AArch64::LDRWui:
  case AArch64::LDURWi:
  case AArch64::LDRXui:
  case AArch64::LDURXi:
  case AArch64::STRSui:
  case AArch64::STURSi:
  case AArch64::LDRSui:
  case AArch64::LDURSi:
    return Opc;
  case AArch64::LDRSWui:
    return AArch64::LDRWui;
  case AArch64::LDURSWi:
    return AArch64::LDURWi;
  }
}

static unsigned getMatchingPairOpcode(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("Opcode has no pairwise equivalent!");
  case AArch64::STRSui:
  case AArch64::STURSi:
    return AArch64::STPSi;
  case AArch64::STRDui:
  case AArch64::STURDi:
    return AArch64::STPDi;
  case AArch64::STRQui:
  case AArch64::STURQi:
    return AArch64::STPQi;
  case AArch64::STRWui:
  case AArch64::STURWi:
    return AArch64::STPWi;
  case AArch64::STRXui:
  case AArch64::STURXi:
    return AArch64::STPXi;
  case AArch64::LDRSui:
  case AArch64::LDURSi:
    return AArch64::LDPSi;
  case AArch64::LDRDui:
  case AArch64::LDURDi:
    return AArch64::LDPDi;
  case AArch64::LDRQui:
  case AArch64::LDURQi:
    return AArch64::LDPQi;
  case AArch64::LDRWui:
  case AArch64::LDURWi:
    return AArch64::LDPWi;
  case AArch64::LDRXui:
  case AArch64::LDURXi:
    return AArch64::LDPXi;
  case AArch64::LDRSWui:
  case AArch64::LDURSWi:
    return AArch64::LDPSWi;
  }
}

static unsigned getPreIndexedOpcode(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("Opcode has no pre-indexed equivalent!");
  case AArch64::STRSui:
    return AArch64::STRSpre;
  case AArch64::STRDui:
    return AArch64::STRDpre;
  case AArch64::STRQui:
    return AArch64::STRQpre;
  case AArch64::STRWui:
    return AArch64::STRWpre;
  case AArch64::STRXui:
    return AArch64::STRXpre;
  case AArch64::LDRSui:
    return AArch64::LDRSpre;
  case AArch64::LDRDui:
    return AArch64::LDRDpre;
  case AArch64::LDRQui:
    return AArch64::LDRQpre;
  case AArch64::LDRWui:
    return AArch64::LDRWpre;
  case AArch64::LDRXui:
    return AArch64::LDRXpre;
  case AArch64::LDRSWui:
    return AArch64::LDRSWpre;
  }
}

static unsigned getPostIndexedOpcode(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("Opcode has no post-indexed wise equivalent!");
  case AArch64::STRSui:
    return AArch64::STRSpost;
  case AArch64::STRDui:
    return AArch64::STRDpost;
  case AArch64::STRQui:
    return AArch64::STRQpost;
  case AArch64::STRWui:
    return AArch64::STRWpost;
  case AArch64::STRXui:
    return AArch64::STRXpost;
  case AArch64::LDRSui:
    return AArch64::LDRSpost;
  case AArch64::LDRDui:
    return AArch64::LDRDpost;
  case AArch64::LDRQui:
    return AArch64::LDRQpost;
  case AArch64::LDRWui:
    return AArch64::LDRWpost;
  case AArch64::LDRXui:
    return AArch64::LDRXpost;
  case AArch64::LDRSWui:
    return AArch64::LDRSWpost;
  }
}

MachineBasicBlock::iterator
AArch64LoadStoreOpt::mergePairedInsns(MachineBasicBlock::iterator I,
                                      MachineBasicBlock::iterator Paired,
                                      const LdStPairFlags &Flags) {
  MachineBasicBlock::iterator NextI = I;
  ++NextI;
  // If NextI is the second of the two instructions to be merged, we need
  // to skip one further. Either way we merge will invalidate the iterator,
  // and we don't need to scan the new instruction, as it's a pairwise
  // instruction, which we're not considering for further action anyway.
  if (NextI == Paired)
    ++NextI;

  int SExtIdx = Flags.getSExtIdx();
  unsigned Opc =
      SExtIdx == -1 ? I->getOpcode() : getMatchingNonSExtOpcode(I->getOpcode());
  bool IsUnscaled = isUnscaledLdst(Opc);
  int OffsetStride =
      IsUnscaled && EnableAArch64UnscaledMemOp ? getMemSize(I) : 1;

  bool MergeForward = Flags.getMergeForward();
  unsigned NewOpc = getMatchingPairOpcode(Opc);
  // Insert our new paired instruction after whichever of the paired
  // instructions MergeForward indicates.
  MachineBasicBlock::iterator InsertionPoint = MergeForward ? Paired : I;
  // Also based on MergeForward is from where we copy the base register operand
  // so we get the flags compatible with the input code.
  MachineOperand &BaseRegOp =
      MergeForward ? Paired->getOperand(1) : I->getOperand(1);

  // Which register is Rt and which is Rt2 depends on the offset order.
  MachineInstr *RtMI, *Rt2MI;
  if (I->getOperand(2).getImm() ==
      Paired->getOperand(2).getImm() + OffsetStride) {
    RtMI = Paired;
    Rt2MI = I;
    // Here we swapped the assumption made for SExtIdx.
    // I.e., we turn ldp I, Paired into ldp Paired, I.
    // Update the index accordingly.
    if (SExtIdx != -1)
      SExtIdx = (SExtIdx + 1) % 2;
  } else {
    RtMI = I;
    Rt2MI = Paired;
  }
  // Handle Unscaled
  int OffsetImm = RtMI->getOperand(2).getImm();
  if (IsUnscaled && EnableAArch64UnscaledMemOp)
    OffsetImm /= OffsetStride;

  // Construct the new instruction.
  MachineInstrBuilder MIB = BuildMI(*I->getParent(), InsertionPoint,
                                    I->getDebugLoc(), TII->get(NewOpc))
                                .addOperand(RtMI->getOperand(0))
                                .addOperand(Rt2MI->getOperand(0))
                                .addOperand(BaseRegOp)
                                .addImm(OffsetImm);
  (void)MIB;

  // FIXME: Do we need/want to copy the mem operands from the source
  //        instructions? Probably. What uses them after this?

  DEBUG(dbgs() << "Creating pair load/store. Replacing instructions:\n    ");
  DEBUG(I->print(dbgs()));
  DEBUG(dbgs() << "    ");
  DEBUG(Paired->print(dbgs()));
  DEBUG(dbgs() << "  with instruction:\n    ");

  if (SExtIdx != -1) {
    // Generate the sign extension for the proper result of the ldp.
    // I.e., with X1, that would be:
    // %W1<def> = KILL %W1, %X1<imp-def>
    // %X1<def> = SBFMXri %X1<kill>, 0, 31
    MachineOperand &DstMO = MIB->getOperand(SExtIdx);
    // Right now, DstMO has the extended register, since it comes from an
    // extended opcode.
    unsigned DstRegX = DstMO.getReg();
    // Get the W variant of that register.
    unsigned DstRegW = TRI->getSubReg(DstRegX, AArch64::sub_32);
    // Update the result of LDP to use the W instead of the X variant.
    DstMO.setReg(DstRegW);
    DEBUG(((MachineInstr *)MIB)->print(dbgs()));
    DEBUG(dbgs() << "\n");
    // Make the machine verifier happy by providing a definition for
    // the X register.
    // Insert this definition right after the generated LDP, i.e., before
    // InsertionPoint.
    MachineInstrBuilder MIBKill =
        BuildMI(*I->getParent(), InsertionPoint, I->getDebugLoc(),
                TII->get(TargetOpcode::KILL), DstRegW)
            .addReg(DstRegW)
            .addReg(DstRegX, RegState::Define);
    MIBKill->getOperand(2).setImplicit();
    // Create the sign extension.
    MachineInstrBuilder MIBSXTW =
        BuildMI(*I->getParent(), InsertionPoint, I->getDebugLoc(),
                TII->get(AArch64::SBFMXri), DstRegX)
            .addReg(DstRegX)
            .addImm(0)
            .addImm(31);
    (void)MIBSXTW;
    DEBUG(dbgs() << "  Extend operand:\n    ");
    DEBUG(((MachineInstr *)MIBSXTW)->print(dbgs()));
    DEBUG(dbgs() << "\n");
  } else {
    DEBUG(((MachineInstr *)MIB)->print(dbgs()));
    DEBUG(dbgs() << "\n");
  }

  // Erase the old instructions.
  I->eraseFromParent();
  Paired->eraseFromParent();

  return NextI;
}

/// trackRegDefsUses - Remember what registers the specified instruction uses
/// and modifies.
static void trackRegDefsUses(MachineInstr *MI, BitVector &ModifiedRegs,
                             BitVector &UsedRegs,
                             const TargetRegisterInfo *TRI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (MO.isRegMask())
      ModifiedRegs.setBitsNotInMask(MO.getRegMask());

    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (MO.isDef()) {
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        ModifiedRegs.set(*AI);
    } else {
      assert(MO.isUse() && "Reg operand not a def and not a use?!?");
      for (MCRegAliasIterator AI(Reg, TRI, true); AI.isValid(); ++AI)
        UsedRegs.set(*AI);
    }
  }
}

static bool inBoundsForPair(bool IsUnscaled, int Offset, int OffsetStride) {
  if (!IsUnscaled && (Offset > 63 || Offset < -64))
    return false;
  if (IsUnscaled) {
    // Convert the byte-offset used by unscaled into an "element" offset used
    // by the scaled pair load/store instructions.
    int ElemOffset = Offset / OffsetStride;
    if (ElemOffset > 63 || ElemOffset < -64)
      return false;
  }
  return true;
}

// Do alignment, specialized to power of 2 and for signed ints,
// avoiding having to do a C-style cast from uint_64t to int when
// using RoundUpToAlignment from include/llvm/Support/MathExtras.h.
// FIXME: Move this function to include/MathExtras.h?
static int alignTo(int Num, int PowOf2) {
  return (Num + PowOf2 - 1) & ~(PowOf2 - 1);
}

static bool mayAlias(MachineInstr *MIa, MachineInstr *MIb,
                     const AArch64InstrInfo *TII) {
  // One of the instructions must modify memory.
  if (!MIa->mayStore() && !MIb->mayStore())
    return false;

  // Both instructions must be memory operations.
  if (!MIa->mayLoadOrStore() && !MIb->mayLoadOrStore())
    return false;

  return !TII->areMemAccessesTriviallyDisjoint(MIa, MIb);
}

static bool mayAlias(MachineInstr *MIa,
                     SmallVectorImpl<MachineInstr *> &MemInsns,
                     const AArch64InstrInfo *TII) {
  for (auto &MIb : MemInsns)
    if (mayAlias(MIa, MIb, TII))
      return true;

  return false;
}

/// findMatchingInsn - Scan the instructions looking for a load/store that can
/// be combined with the current instruction into a load/store pair.
MachineBasicBlock::iterator
AArch64LoadStoreOpt::findMatchingInsn(MachineBasicBlock::iterator I,
                                      LdStPairFlags &Flags,
                                      unsigned Limit) {
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineBasicBlock::iterator MBBI = I;
  MachineInstr *FirstMI = I;
  ++MBBI;

  unsigned Opc = FirstMI->getOpcode();
  bool MayLoad = FirstMI->mayLoad();
  bool IsUnscaled = isUnscaledLdst(Opc);
  unsigned Reg = FirstMI->getOperand(0).getReg();
  unsigned BaseReg = FirstMI->getOperand(1).getReg();
  int Offset = FirstMI->getOperand(2).getImm();

  // Early exit if the first instruction modifies the base register.
  // e.g., ldr x0, [x0]
  // Early exit if the offset if not possible to match. (6 bits of positive
  // range, plus allow an extra one in case we find a later insn that matches
  // with Offset-1
  if (FirstMI->modifiesRegister(BaseReg, TRI))
    return E;
  int OffsetStride =
      IsUnscaled && EnableAArch64UnscaledMemOp ? getMemSize(FirstMI) : 1;
  if (!inBoundsForPair(IsUnscaled, Offset, OffsetStride))
    return E;

  // Track which registers have been modified and used between the first insn
  // (inclusive) and the second insn.
  BitVector ModifiedRegs, UsedRegs;
  ModifiedRegs.resize(TRI->getNumRegs());
  UsedRegs.resize(TRI->getNumRegs());

  // Remember any instructions that read/write memory between FirstMI and MI.
  SmallVector<MachineInstr *, 4> MemInsns;

  for (unsigned Count = 0; MBBI != E && Count < Limit; ++MBBI) {
    MachineInstr *MI = MBBI;
    // Skip DBG_VALUE instructions. Otherwise debug info can affect the
    // optimization by changing how far we scan.
    if (MI->isDebugValue())
      continue;

    // Now that we know this is a real instruction, count it.
    ++Count;

    bool CanMergeOpc = Opc == MI->getOpcode();
    Flags.setSExtIdx(-1);
    if (!CanMergeOpc) {
      bool IsValidLdStrOpc;
      unsigned NonSExtOpc = getMatchingNonSExtOpcode(Opc, &IsValidLdStrOpc);
      if (!IsValidLdStrOpc)
        continue;
      // Opc will be the first instruction in the pair.
      Flags.setSExtIdx(NonSExtOpc == (unsigned)Opc ? 1 : 0);
      CanMergeOpc = NonSExtOpc == getMatchingNonSExtOpcode(MI->getOpcode());
    }

    if (CanMergeOpc && MI->getOperand(2).isImm()) {
      // If we've found another instruction with the same opcode, check to see
      // if the base and offset are compatible with our starting instruction.
      // These instructions all have scaled immediate operands, so we just
      // check for +1/-1. Make sure to check the new instruction offset is
      // actually an immediate and not a symbolic reference destined for
      // a relocation.
      //
      // Pairwise instructions have a 7-bit signed offset field. Single insns
      // have a 12-bit unsigned offset field. To be a valid combine, the
      // final offset must be in range.
      unsigned MIBaseReg = MI->getOperand(1).getReg();
      int MIOffset = MI->getOperand(2).getImm();
      if (BaseReg == MIBaseReg && ((Offset == MIOffset + OffsetStride) ||
                                   (Offset + OffsetStride == MIOffset))) {
        int MinOffset = Offset < MIOffset ? Offset : MIOffset;
        // If this is a volatile load/store that otherwise matched, stop looking
        // as something is going on that we don't have enough information to
        // safely transform. Similarly, stop if we see a hint to avoid pairs.
        if (MI->hasOrderedMemoryRef() || TII->isLdStPairSuppressed(MI))
          return E;
        // If the resultant immediate offset of merging these instructions
        // is out of range for a pairwise instruction, bail and keep looking.
        bool MIIsUnscaled = isUnscaledLdst(MI->getOpcode());
        if (!inBoundsForPair(MIIsUnscaled, MinOffset, OffsetStride)) {
          trackRegDefsUses(MI, ModifiedRegs, UsedRegs, TRI);
          if (MI->mayLoadOrStore())
            MemInsns.push_back(MI);
          continue;
        }
        // If the alignment requirements of the paired (scaled) instruction
        // can't express the offset of the unscaled input, bail and keep
        // looking.
        if (IsUnscaled && EnableAArch64UnscaledMemOp &&
            (alignTo(MinOffset, OffsetStride) != MinOffset)) {
          trackRegDefsUses(MI, ModifiedRegs, UsedRegs, TRI);
          if (MI->mayLoadOrStore())
            MemInsns.push_back(MI);
          continue;
        }
        // If the destination register of the loads is the same register, bail
        // and keep looking. A load-pair instruction with both destination
        // registers the same is UNPREDICTABLE and will result in an exception.
        if (MayLoad && Reg == MI->getOperand(0).getReg()) {
          trackRegDefsUses(MI, ModifiedRegs, UsedRegs, TRI);
          if (MI->mayLoadOrStore())
            MemInsns.push_back(MI);
          continue;
        }

        // If the Rt of the second instruction was not modified or used between
        // the two instructions and none of the instructions between the second
        // and first alias with the second, we can combine the second into the
        // first.
        if (!ModifiedRegs[MI->getOperand(0).getReg()] &&
            !(MI->mayLoad() && UsedRegs[MI->getOperand(0).getReg()]) &&
            !mayAlias(MI, MemInsns, TII)) {
          Flags.setMergeForward(false);
          return MBBI;
        }

        // Likewise, if the Rt of the first instruction is not modified or used
        // between the two instructions and none of the instructions between the
        // first and the second alias with the first, we can combine the first
        // into the second.
        if (!ModifiedRegs[FirstMI->getOperand(0).getReg()] &&
            !(FirstMI->mayLoad() &&
              UsedRegs[FirstMI->getOperand(0).getReg()]) &&
            !mayAlias(FirstMI, MemInsns, TII)) {
          Flags.setMergeForward(true);
          return MBBI;
        }
        // Unable to combine these instructions due to interference in between.
        // Keep looking.
      }
    }

    // If the instruction wasn't a matching load or store.  Stop searching if we
    // encounter a call instruction that might modify memory.
    if (MI->isCall())
      return E;

    // Update modified / uses register lists.
    trackRegDefsUses(MI, ModifiedRegs, UsedRegs, TRI);

    // Otherwise, if the base register is modified, we have no match, so
    // return early.
    if (ModifiedRegs[BaseReg])
      return E;

    // Update list of instructions that read/write memory.
    if (MI->mayLoadOrStore())
      MemInsns.push_back(MI);
  }
  return E;
}

MachineBasicBlock::iterator
AArch64LoadStoreOpt::mergePreIdxUpdateInsn(MachineBasicBlock::iterator I,
                                           MachineBasicBlock::iterator Update) {
  assert((Update->getOpcode() == AArch64::ADDXri ||
          Update->getOpcode() == AArch64::SUBXri) &&
         "Unexpected base register update instruction to merge!");
  MachineBasicBlock::iterator NextI = I;
  // Return the instruction following the merged instruction, which is
  // the instruction following our unmerged load. Unless that's the add/sub
  // instruction we're merging, in which case it's the one after that.
  if (++NextI == Update)
    ++NextI;

  int Value = Update->getOperand(2).getImm();
  assert(AArch64_AM::getShiftValue(Update->getOperand(3).getImm()) == 0 &&
         "Can't merge 1 << 12 offset into pre-indexed load / store");
  if (Update->getOpcode() == AArch64::SUBXri)
    Value = -Value;

  unsigned NewOpc = getPreIndexedOpcode(I->getOpcode());
  MachineInstrBuilder MIB =
      BuildMI(*I->getParent(), I, I->getDebugLoc(), TII->get(NewOpc))
          .addOperand(Update->getOperand(0))
          .addOperand(I->getOperand(0))
          .addOperand(I->getOperand(1))
          .addImm(Value);
  (void)MIB;

  DEBUG(dbgs() << "Creating pre-indexed load/store.");
  DEBUG(dbgs() << "    Replacing instructions:\n    ");
  DEBUG(I->print(dbgs()));
  DEBUG(dbgs() << "    ");
  DEBUG(Update->print(dbgs()));
  DEBUG(dbgs() << "  with instruction:\n    ");
  DEBUG(((MachineInstr *)MIB)->print(dbgs()));
  DEBUG(dbgs() << "\n");

  // Erase the old instructions for the block.
  I->eraseFromParent();
  Update->eraseFromParent();

  return NextI;
}

MachineBasicBlock::iterator AArch64LoadStoreOpt::mergePostIdxUpdateInsn(
    MachineBasicBlock::iterator I, MachineBasicBlock::iterator Update) {
  assert((Update->getOpcode() == AArch64::ADDXri ||
          Update->getOpcode() == AArch64::SUBXri) &&
         "Unexpected base register update instruction to merge!");
  MachineBasicBlock::iterator NextI = I;
  // Return the instruction following the merged instruction, which is
  // the instruction following our unmerged load. Unless that's the add/sub
  // instruction we're merging, in which case it's the one after that.
  if (++NextI == Update)
    ++NextI;

  int Value = Update->getOperand(2).getImm();
  assert(AArch64_AM::getShiftValue(Update->getOperand(3).getImm()) == 0 &&
         "Can't merge 1 << 12 offset into post-indexed load / store");
  if (Update->getOpcode() == AArch64::SUBXri)
    Value = -Value;

  unsigned NewOpc = getPostIndexedOpcode(I->getOpcode());
  MachineInstrBuilder MIB =
      BuildMI(*I->getParent(), I, I->getDebugLoc(), TII->get(NewOpc))
          .addOperand(Update->getOperand(0))
          .addOperand(I->getOperand(0))
          .addOperand(I->getOperand(1))
          .addImm(Value);
  (void)MIB;

  DEBUG(dbgs() << "Creating post-indexed load/store.");
  DEBUG(dbgs() << "    Replacing instructions:\n    ");
  DEBUG(I->print(dbgs()));
  DEBUG(dbgs() << "    ");
  DEBUG(Update->print(dbgs()));
  DEBUG(dbgs() << "  with instruction:\n    ");
  DEBUG(((MachineInstr *)MIB)->print(dbgs()));
  DEBUG(dbgs() << "\n");

  // Erase the old instructions for the block.
  I->eraseFromParent();
  Update->eraseFromParent();

  return NextI;
}

static bool isMatchingUpdateInsn(MachineInstr *MI, unsigned BaseReg,
                                 int Offset) {
  switch (MI->getOpcode()) {
  default:
    break;
  case AArch64::SUBXri:
    // Negate the offset for a SUB instruction.
    Offset *= -1;
  // FALLTHROUGH
  case AArch64::ADDXri:
    // Make sure it's a vanilla immediate operand, not a relocation or
    // anything else we can't handle.
    if (!MI->getOperand(2).isImm())
      break;
    // Watch out for 1 << 12 shifted value.
    if (AArch64_AM::getShiftValue(MI->getOperand(3).getImm()))
      break;
    // If the instruction has the base register as source and dest and the
    // immediate will fit in a signed 9-bit integer, then we have a match.
    if (MI->getOperand(0).getReg() == BaseReg &&
        MI->getOperand(1).getReg() == BaseReg &&
        MI->getOperand(2).getImm() <= 255 &&
        MI->getOperand(2).getImm() >= -256) {
      // If we have a non-zero Offset, we check that it matches the amount
      // we're adding to the register.
      if (!Offset || Offset == MI->getOperand(2).getImm())
        return true;
    }
    break;
  }
  return false;
}

MachineBasicBlock::iterator AArch64LoadStoreOpt::findMatchingUpdateInsnForward(
    MachineBasicBlock::iterator I, unsigned Limit, int Value) {
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineInstr *MemMI = I;
  MachineBasicBlock::iterator MBBI = I;
  const MachineFunction &MF = *MemMI->getParent()->getParent();

  unsigned DestReg = MemMI->getOperand(0).getReg();
  unsigned BaseReg = MemMI->getOperand(1).getReg();
  int Offset = MemMI->getOperand(2).getImm() *
               TII->getRegClass(MemMI->getDesc(), 0, TRI, MF)->getSize();

  // If the base register overlaps the destination register, we can't
  // merge the update.
  if (DestReg == BaseReg || TRI->isSubRegister(BaseReg, DestReg))
    return E;

  // Scan forward looking for post-index opportunities.
  // Updating instructions can't be formed if the memory insn already
  // has an offset other than the value we're looking for.
  if (Offset != Value)
    return E;

  // Track which registers have been modified and used between the first insn
  // (inclusive) and the second insn.
  BitVector ModifiedRegs, UsedRegs;
  ModifiedRegs.resize(TRI->getNumRegs());
  UsedRegs.resize(TRI->getNumRegs());
  ++MBBI;
  for (unsigned Count = 0; MBBI != E; ++MBBI) {
    MachineInstr *MI = MBBI;
    // Skip DBG_VALUE instructions. Otherwise debug info can affect the
    // optimization by changing how far we scan.
    if (MI->isDebugValue())
      continue;

    // Now that we know this is a real instruction, count it.
    ++Count;

    // If we found a match, return it.
    if (isMatchingUpdateInsn(MI, BaseReg, Value))
      return MBBI;

    // Update the status of what the instruction clobbered and used.
    trackRegDefsUses(MI, ModifiedRegs, UsedRegs, TRI);

    // Otherwise, if the base register is used or modified, we have no match, so
    // return early.
    if (ModifiedRegs[BaseReg] || UsedRegs[BaseReg])
      return E;
  }
  return E;
}

MachineBasicBlock::iterator AArch64LoadStoreOpt::findMatchingUpdateInsnBackward(
    MachineBasicBlock::iterator I, unsigned Limit) {
  MachineBasicBlock::iterator B = I->getParent()->begin();
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineInstr *MemMI = I;
  MachineBasicBlock::iterator MBBI = I;
  const MachineFunction &MF = *MemMI->getParent()->getParent();

  unsigned DestReg = MemMI->getOperand(0).getReg();
  unsigned BaseReg = MemMI->getOperand(1).getReg();
  int Offset = MemMI->getOperand(2).getImm();
  unsigned RegSize = TII->getRegClass(MemMI->getDesc(), 0, TRI, MF)->getSize();

  // If the load/store is the first instruction in the block, there's obviously
  // not any matching update. Ditto if the memory offset isn't zero.
  if (MBBI == B || Offset != 0)
    return E;
  // If the base register overlaps the destination register, we can't
  // merge the update.
  if (DestReg == BaseReg || TRI->isSubRegister(BaseReg, DestReg))
    return E;

  // Track which registers have been modified and used between the first insn
  // (inclusive) and the second insn.
  BitVector ModifiedRegs, UsedRegs;
  ModifiedRegs.resize(TRI->getNumRegs());
  UsedRegs.resize(TRI->getNumRegs());
  --MBBI;
  for (unsigned Count = 0; MBBI != B; --MBBI) {
    MachineInstr *MI = MBBI;
    // Skip DBG_VALUE instructions. Otherwise debug info can affect the
    // optimization by changing how far we scan.
    if (MI->isDebugValue())
      continue;

    // Now that we know this is a real instruction, count it.
    ++Count;

    // If we found a match, return it.
    if (isMatchingUpdateInsn(MI, BaseReg, RegSize))
      return MBBI;

    // Update the status of what the instruction clobbered and used.
    trackRegDefsUses(MI, ModifiedRegs, UsedRegs, TRI);

    // Otherwise, if the base register is used or modified, we have no match, so
    // return early.
    if (ModifiedRegs[BaseReg] || UsedRegs[BaseReg])
      return E;
  }
  return E;
}

bool AArch64LoadStoreOpt::optimizeBlock(MachineBasicBlock &MBB) {
  bool Modified = false;
  // Two tranformations to do here:
  // 1) Find loads and stores that can be merged into a single load or store
  //    pair instruction.
  //      e.g.,
  //        ldr x0, [x2]
  //        ldr x1, [x2, #8]
  //        ; becomes
  //        ldp x0, x1, [x2]
  // 2) Find base register updates that can be merged into the load or store
  //    as a base-reg writeback.
  //      e.g.,
  //        ldr x0, [x2]
  //        add x2, x2, #4
  //        ; becomes
  //        ldr x0, [x2], #4

  for (MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
       MBBI != E;) {
    MachineInstr *MI = MBBI;
    switch (MI->getOpcode()) {
    default:
      // Just move on to the next instruction.
      ++MBBI;
      break;
    case AArch64::STRSui:
    case AArch64::STRDui:
    case AArch64::STRQui:
    case AArch64::STRXui:
    case AArch64::STRWui:
    case AArch64::LDRSui:
    case AArch64::LDRDui:
    case AArch64::LDRQui:
    case AArch64::LDRXui:
    case AArch64::LDRWui:
    case AArch64::LDRSWui:
    // do the unscaled versions as well
    case AArch64::STURSi:
    case AArch64::STURDi:
    case AArch64::STURQi:
    case AArch64::STURWi:
    case AArch64::STURXi:
    case AArch64::LDURSi:
    case AArch64::LDURDi:
    case AArch64::LDURQi:
    case AArch64::LDURWi:
    case AArch64::LDURXi:
    case AArch64::LDURSWi: {
      // If this is a volatile load/store, don't mess with it.
      if (MI->hasOrderedMemoryRef()) {
        ++MBBI;
        break;
      }
      // Make sure this is a reg+imm (as opposed to an address reloc).
      if (!MI->getOperand(2).isImm()) {
        ++MBBI;
        break;
      }
      // Check if this load/store has a hint to avoid pair formation.
      // MachineMemOperands hints are set by the AArch64StorePairSuppress pass.
      if (TII->isLdStPairSuppressed(MI)) {
        ++MBBI;
        break;
      }
      // Look ahead up to ScanLimit instructions for a pairable instruction.
      LdStPairFlags Flags;
      MachineBasicBlock::iterator Paired =
          findMatchingInsn(MBBI, Flags, ScanLimit);
      if (Paired != E) {
        // Merge the loads into a pair. Keeping the iterator straight is a
        // pain, so we let the merge routine tell us what the next instruction
        // is after it's done mucking about.
        MBBI = mergePairedInsns(MBBI, Paired, Flags);

        Modified = true;
        ++NumPairCreated;
        if (isUnscaledLdst(MI->getOpcode()))
          ++NumUnscaledPairCreated;
        break;
      }
      ++MBBI;
      break;
    }
      // FIXME: Do the other instructions.
    }
  }

  for (MachineBasicBlock::iterator MBBI = MBB.begin(), E = MBB.end();
       MBBI != E;) {
    MachineInstr *MI = MBBI;
    // Do update merging. It's simpler to keep this separate from the above
    // switch, though not strictly necessary.
    unsigned Opc = MI->getOpcode();
    switch (Opc) {
    default:
      // Just move on to the next instruction.
      ++MBBI;
      break;
    case AArch64::STRSui:
    case AArch64::STRDui:
    case AArch64::STRQui:
    case AArch64::STRXui:
    case AArch64::STRWui:
    case AArch64::LDRSui:
    case AArch64::LDRDui:
    case AArch64::LDRQui:
    case AArch64::LDRXui:
    case AArch64::LDRWui:
    // do the unscaled versions as well
    case AArch64::STURSi:
    case AArch64::STURDi:
    case AArch64::STURQi:
    case AArch64::STURWi:
    case AArch64::STURXi:
    case AArch64::LDURSi:
    case AArch64::LDURDi:
    case AArch64::LDURQi:
    case AArch64::LDURWi:
    case AArch64::LDURXi: {
      // Make sure this is a reg+imm (as opposed to an address reloc).
      if (!MI->getOperand(2).isImm()) {
        ++MBBI;
        break;
      }
      // Look ahead up to ScanLimit instructions for a mergable instruction.
      MachineBasicBlock::iterator Update =
          findMatchingUpdateInsnForward(MBBI, ScanLimit, 0);
      if (Update != E) {
        // Merge the update into the ld/st.
        MBBI = mergePostIdxUpdateInsn(MBBI, Update);
        Modified = true;
        ++NumPostFolded;
        break;
      }
      // Don't know how to handle pre/post-index versions, so move to the next
      // instruction.
      if (isUnscaledLdst(Opc)) {
        ++MBBI;
        break;
      }

      // Look back to try to find a pre-index instruction. For example,
      // add x0, x0, #8
      // ldr x1, [x0]
      //   merged into:
      // ldr x1, [x0, #8]!
      Update = findMatchingUpdateInsnBackward(MBBI, ScanLimit);
      if (Update != E) {
        // Merge the update into the ld/st.
        MBBI = mergePreIdxUpdateInsn(MBBI, Update);
        Modified = true;
        ++NumPreFolded;
        break;
      }

      // Look forward to try to find a post-index instruction. For example,
      // ldr x1, [x0, #64]
      // add x0, x0, #64
      //   merged into:
      // ldr x1, [x0, #64]!

      // The immediate in the load/store is scaled by the size of the register
      // being loaded. The immediate in the add we're looking for,
      // however, is not, so adjust here.
      int Value = MI->getOperand(2).getImm() *
                  TII->getRegClass(MI->getDesc(), 0, TRI, *(MBB.getParent()))
                      ->getSize();
      Update = findMatchingUpdateInsnForward(MBBI, ScanLimit, Value);
      if (Update != E) {
        // Merge the update into the ld/st.
        MBBI = mergePreIdxUpdateInsn(MBBI, Update);
        Modified = true;
        ++NumPreFolded;
        break;
      }

      // Nothing found. Just move to the next instruction.
      ++MBBI;
      break;
    }
      // FIXME: Do the other instructions.
    }
  }

  return Modified;
}

bool AArch64LoadStoreOpt::runOnMachineFunction(MachineFunction &Fn) {
  TII = static_cast<const AArch64InstrInfo *>(Fn.getSubtarget().getInstrInfo());
  TRI = Fn.getSubtarget().getRegisterInfo();

  bool Modified = false;
  for (auto &MBB : Fn)
    Modified |= optimizeBlock(MBB);

  return Modified;
}

// FIXME: Do we need/want a pre-alloc pass like ARM has to try to keep
// loads and stores near one another?

/// createARMLoadStoreOptimizationPass - returns an instance of the load / store
/// optimization pass.
FunctionPass *llvm::createAArch64LoadStoreOptimizationPass() {
  return new AArch64LoadStoreOpt();
}
