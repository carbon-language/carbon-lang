//===-- ARM64LoadStoreOptimizer.cpp - ARM64 load/store opt. pass --*- C++ -*-=//
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

#define DEBUG_TYPE "arm64-ldst-opt"
#include "ARM64InstrInfo.h"
#include "MCTargetDesc/ARM64AddressingModes.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

/// ARM64AllocLoadStoreOpt - Post-register allocation pass to combine
/// load / store instructions to form ldp / stp instructions.

STATISTIC(NumPairCreated, "Number of load/store pair instructions generated");
STATISTIC(NumPostFolded, "Number of post-index updates folded");
STATISTIC(NumPreFolded, "Number of pre-index updates folded");
STATISTIC(NumUnscaledPairCreated,
          "Number of load/store from unscaled generated");

static cl::opt<bool> DoLoadStoreOpt("arm64-load-store-opt", cl::init(true),
                                    cl::Hidden);
static cl::opt<unsigned> ScanLimit("arm64-load-store-scan-limit", cl::init(20),
                                   cl::Hidden);

// Place holder while testing unscaled load/store combining
static cl::opt<bool>
EnableARM64UnscaledMemOp("arm64-unscaled-mem-op", cl::Hidden,
                         cl::desc("Allow ARM64 unscaled load/store combining"),
                         cl::init(true));

namespace {
struct ARM64LoadStoreOpt : public MachineFunctionPass {
  static char ID;
  ARM64LoadStoreOpt() : MachineFunctionPass(ID) {}

  const ARM64InstrInfo *TII;
  const TargetRegisterInfo *TRI;

  // Scan the instructions looking for a load/store that can be combined
  // with the current instruction into a load/store pair.
  // Return the matching instruction if one is found, else MBB->end().
  // If a matching instruction is found, mergeForward is set to true if the
  // merge is to remove the first instruction and replace the second with
  // a pair-wise insn, and false if the reverse is true.
  MachineBasicBlock::iterator findMatchingInsn(MachineBasicBlock::iterator I,
                                               bool &mergeForward,
                                               unsigned Limit);
  // Merge the two instructions indicated into a single pair-wise instruction.
  // If mergeForward is true, erase the first instruction and fold its
  // operation into the second. If false, the reverse. Return the instruction
  // following the first instruction (which may change during proecessing).
  MachineBasicBlock::iterator
  mergePairedInsns(MachineBasicBlock::iterator I,
                   MachineBasicBlock::iterator Paired, bool mergeForward);

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

  virtual bool runOnMachineFunction(MachineFunction &Fn);

  virtual const char *getPassName() const {
    return "ARM64 load / store optimization pass";
  }

private:
  int getMemSize(MachineInstr *MemMI);
};
char ARM64LoadStoreOpt::ID = 0;
}

static bool isUnscaledLdst(unsigned Opc) {
  switch (Opc) {
  default:
    return false;
  case ARM64::STURSi:
    return true;
  case ARM64::STURDi:
    return true;
  case ARM64::STURQi:
    return true;
  case ARM64::STURWi:
    return true;
  case ARM64::STURXi:
    return true;
  case ARM64::LDURSi:
    return true;
  case ARM64::LDURDi:
    return true;
  case ARM64::LDURQi:
    return true;
  case ARM64::LDURWi:
    return true;
  case ARM64::LDURXi:
    return true;
  }
}

// Size in bytes of the data moved by an unscaled load or store
int ARM64LoadStoreOpt::getMemSize(MachineInstr *MemMI) {
  switch (MemMI->getOpcode()) {
  default:
    llvm_unreachable("Opcode has has unknown size!");
  case ARM64::STRSui:
  case ARM64::STURSi:
    return 4;
  case ARM64::STRDui:
  case ARM64::STURDi:
    return 8;
  case ARM64::STRQui:
  case ARM64::STURQi:
    return 16;
  case ARM64::STRWui:
  case ARM64::STURWi:
    return 4;
  case ARM64::STRXui:
  case ARM64::STURXi:
    return 8;
  case ARM64::LDRSui:
  case ARM64::LDURSi:
    return 4;
  case ARM64::LDRDui:
  case ARM64::LDURDi:
    return 8;
  case ARM64::LDRQui:
  case ARM64::LDURQi:
    return 16;
  case ARM64::LDRWui:
  case ARM64::LDURWi:
    return 4;
  case ARM64::LDRXui:
  case ARM64::LDURXi:
    return 8;
  }
}

static unsigned getMatchingPairOpcode(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("Opcode has no pairwise equivalent!");
  case ARM64::STRSui:
  case ARM64::STURSi:
    return ARM64::STPSi;
  case ARM64::STRDui:
  case ARM64::STURDi:
    return ARM64::STPDi;
  case ARM64::STRQui:
  case ARM64::STURQi:
    return ARM64::STPQi;
  case ARM64::STRWui:
  case ARM64::STURWi:
    return ARM64::STPWi;
  case ARM64::STRXui:
  case ARM64::STURXi:
    return ARM64::STPXi;
  case ARM64::LDRSui:
  case ARM64::LDURSi:
    return ARM64::LDPSi;
  case ARM64::LDRDui:
  case ARM64::LDURDi:
    return ARM64::LDPDi;
  case ARM64::LDRQui:
  case ARM64::LDURQi:
    return ARM64::LDPQi;
  case ARM64::LDRWui:
  case ARM64::LDURWi:
    return ARM64::LDPWi;
  case ARM64::LDRXui:
  case ARM64::LDURXi:
    return ARM64::LDPXi;
  }
}

static unsigned getPreIndexedOpcode(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("Opcode has no pre-indexed equivalent!");
  case ARM64::STRSui:    return ARM64::STRSpre;
  case ARM64::STRDui:    return ARM64::STRDpre;
  case ARM64::STRQui:    return ARM64::STRQpre;
  case ARM64::STRWui:    return ARM64::STRWpre;
  case ARM64::STRXui:    return ARM64::STRXpre;
  case ARM64::LDRSui:    return ARM64::LDRSpre;
  case ARM64::LDRDui:    return ARM64::LDRDpre;
  case ARM64::LDRQui:    return ARM64::LDRQpre;
  case ARM64::LDRWui:    return ARM64::LDRWpre;
  case ARM64::LDRXui:    return ARM64::LDRXpre;
  }
}

static unsigned getPostIndexedOpcode(unsigned Opc) {
  switch (Opc) {
  default:
    llvm_unreachable("Opcode has no post-indexed wise equivalent!");
  case ARM64::STRSui:
    return ARM64::STRSpost;
  case ARM64::STRDui:
    return ARM64::STRDpost;
  case ARM64::STRQui:
    return ARM64::STRQpost;
  case ARM64::STRWui:
    return ARM64::STRWpost;
  case ARM64::STRXui:
    return ARM64::STRXpost;
  case ARM64::LDRSui:
    return ARM64::LDRSpost;
  case ARM64::LDRDui:
    return ARM64::LDRDpost;
  case ARM64::LDRQui:
    return ARM64::LDRQpost;
  case ARM64::LDRWui:
    return ARM64::LDRWpost;
  case ARM64::LDRXui:
    return ARM64::LDRXpost;
  }
}

MachineBasicBlock::iterator
ARM64LoadStoreOpt::mergePairedInsns(MachineBasicBlock::iterator I,
                                    MachineBasicBlock::iterator Paired,
                                    bool mergeForward) {
  MachineBasicBlock::iterator NextI = I;
  ++NextI;
  // If NextI is the second of the two instructions to be merged, we need
  // to skip one further. Either way we merge will invalidate the iterator,
  // and we don't need to scan the new instruction, as it's a pairwise
  // instruction, which we're not considering for further action anyway.
  if (NextI == Paired)
    ++NextI;

  bool IsUnscaled = isUnscaledLdst(I->getOpcode());
  int OffsetStride = IsUnscaled && EnableARM64UnscaledMemOp ? getMemSize(I) : 1;

  unsigned NewOpc = getMatchingPairOpcode(I->getOpcode());
  // Insert our new paired instruction after whichever of the paired
  // instructions mergeForward indicates.
  MachineBasicBlock::iterator InsertionPoint = mergeForward ? Paired : I;
  // Also based on mergeForward is from where we copy the base register operand
  // so we get the flags compatible with the input code.
  MachineOperand &BaseRegOp =
      mergeForward ? Paired->getOperand(1) : I->getOperand(1);

  // Which register is Rt and which is Rt2 depends on the offset order.
  MachineInstr *RtMI, *Rt2MI;
  if (I->getOperand(2).getImm() ==
      Paired->getOperand(2).getImm() + OffsetStride) {
    RtMI = Paired;
    Rt2MI = I;
  } else {
    RtMI = I;
    Rt2MI = Paired;
  }
  // Handle Unscaled
  int OffsetImm = RtMI->getOperand(2).getImm();
  if (IsUnscaled && EnableARM64UnscaledMemOp)
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
  DEBUG(((MachineInstr *)MIB)->print(dbgs()));
  DEBUG(dbgs() << "\n");

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
    int elemOffset = Offset / OffsetStride;
    if (elemOffset > 63 || elemOffset < -64)
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

/// findMatchingInsn - Scan the instructions looking for a load/store that can
/// be combined with the current instruction into a load/store pair.
MachineBasicBlock::iterator
ARM64LoadStoreOpt::findMatchingInsn(MachineBasicBlock::iterator I,
                                    bool &mergeForward, unsigned Limit) {
  MachineBasicBlock::iterator E = I->getParent()->end();
  MachineBasicBlock::iterator MBBI = I;
  MachineInstr *FirstMI = I;
  ++MBBI;

  int Opc = FirstMI->getOpcode();
  bool mayLoad = FirstMI->mayLoad();
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
      IsUnscaled && EnableARM64UnscaledMemOp ? getMemSize(FirstMI) : 1;
  if (!inBoundsForPair(IsUnscaled, Offset, OffsetStride))
    return E;

  // Track which registers have been modified and used between the first insn
  // (inclusive) and the second insn.
  BitVector ModifiedRegs, UsedRegs;
  ModifiedRegs.resize(TRI->getNumRegs());
  UsedRegs.resize(TRI->getNumRegs());
  for (unsigned Count = 0; MBBI != E && Count < Limit; ++MBBI) {
    MachineInstr *MI = MBBI;
    // Skip DBG_VALUE instructions. Otherwise debug info can affect the
    // optimization by changing how far we scan.
    if (MI->isDebugValue())
      continue;

    // Now that we know this is a real instruction, count it.
    ++Count;

    if (Opc == MI->getOpcode() && MI->getOperand(2).isImm()) {
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
          continue;
        }
        // If the alignment requirements of the paired (scaled) instruction
        // can't express the offset of the unscaled input, bail and keep
        // looking.
        if (IsUnscaled && EnableARM64UnscaledMemOp &&
            (alignTo(MinOffset, OffsetStride) != MinOffset)) {
          trackRegDefsUses(MI, ModifiedRegs, UsedRegs, TRI);
          continue;
        }
        // If the destination register of the loads is the same register, bail
        // and keep looking. A load-pair instruction with both destination
        // registers the same is UNPREDICTABLE and will result in an exception.
        if (mayLoad && Reg == MI->getOperand(0).getReg()) {
          trackRegDefsUses(MI, ModifiedRegs, UsedRegs, TRI);
          continue;
        }

        // If the Rt of the second instruction was not modified or used between
        // the two instructions, we can combine the second into the first.
        if (!ModifiedRegs[MI->getOperand(0).getReg()] &&
            !UsedRegs[MI->getOperand(0).getReg()]) {
          mergeForward = false;
          return MBBI;
        }

        // Likewise, if the Rt of the first instruction is not modified or used
        // between the two instructions, we can combine the first into the
        // second.
        if (!ModifiedRegs[FirstMI->getOperand(0).getReg()] &&
            !UsedRegs[FirstMI->getOperand(0).getReg()]) {
          mergeForward = true;
          return MBBI;
        }
        // Unable to combine these instructions due to interference in between.
        // Keep looking.
      }
    }

    // If the instruction wasn't a matching load or store, but does (or can)
    // modify memory, stop searching, as we don't have alias analysis or
    // anything like that to tell us whether the access is tromping on the
    // locations we care about. The big one we want to catch is calls.
    //
    // FIXME: Theoretically, we can do better than that for SP and FP based
    // references since we can effectively know where those are touching. It's
    // unclear if it's worth the extra code, though. Most paired instructions
    // will be sequential, perhaps with a few intervening non-memory related
    // instructions.
    if (MI->mayStore() || MI->isCall())
      return E;
    // Likewise, if we're matching a store instruction, we don't want to
    // move across a load, as it may be reading the same location.
    if (FirstMI->mayStore() && MI->mayLoad())
      return E;

    // Update modified / uses register lists.
    trackRegDefsUses(MI, ModifiedRegs, UsedRegs, TRI);

    // Otherwise, if the base register is modified, we have no match, so
    // return early.
    if (ModifiedRegs[BaseReg])
      return E;
  }
  return E;
}

MachineBasicBlock::iterator
ARM64LoadStoreOpt::mergePreIdxUpdateInsn(MachineBasicBlock::iterator I,
                                         MachineBasicBlock::iterator Update) {
  assert((Update->getOpcode() == ARM64::ADDXri ||
          Update->getOpcode() == ARM64::SUBXri) &&
         "Unexpected base register update instruction to merge!");
  MachineBasicBlock::iterator NextI = I;
  // Return the instruction following the merged instruction, which is
  // the instruction following our unmerged load. Unless that's the add/sub
  // instruction we're merging, in which case it's the one after that.
  if (++NextI == Update)
    ++NextI;

  int Value = Update->getOperand(2).getImm();
  assert(ARM64_AM::getShiftValue(Update->getOperand(3).getImm()) == 0 &&
         "Can't merge 1 << 12 offset into pre-indexed load / store");
  if (Update->getOpcode() == ARM64::SUBXri)
    Value = -Value;

  unsigned NewOpc = getPreIndexedOpcode(I->getOpcode());
  MachineInstrBuilder MIB =
      BuildMI(*I->getParent(), I, I->getDebugLoc(), TII->get(NewOpc))
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

MachineBasicBlock::iterator
ARM64LoadStoreOpt::mergePostIdxUpdateInsn(MachineBasicBlock::iterator I,
                                          MachineBasicBlock::iterator Update) {
  assert((Update->getOpcode() == ARM64::ADDXri ||
          Update->getOpcode() == ARM64::SUBXri) &&
         "Unexpected base register update instruction to merge!");
  MachineBasicBlock::iterator NextI = I;
  // Return the instruction following the merged instruction, which is
  // the instruction following our unmerged load. Unless that's the add/sub
  // instruction we're merging, in which case it's the one after that.
  if (++NextI == Update)
    ++NextI;

  int Value = Update->getOperand(2).getImm();
  assert(ARM64_AM::getShiftValue(Update->getOperand(3).getImm()) == 0 &&
         "Can't merge 1 << 12 offset into post-indexed load / store");
  if (Update->getOpcode() == ARM64::SUBXri)
    Value = -Value;

  unsigned NewOpc = getPostIndexedOpcode(I->getOpcode());
  MachineInstrBuilder MIB =
      BuildMI(*I->getParent(), I, I->getDebugLoc(), TII->get(NewOpc))
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
  case ARM64::SUBXri:
    // Negate the offset for a SUB instruction.
    Offset *= -1;
  // FALLTHROUGH
  case ARM64::ADDXri:
    // Make sure it's a vanilla immediate operand, not a relocation or
    // anything else we can't handle.
    if (!MI->getOperand(2).isImm())
      break;
    // Watch out for 1 << 12 shifted value.
    if (ARM64_AM::getShiftValue(MI->getOperand(3).getImm()))
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

MachineBasicBlock::iterator
ARM64LoadStoreOpt::findMatchingUpdateInsnForward(MachineBasicBlock::iterator I,
                                                 unsigned Limit, int Value) {
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

MachineBasicBlock::iterator
ARM64LoadStoreOpt::findMatchingUpdateInsnBackward(MachineBasicBlock::iterator I,
                                                  unsigned Limit) {
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

bool ARM64LoadStoreOpt::optimizeBlock(MachineBasicBlock &MBB) {
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
    case ARM64::STRSui:
    case ARM64::STRDui:
    case ARM64::STRQui:
    case ARM64::STRXui:
    case ARM64::STRWui:
    case ARM64::LDRSui:
    case ARM64::LDRDui:
    case ARM64::LDRQui:
    case ARM64::LDRXui:
    case ARM64::LDRWui:
    // do the unscaled versions as well
    case ARM64::STURSi:
    case ARM64::STURDi:
    case ARM64::STURQi:
    case ARM64::STURWi:
    case ARM64::STURXi:
    case ARM64::LDURSi:
    case ARM64::LDURDi:
    case ARM64::LDURQi:
    case ARM64::LDURWi:
    case ARM64::LDURXi: {
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
      // MachineMemOperands hints are set by the ARM64StorePairSuppress pass.
      if (TII->isLdStPairSuppressed(MI)) {
        ++MBBI;
        break;
      }
      // Look ahead up to ScanLimit instructions for a pairable instruction.
      bool mergeForward = false;
      MachineBasicBlock::iterator Paired =
          findMatchingInsn(MBBI, mergeForward, ScanLimit);
      if (Paired != E) {
        // Merge the loads into a pair. Keeping the iterator straight is a
        // pain, so we let the merge routine tell us what the next instruction
        // is after it's done mucking about.
        MBBI = mergePairedInsns(MBBI, Paired, mergeForward);

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
    int Opc = MI->getOpcode();
    switch (Opc) {
    default:
      // Just move on to the next instruction.
      ++MBBI;
      break;
    case ARM64::STRSui:
    case ARM64::STRDui:
    case ARM64::STRQui:
    case ARM64::STRXui:
    case ARM64::STRWui:
    case ARM64::LDRSui:
    case ARM64::LDRDui:
    case ARM64::LDRQui:
    case ARM64::LDRXui:
    case ARM64::LDRWui:
    // do the unscaled versions as well
    case ARM64::STURSi:
    case ARM64::STURDi:
    case ARM64::STURQi:
    case ARM64::STURWi:
    case ARM64::STURXi:
    case ARM64::LDURSi:
    case ARM64::LDURDi:
    case ARM64::LDURQi:
    case ARM64::LDURWi:
    case ARM64::LDURXi: {
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
      // ldr x1, [x0], #64

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

bool ARM64LoadStoreOpt::runOnMachineFunction(MachineFunction &Fn) {
  // Early exit if pass disabled.
  if (!DoLoadStoreOpt)
    return false;

  const TargetMachine &TM = Fn.getTarget();
  TII = static_cast<const ARM64InstrInfo *>(TM.getInstrInfo());
  TRI = TM.getRegisterInfo();

  bool Modified = false;
  for (MachineFunction::iterator MFI = Fn.begin(), E = Fn.end(); MFI != E;
       ++MFI) {
    MachineBasicBlock &MBB = *MFI;
    Modified |= optimizeBlock(MBB);
  }

  return Modified;
}

// FIXME: Do we need/want a pre-alloc pass like ARM has to try to keep
// loads and stores near one another?

/// createARMLoadStoreOptimizationPass - returns an instance of the load / store
/// optimization pass.
FunctionPass *llvm::createARM64LoadStoreOptimizationPass() {
  return new ARM64LoadStoreOpt();
}
