//===-- PeepholeOptimizer.cpp - Peephole Optimizations --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Perform peephole optimizations on the machine code:
//
// - Optimize Extensions
//
//     Optimization of sign / zero extension instructions. It may be extended to
//     handle other instructions with similar properties.
//
//     On some targets, some instructions, e.g. X86 sign / zero extension, may
//     leave the source value in the lower part of the result. This optimization
//     will replace some uses of the pre-extension value with uses of the
//     sub-register of the results.
//
// - Optimize Comparisons
//
//     Optimization of comparison instructions. For instance, in this code:
//
//       sub r1, 1
//       cmp r1, 0
//       bz  L1
//
//     If the "sub" instruction all ready sets (or could be modified to set) the
//     same flag that the "cmp" instruction sets and that "bz" uses, then we can
//     eliminate the "cmp" instruction.
//
//     Another instance, in this code:
//
//       sub r1, r3 | sub r1, imm
//       cmp r3, r1 or cmp r1, r3 | cmp r1, imm
//       bge L1
//
//     If the branch instruction can use flag from "sub", then we can replace
//     "sub" with "subs" and eliminate the "cmp" instruction.
//
// - Optimize Loads:
//
//     Loads that can be folded into a later instruction. A load is foldable
//     if it loads to virtual registers and the virtual register defined has 
//     a single use.
//
// - Optimize Copies and Bitcast:
//
//     Rewrite copies and bitcasts to avoid cross register bank copies
//     when possible.
//     E.g., Consider the following example, where capital and lower
//     letters denote different register file:
//     b = copy A <-- cross-bank copy
//     C = copy b <-- cross-bank copy
//   =>
//     b = copy A <-- cross-bank copy
//     C = copy A <-- same-bank copy
//
//     E.g., for bitcast:
//     b = bitcast A <-- cross-bank copy
//     C = bitcast b <-- cross-bank copy
//   =>
//     b = bitcast A <-- cross-bank copy
//     C = copy A    <-- same-bank copy
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

#define DEBUG_TYPE "peephole-opt"

// Optimize Extensions
static cl::opt<bool>
Aggressive("aggressive-ext-opt", cl::Hidden,
           cl::desc("Aggressive extension optimization"));

static cl::opt<bool>
DisablePeephole("disable-peephole", cl::Hidden, cl::init(false),
                cl::desc("Disable the peephole optimizer"));

static cl::opt<bool>
DisableAdvCopyOpt("disable-adv-copy-opt", cl::Hidden, cl::init(true),
                  cl::desc("Disable advanced copy optimization"));

STATISTIC(NumReuse,      "Number of extension results reused");
STATISTIC(NumCmps,       "Number of compares eliminated");
STATISTIC(NumImmFold,    "Number of move immediate folded");
STATISTIC(NumLoadFold,   "Number of loads folded");
STATISTIC(NumSelects,    "Number of selects optimized");
STATISTIC(NumCopiesBitcasts, "Number of copies/bitcasts optimized");

namespace {
  class PeepholeOptimizer : public MachineFunctionPass {
    const TargetMachine   *TM;
    const TargetInstrInfo *TII;
    MachineRegisterInfo   *MRI;
    MachineDominatorTree  *DT;  // Machine dominator tree

  public:
    static char ID; // Pass identification
    PeepholeOptimizer() : MachineFunctionPass(ID) {
      initializePeepholeOptimizerPass(*PassRegistry::getPassRegistry());
    }

    bool runOnMachineFunction(MachineFunction &MF) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      if (Aggressive) {
        AU.addRequired<MachineDominatorTree>();
        AU.addPreserved<MachineDominatorTree>();
      }
    }

  private:
    bool optimizeCmpInstr(MachineInstr *MI, MachineBasicBlock *MBB);
    bool optimizeExtInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                          SmallPtrSet<MachineInstr*, 8> &LocalMIs);
    bool optimizeSelect(MachineInstr *MI);
    bool optimizeCopyOrBitcast(MachineInstr *MI);
    bool isMoveImmediate(MachineInstr *MI,
                         SmallSet<unsigned, 4> &ImmDefRegs,
                         DenseMap<unsigned, MachineInstr*> &ImmDefMIs);
    bool foldImmediate(MachineInstr *MI, MachineBasicBlock *MBB,
                       SmallSet<unsigned, 4> &ImmDefRegs,
                       DenseMap<unsigned, MachineInstr*> &ImmDefMIs);
    bool isLoadFoldable(MachineInstr *MI,
                        SmallSet<unsigned, 16> &FoldAsLoadDefCandidates);
  };

  /// \brief Helper class to track the possible sources of a value defined by
  /// a (chain of) copy related instructions.
  /// Given a definition (instruction and definition index), this class
  /// follows the use-def chain to find successive suitable sources.
  /// The given source can be used to rewrite the definition into
  /// def = COPY src.
  ///
  /// For instance, let us consider the following snippet:
  /// v0 =
  /// v2 = INSERT_SUBREG v1, v0, sub0
  /// def = COPY v2.sub0
  ///
  /// Using a ValueTracker for def = COPY v2.sub0 will give the following
  /// suitable sources:
  /// v2.sub0 and v0.
  /// Then, def can be rewritten into def = COPY v0.
  class ValueTracker {
  private:
    /// The current point into the use-def chain.
    const MachineInstr *Def;
    /// The index of the definition in Def.
    unsigned DefIdx;
    /// The sub register index of the definition.
    unsigned DefSubReg;
    /// The register where the value can be found.
    unsigned Reg;
    /// Specifiy whether or not the value tracking looks through
    /// complex instructions. When this is false, the value tracker
    /// bails on everything that is not a copy or a bitcast.
    ///
    /// Note: This could have been implemented as a specialized version of
    /// the ValueTracker class but that would have complicated the code of
    /// the users of this class.
    bool UseAdvancedTracking;
    /// Optional MachineRegisterInfo used to perform some complex
    /// tracking.
    const MachineRegisterInfo *MRI;

    /// \brief Dispatcher to the right underlying implementation of
    /// getNextSource.
    bool getNextSourceImpl(unsigned &SrcIdx, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for Copy instructions.
    bool getNextSourceFromCopy(unsigned &SrcIdx, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for Bitcast instructions.
    bool getNextSourceFromBitcast(unsigned &SrcIdx, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for RegSequence
    /// instructions.
    bool getNextSourceFromRegSequence(unsigned &SrcIdx, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for InsertSubreg
    /// instructions.
    bool getNextSourceFromInsertSubreg(unsigned &SrcIdx, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for ExtractSubreg
    /// instructions.
    bool getNextSourceFromExtractSubreg(unsigned &SrcIdx, unsigned &SrcSubReg);
    /// \brief Specialized version of getNextSource for SubregToReg
    /// instructions.
    bool getNextSourceFromSubregToReg(unsigned &SrcIdx, unsigned &SrcSubReg);

  public:
    /// \brief Create a ValueTracker instance for the value defines by \p MI
    /// at the operand index \p DefIdx.
    /// \p DefSubReg represents the sub register index the value tracker will
    /// track. It does not need to match the sub register index used in \p MI.
    /// \p UseAdvancedTracking specifies whether or not the value tracker looks
    /// through complex instructions. By default (false), it handles only copy
    /// and bitcast instructions.
    /// \p MRI useful to perform some complex checks.
    ValueTracker(const MachineInstr &MI, unsigned DefIdx, unsigned DefSubReg,
                 bool UseAdvancedTracking = false,
                 const MachineRegisterInfo *MRI = nullptr)
        : Def(&MI), DefIdx(DefIdx), DefSubReg(DefSubReg),
          UseAdvancedTracking(UseAdvancedTracking), MRI(MRI) {
      assert(Def->getOperand(DefIdx).isDef() &&
             Def->getOperand(DefIdx).isReg() &&
             "Definition does not match machine instruction");
      // Initially the value is in the defined register.
      Reg = Def->getOperand(DefIdx).getReg();
    }

    /// \brief Following the use-def chain, get the next available source
    /// for the tracked value.
    /// When the returned value is not nullptr, getReg() gives the register
    /// that contain the tracked value.
    /// \note The sub register index returned in \p SrcSubReg must be used
    /// on that getReg() to access the actual value.
    /// \return Unless the returned value is nullptr (i.e., no source found),
    /// \p SrcIdx gives the index of the next source in the returned
    /// instruction and \p SrcSubReg the index to be used on that source to
    /// get the tracked value. When nullptr is returned, no alternative source
    /// has been found.
    const MachineInstr *getNextSource(unsigned &SrcIdx, unsigned &SrcSubReg);

    /// \brief Get the last register where the initial value can be found.
    /// Initially this is the register of the definition.
    /// Then, after each successful call to getNextSource, this is the
    /// register of the last source.
    unsigned getReg() const { return Reg; }
  };
}

char PeepholeOptimizer::ID = 0;
char &llvm::PeepholeOptimizerID = PeepholeOptimizer::ID;
INITIALIZE_PASS_BEGIN(PeepholeOptimizer, "peephole-opts",
                "Peephole Optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(PeepholeOptimizer, "peephole-opts",
                "Peephole Optimizations", false, false)

/// optimizeExtInstr - If instruction is a copy-like instruction, i.e. it reads
/// a single register and writes a single register and it does not modify the
/// source, and if the source value is preserved as a sub-register of the
/// result, then replace all reachable uses of the source with the subreg of the
/// result.
///
/// Do not generate an EXTRACT that is used only in a debug use, as this changes
/// the code. Since this code does not currently share EXTRACTs, just ignore all
/// debug uses.
bool PeepholeOptimizer::
optimizeExtInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                 SmallPtrSet<MachineInstr*, 8> &LocalMIs) {
  unsigned SrcReg, DstReg, SubIdx;
  if (!TII->isCoalescableExtInstr(*MI, SrcReg, DstReg, SubIdx))
    return false;

  if (TargetRegisterInfo::isPhysicalRegister(DstReg) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg))
    return false;

  if (MRI->hasOneNonDBGUse(SrcReg))
    // No other uses.
    return false;

  // Ensure DstReg can get a register class that actually supports
  // sub-registers. Don't change the class until we commit.
  const TargetRegisterClass *DstRC = MRI->getRegClass(DstReg);
  DstRC = TM->getRegisterInfo()->getSubClassWithSubReg(DstRC, SubIdx);
  if (!DstRC)
    return false;

  // The ext instr may be operating on a sub-register of SrcReg as well.
  // PPC::EXTSW is a 32 -> 64-bit sign extension, but it reads a 64-bit
  // register.
  // If UseSrcSubIdx is Set, SubIdx also applies to SrcReg, and only uses of
  // SrcReg:SubIdx should be replaced.
  bool UseSrcSubIdx = TM->getRegisterInfo()->
    getSubClassWithSubReg(MRI->getRegClass(SrcReg), SubIdx) != nullptr;

  // The source has other uses. See if we can replace the other uses with use of
  // the result of the extension.
  SmallPtrSet<MachineBasicBlock*, 4> ReachedBBs;
  for (MachineInstr &UI : MRI->use_nodbg_instructions(DstReg))
    ReachedBBs.insert(UI.getParent());

  // Uses that are in the same BB of uses of the result of the instruction.
  SmallVector<MachineOperand*, 8> Uses;

  // Uses that the result of the instruction can reach.
  SmallVector<MachineOperand*, 8> ExtendedUses;

  bool ExtendLife = true;
  for (MachineOperand &UseMO : MRI->use_nodbg_operands(SrcReg)) {
    MachineInstr *UseMI = UseMO.getParent();
    if (UseMI == MI)
      continue;

    if (UseMI->isPHI()) {
      ExtendLife = false;
      continue;
    }

    // Only accept uses of SrcReg:SubIdx.
    if (UseSrcSubIdx && UseMO.getSubReg() != SubIdx)
      continue;

    // It's an error to translate this:
    //
    //    %reg1025 = <sext> %reg1024
    //     ...
    //    %reg1026 = SUBREG_TO_REG 0, %reg1024, 4
    //
    // into this:
    //
    //    %reg1025 = <sext> %reg1024
    //     ...
    //    %reg1027 = COPY %reg1025:4
    //    %reg1026 = SUBREG_TO_REG 0, %reg1027, 4
    //
    // The problem here is that SUBREG_TO_REG is there to assert that an
    // implicit zext occurs. It doesn't insert a zext instruction. If we allow
    // the COPY here, it will give us the value after the <sext>, not the
    // original value of %reg1024 before <sext>.
    if (UseMI->getOpcode() == TargetOpcode::SUBREG_TO_REG)
      continue;

    MachineBasicBlock *UseMBB = UseMI->getParent();
    if (UseMBB == MBB) {
      // Local uses that come after the extension.
      if (!LocalMIs.count(UseMI))
        Uses.push_back(&UseMO);
    } else if (ReachedBBs.count(UseMBB)) {
      // Non-local uses where the result of the extension is used. Always
      // replace these unless it's a PHI.
      Uses.push_back(&UseMO);
    } else if (Aggressive && DT->dominates(MBB, UseMBB)) {
      // We may want to extend the live range of the extension result in order
      // to replace these uses.
      ExtendedUses.push_back(&UseMO);
    } else {
      // Both will be live out of the def MBB anyway. Don't extend live range of
      // the extension result.
      ExtendLife = false;
      break;
    }
  }

  if (ExtendLife && !ExtendedUses.empty())
    // Extend the liveness of the extension result.
    std::copy(ExtendedUses.begin(), ExtendedUses.end(),
              std::back_inserter(Uses));

  // Now replace all uses.
  bool Changed = false;
  if (!Uses.empty()) {
    SmallPtrSet<MachineBasicBlock*, 4> PHIBBs;

    // Look for PHI uses of the extended result, we don't want to extend the
    // liveness of a PHI input. It breaks all kinds of assumptions down
    // stream. A PHI use is expected to be the kill of its source values.
    for (MachineInstr &UI : MRI->use_nodbg_instructions(DstReg))
      if (UI.isPHI())
        PHIBBs.insert(UI.getParent());

    const TargetRegisterClass *RC = MRI->getRegClass(SrcReg);
    for (unsigned i = 0, e = Uses.size(); i != e; ++i) {
      MachineOperand *UseMO = Uses[i];
      MachineInstr *UseMI = UseMO->getParent();
      MachineBasicBlock *UseMBB = UseMI->getParent();
      if (PHIBBs.count(UseMBB))
        continue;

      // About to add uses of DstReg, clear DstReg's kill flags.
      if (!Changed) {
        MRI->clearKillFlags(DstReg);
        MRI->constrainRegClass(DstReg, DstRC);
      }

      unsigned NewVR = MRI->createVirtualRegister(RC);
      MachineInstr *Copy = BuildMI(*UseMBB, UseMI, UseMI->getDebugLoc(),
                                   TII->get(TargetOpcode::COPY), NewVR)
        .addReg(DstReg, 0, SubIdx);
      // SubIdx applies to both SrcReg and DstReg when UseSrcSubIdx is set.
      if (UseSrcSubIdx) {
        Copy->getOperand(0).setSubReg(SubIdx);
        Copy->getOperand(0).setIsUndef();
      }
      UseMO->setReg(NewVR);
      ++NumReuse;
      Changed = true;
    }
  }

  return Changed;
}

/// optimizeCmpInstr - If the instruction is a compare and the previous
/// instruction it's comparing against all ready sets (or could be modified to
/// set) the same flag as the compare, then we can remove the comparison and use
/// the flag from the previous instruction.
bool PeepholeOptimizer::optimizeCmpInstr(MachineInstr *MI,
                                         MachineBasicBlock *MBB) {
  // If this instruction is a comparison against zero and isn't comparing a
  // physical register, we can try to optimize it.
  unsigned SrcReg, SrcReg2;
  int CmpMask, CmpValue;
  if (!TII->analyzeCompare(MI, SrcReg, SrcReg2, CmpMask, CmpValue) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg) ||
      (SrcReg2 != 0 && TargetRegisterInfo::isPhysicalRegister(SrcReg2)))
    return false;

  // Attempt to optimize the comparison instruction.
  if (TII->optimizeCompareInstr(MI, SrcReg, SrcReg2, CmpMask, CmpValue, MRI)) {
    ++NumCmps;
    return true;
  }

  return false;
}

/// Optimize a select instruction.
bool PeepholeOptimizer::optimizeSelect(MachineInstr *MI) {
  unsigned TrueOp = 0;
  unsigned FalseOp = 0;
  bool Optimizable = false;
  SmallVector<MachineOperand, 4> Cond;
  if (TII->analyzeSelect(MI, Cond, TrueOp, FalseOp, Optimizable))
    return false;
  if (!Optimizable)
    return false;
  if (!TII->optimizeSelect(MI))
    return false;
  MI->eraseFromParent();
  ++NumSelects;
  return true;
}

/// \brief Check if the registers defined by the pair (RegisterClass, SubReg)
/// share the same register file.
static bool shareSameRegisterFile(const TargetRegisterInfo &TRI,
                                  const TargetRegisterClass *DefRC,
                                  unsigned DefSubReg,
                                  const TargetRegisterClass *SrcRC,
                                  unsigned SrcSubReg) {
  // Same register class.
  if (DefRC == SrcRC)
    return true;

  // Both operands are sub registers. Check if they share a register class.
  unsigned SrcIdx, DefIdx;
  if (SrcSubReg && DefSubReg)
    return TRI.getCommonSuperRegClass(SrcRC, SrcSubReg, DefRC, DefSubReg,
                                      SrcIdx, DefIdx) != nullptr;
  // At most one of the register is a sub register, make it Src to avoid
  // duplicating the test.
  if (!SrcSubReg) {
    std::swap(DefSubReg, SrcSubReg);
    std::swap(DefRC, SrcRC);
  }

  // One of the register is a sub register, check if we can get a superclass.
  if (SrcSubReg)
    return TRI.getMatchingSuperRegClass(SrcRC, DefRC, SrcSubReg) != nullptr;
  // Plain copy.
  return TRI.getCommonSubClass(DefRC, SrcRC) != nullptr;
}

/// \brief Get the index of the definition and source for \p Copy
/// instruction.
/// \pre Copy.isCopy() or Copy.isBitcast().
/// \return True if the Copy instruction has only one register source
/// and one register definition. Otherwise, \p DefIdx and \p SrcIdx
/// are invalid.
static bool getCopyOrBitcastDefUseIdx(const MachineInstr &Copy,
                                      unsigned &DefIdx, unsigned &SrcIdx) {
  assert((Copy.isCopy() || Copy.isBitcast()) && "Wrong operation type.");
  if (Copy.isCopy()) {
    // Copy instruction are supposed to be: Def = Src.
     if (Copy.getDesc().getNumOperands() != 2)
       return false;
     DefIdx = 0;
     SrcIdx = 1;
     assert(Copy.getOperand(DefIdx).isDef() && "Use comes before def!");
     return true;
  }
  // Bitcast case.
  // Bitcasts with more than one def are not supported.
  if (Copy.getDesc().getNumDefs() != 1)
    return false;
  // Initialize SrcIdx to an undefined operand.
  SrcIdx = Copy.getDesc().getNumOperands();
  for (unsigned OpIdx = 0, EndOpIdx = SrcIdx; OpIdx != EndOpIdx; ++OpIdx) {
    const MachineOperand &MO = Copy.getOperand(OpIdx);
    if (!MO.isReg() || !MO.getReg())
      continue;
    if (MO.isDef())
      DefIdx = OpIdx;
    else if (SrcIdx != EndOpIdx)
      // Multiple sources?
      return false;
    SrcIdx = OpIdx;
  }
  return true;
}

/// \brief Optimize a copy or bitcast instruction to avoid cross
/// register bank copy. The optimization looks through a chain of
/// copies and try to find a source that has a compatible register
/// class.
/// Two register classes are considered to be compatible if they share
/// the same register bank.
/// New copies issued by this optimization are register allocator
/// friendly. This optimization does not remove any copy as it may
/// overconstraint the register allocator, but replaces some when
/// possible.
/// \pre \p MI is a Copy (MI->isCopy() is true)
/// \return True, when \p MI has been optimized. In that case, \p MI has
/// been removed from its parent.
bool PeepholeOptimizer::optimizeCopyOrBitcast(MachineInstr *MI) {
  unsigned DefIdx, SrcIdx;
  if (!MI || !getCopyOrBitcastDefUseIdx(*MI, DefIdx, SrcIdx))
    return false;

  const MachineOperand &MODef = MI->getOperand(DefIdx);
  assert(MODef.isReg() && "Copies must be between registers.");
  unsigned Def = MODef.getReg();

  if (TargetRegisterInfo::isPhysicalRegister(Def))
    return false;

  const TargetRegisterClass *DefRC = MRI->getRegClass(Def);
  unsigned DefSubReg = MODef.getSubReg();

  unsigned Src;
  unsigned SrcSubReg;
  bool ShouldRewrite = false;
  const TargetRegisterInfo &TRI = *TM->getRegisterInfo();

  // Follow the chain of copies until we reach the top of the use-def chain
  // or find a more suitable source.
  ValueTracker ValTracker(*MI, DefIdx, DefSubReg, !DisableAdvCopyOpt, MRI);
  do {
    unsigned CopySrcIdx, CopySrcSubReg;
    if (!ValTracker.getNextSource(CopySrcIdx, CopySrcSubReg))
      break;
    Src = ValTracker.getReg();
    SrcSubReg = CopySrcSubReg;

    // Do not extend the live-ranges of physical registers as they add
    // constraints to the register allocator.
    // Moreover, if we want to extend the live-range of a physical register,
    // unlike SSA virtual register, we will have to check that they are not
    // redefine before the related use.
    if (TargetRegisterInfo::isPhysicalRegister(Src))
      break;

    const TargetRegisterClass *SrcRC = MRI->getRegClass(Src);

    // If this source does not incur a cross register bank copy, use it.
    ShouldRewrite = shareSameRegisterFile(TRI, DefRC, DefSubReg, SrcRC,
                                          SrcSubReg);
  } while (!ShouldRewrite);

  // If we did not find a more suitable source, there is nothing to optimize.
  if (!ShouldRewrite || Src == MI->getOperand(SrcIdx).getReg())
    return false;

  // Rewrite the copy to avoid a cross register bank penalty. 
  unsigned NewVR = TargetRegisterInfo::isPhysicalRegister(Def) ? Def :
    MRI->createVirtualRegister(DefRC);
  MachineInstr *NewCopy = BuildMI(*MI->getParent(), MI, MI->getDebugLoc(),
                                  TII->get(TargetOpcode::COPY), NewVR)
    .addReg(Src, 0, SrcSubReg);
  NewCopy->getOperand(0).setSubReg(DefSubReg);

  MRI->replaceRegWith(Def, NewVR);
  MRI->clearKillFlags(NewVR);
  // We extended the lifetime of Src.
  // Clear the kill flags to account for that.
  MRI->clearKillFlags(Src);
  MI->eraseFromParent();
  ++NumCopiesBitcasts;
  return true;
}

/// isLoadFoldable - Check whether MI is a candidate for folding into a later
/// instruction. We only fold loads to virtual registers and the virtual
/// register defined has a single use.
bool PeepholeOptimizer::isLoadFoldable(
                              MachineInstr *MI,
                              SmallSet<unsigned, 16> &FoldAsLoadDefCandidates) {
  if (!MI->canFoldAsLoad() || !MI->mayLoad())
    return false;
  const MCInstrDesc &MCID = MI->getDesc();
  if (MCID.getNumDefs() != 1)
    return false;

  unsigned Reg = MI->getOperand(0).getReg();
  // To reduce compilation time, we check MRI->hasOneNonDBGUse when inserting
  // loads. It should be checked when processing uses of the load, since
  // uses can be removed during peephole.
  if (!MI->getOperand(0).getSubReg() &&
      TargetRegisterInfo::isVirtualRegister(Reg) &&
      MRI->hasOneNonDBGUse(Reg)) {
    FoldAsLoadDefCandidates.insert(Reg);
    return true;
  }
  return false;
}

bool PeepholeOptimizer::isMoveImmediate(MachineInstr *MI,
                                        SmallSet<unsigned, 4> &ImmDefRegs,
                                 DenseMap<unsigned, MachineInstr*> &ImmDefMIs) {
  const MCInstrDesc &MCID = MI->getDesc();
  if (!MI->isMoveImmediate())
    return false;
  if (MCID.getNumDefs() != 1)
    return false;
  unsigned Reg = MI->getOperand(0).getReg();
  if (TargetRegisterInfo::isVirtualRegister(Reg)) {
    ImmDefMIs.insert(std::make_pair(Reg, MI));
    ImmDefRegs.insert(Reg);
    return true;
  }

  return false;
}

/// foldImmediate - Try folding register operands that are defined by move
/// immediate instructions, i.e. a trivial constant folding optimization, if
/// and only if the def and use are in the same BB.
bool PeepholeOptimizer::foldImmediate(MachineInstr *MI, MachineBasicBlock *MBB,
                                      SmallSet<unsigned, 4> &ImmDefRegs,
                                 DenseMap<unsigned, MachineInstr*> &ImmDefMIs) {
  for (unsigned i = 0, e = MI->getDesc().getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || MO.isDef())
      continue;
    unsigned Reg = MO.getReg();
    if (!TargetRegisterInfo::isVirtualRegister(Reg))
      continue;
    if (ImmDefRegs.count(Reg) == 0)
      continue;
    DenseMap<unsigned, MachineInstr*>::iterator II = ImmDefMIs.find(Reg);
    assert(II != ImmDefMIs.end());
    if (TII->FoldImmediate(MI, II->second, Reg, MRI)) {
      ++NumImmFold;
      return true;
    }
  }
  return false;
}

bool PeepholeOptimizer::runOnMachineFunction(MachineFunction &MF) {
  if (skipOptnoneFunction(*MF.getFunction()))
    return false;

  DEBUG(dbgs() << "********** PEEPHOLE OPTIMIZER **********\n");
  DEBUG(dbgs() << "********** Function: " << MF.getName() << '\n');

  if (DisablePeephole)
    return false;

  TM  = &MF.getTarget();
  TII = TM->getInstrInfo();
  MRI = &MF.getRegInfo();
  DT  = Aggressive ? &getAnalysis<MachineDominatorTree>() : nullptr;

  bool Changed = false;

  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;

    bool SeenMoveImm = false;
    SmallPtrSet<MachineInstr*, 8> LocalMIs;
    SmallSet<unsigned, 4> ImmDefRegs;
    DenseMap<unsigned, MachineInstr*> ImmDefMIs;
    SmallSet<unsigned, 16> FoldAsLoadDefCandidates;

    for (MachineBasicBlock::iterator
           MII = I->begin(), MIE = I->end(); MII != MIE; ) {
      MachineInstr *MI = &*MII;
      // We may be erasing MI below, increment MII now.
      ++MII;
      LocalMIs.insert(MI);

      // Skip debug values. They should not affect this peephole optimization.
      if (MI->isDebugValue())
          continue;

      // If there exists an instruction which belongs to the following
      // categories, we will discard the load candidates.
      if (MI->isPosition() || MI->isPHI() || MI->isImplicitDef() ||
          MI->isKill() || MI->isInlineAsm() ||
          MI->hasUnmodeledSideEffects()) {
        FoldAsLoadDefCandidates.clear();
        continue;
      }
      if (MI->mayStore() || MI->isCall())
        FoldAsLoadDefCandidates.clear();

      if (((MI->isBitcast() || MI->isCopy()) && optimizeCopyOrBitcast(MI)) ||
          (MI->isCompare() && optimizeCmpInstr(MI, MBB)) ||
          (MI->isSelect() && optimizeSelect(MI))) {
        // MI is deleted.
        LocalMIs.erase(MI);
        Changed = true;
        continue;
      }

      if (isMoveImmediate(MI, ImmDefRegs, ImmDefMIs)) {
        SeenMoveImm = true;
      } else {
        Changed |= optimizeExtInstr(MI, MBB, LocalMIs);
        // optimizeExtInstr might have created new instructions after MI
        // and before the already incremented MII. Adjust MII so that the
        // next iteration sees the new instructions.
        MII = MI;
        ++MII;
        if (SeenMoveImm)
          Changed |= foldImmediate(MI, MBB, ImmDefRegs, ImmDefMIs);
      }

      // Check whether MI is a load candidate for folding into a later
      // instruction. If MI is not a candidate, check whether we can fold an
      // earlier load into MI.
      if (!isLoadFoldable(MI, FoldAsLoadDefCandidates) &&
          !FoldAsLoadDefCandidates.empty()) {
        const MCInstrDesc &MIDesc = MI->getDesc();
        for (unsigned i = MIDesc.getNumDefs(); i != MIDesc.getNumOperands();
             ++i) {
          const MachineOperand &MOp = MI->getOperand(i);
          if (!MOp.isReg())
            continue;
          unsigned FoldAsLoadDefReg = MOp.getReg();
          if (FoldAsLoadDefCandidates.count(FoldAsLoadDefReg)) {
            // We need to fold load after optimizeCmpInstr, since
            // optimizeCmpInstr can enable folding by converting SUB to CMP.
            // Save FoldAsLoadDefReg because optimizeLoadInstr() resets it and
            // we need it for markUsesInDebugValueAsUndef().
            unsigned FoldedReg = FoldAsLoadDefReg;
            MachineInstr *DefMI = nullptr;
            MachineInstr *FoldMI = TII->optimizeLoadInstr(MI, MRI,
                                                          FoldAsLoadDefReg,
                                                          DefMI);
            if (FoldMI) {
              // Update LocalMIs since we replaced MI with FoldMI and deleted
              // DefMI.
              DEBUG(dbgs() << "Replacing: " << *MI);
              DEBUG(dbgs() << "     With: " << *FoldMI);
              LocalMIs.erase(MI);
              LocalMIs.erase(DefMI);
              LocalMIs.insert(FoldMI);
              MI->eraseFromParent();
              DefMI->eraseFromParent();
              MRI->markUsesInDebugValueAsUndef(FoldedReg);
              FoldAsLoadDefCandidates.erase(FoldedReg);
              ++NumLoadFold;
              // MI is replaced with FoldMI.
              Changed = true;
              break;
            }
          }
        }
      }
    }
  }

  return Changed;
}

bool ValueTracker::getNextSourceFromCopy(unsigned &SrcIdx,
                                         unsigned &SrcSubReg) {
  assert(Def->isCopy() && "Invalid definition");
  // Copy instruction are supposed to be: Def = Src.
  // If someone breaks this assumption, bad things will happen everywhere.
  assert(Def->getDesc().getNumOperands() == 2 && "Invalid number of operands");

  if (Def->getOperand(DefIdx).getSubReg() != DefSubReg)
    // If we look for a different subreg, it means we want a subreg of src.
    // Bails as we do not support composing subreg yet.
    return false;
  // Otherwise, we want the whole source.
  SrcIdx = 1;
  SrcSubReg = Def->getOperand(SrcIdx).getSubReg();
  return true;
}

bool ValueTracker::getNextSourceFromBitcast(unsigned &SrcIdx,
                                            unsigned &SrcSubReg) {
  assert(Def->isBitcast() && "Invalid definition");

  // Bail if there are effects that a plain copy will not expose.
  if (Def->hasUnmodeledSideEffects())
    return false;

  // Bitcasts with more than one def are not supported.
  if (Def->getDesc().getNumDefs() != 1)
    return false;
  if (Def->getOperand(DefIdx).getSubReg() != DefSubReg)
    // If we look for a different subreg, it means we want a subreg of the src.
    // Bails as we do not support composing subreg yet.
    return false;

  SrcIdx = Def->getDesc().getNumOperands();
  for (unsigned OpIdx = DefIdx + 1, EndOpIdx = SrcIdx; OpIdx != EndOpIdx;
       ++OpIdx) {
    const MachineOperand &MO = Def->getOperand(OpIdx);
    if (!MO.isReg() || !MO.getReg())
      continue;
    assert(!MO.isDef() && "We should have skipped all the definitions by now");
    if (SrcIdx != EndOpIdx)
      // Multiple sources?
      return false;
    SrcIdx = OpIdx;
  }
  SrcSubReg = Def->getOperand(SrcIdx).getSubReg();
  return true;
}

bool ValueTracker::getNextSourceFromRegSequence(unsigned &SrcIdx,
                                                unsigned &SrcSubReg) {
  assert(Def->isRegSequence() && "Invalid definition");

  if (Def->getOperand(DefIdx).getSubReg())
    // If we are composing subreg, bails out.
    // The case we are checking is Def.<subreg> = REG_SEQUENCE.
    // This should almost never happen as the SSA property is tracked at
    // the register level (as opposed to the subreg level).
    // I.e.,
    // Def.sub0 =
    // Def.sub1 =
    // is a valid SSA representation for Def.sub0 and Def.sub1, but not for
    // Def. Thus, it must not be generated.
    // However, some code could theoretically generates a single
    // Def.sub0 (i.e, not defining the other subregs) and we would
    // have this case.
    // If we can ascertain (or force) that this never happens, we could
    // turn that into an assertion.
    return false;

  // We are looking at:
  // Def = REG_SEQUENCE v0, sub0, v1, sub1, ...
  // Check if one of the operand defines the subreg we are interested in.
  for (unsigned OpIdx = DefIdx + 1, EndOpIdx = Def->getNumOperands();
       OpIdx != EndOpIdx; OpIdx += 2) {
    const MachineOperand &MOSubIdx = Def->getOperand(OpIdx + 1);
    assert(MOSubIdx.isImm() &&
           "One of the subindex of the reg_sequence is not an immediate");
    if (MOSubIdx.getImm() == DefSubReg) {
      assert(Def->getOperand(OpIdx).isReg() &&
             "One of the source of the reg_sequence is not a register");
      SrcIdx = OpIdx;
      SrcSubReg = Def->getOperand(SrcIdx).getSubReg();
      return true;
    }
  }

  // If the subreg we are tracking is super-defined by another subreg,
  // we could follow this value. However, this would require to compose
  // the subreg and we do not do that for now.
  return false;
}

bool ValueTracker::getNextSourceFromInsertSubreg(unsigned &SrcIdx,
                                                 unsigned &SrcSubReg) {
  assert(Def->isInsertSubreg() && "Invalid definition");
  if (Def->getOperand(DefIdx).getSubReg())
    // If we are composing subreg, bails out.
    // Same remark as getNextSourceFromRegSequence.
    // I.e., this may be turned into an assert.
    return false;

  // We are looking at:
  // Def = INSERT_SUBREG v0, v1, sub1
  // There are two cases:
  // 1. DefSubReg == sub1, get v1.
  // 2. DefSubReg != sub1, the value may be available through v0.

  // #1 Check if the inserted register matches the require sub index.
  unsigned InsertedSubReg = Def->getOperand(3).getImm();
  if (InsertedSubReg == DefSubReg) {
    SrcIdx = 2;
    SrcSubReg = Def->getOperand(SrcIdx).getSubReg();
    return true;
  }
  // #2 Otherwise, if the sub register we are looking for is not partial
  // defined by the inserted element, we can look through the main
  // register (v0).
  // To check the overlapping we need a MRI and a TRI.
  if (!MRI)
    return false;

  const MachineOperand &MODef = Def->getOperand(DefIdx);
  const MachineOperand &MOBase = Def->getOperand(1);
  // If the result register (Def) and the base register (v0) do not
  // have the same register class or if we have to compose
  // subregisters, bails out.
  if (MRI->getRegClass(MODef.getReg()) != MRI->getRegClass(MOBase.getReg()) ||
      MOBase.getSubReg())
    return false;

  // Get the TRI and check if inserted sub register overlaps with the
  // sub register we are tracking.
  const TargetRegisterInfo *TRI = MRI->getTargetRegisterInfo();
  if (!TRI ||
      (TRI->getSubRegIndexLaneMask(DefSubReg) &
       TRI->getSubRegIndexLaneMask(InsertedSubReg)) != 0)
    return false;
  // At this point, the value is available in v0 via the same subreg
  // we used for Def.
  SrcIdx = 1;
  SrcSubReg = DefSubReg;
  return true;
}

bool ValueTracker::getNextSourceFromExtractSubreg(unsigned &SrcIdx,
                                                  unsigned &SrcSubReg) {
  assert(Def->isExtractSubreg() && "Invalid definition");
  // We are looking at:
  // Def = EXTRACT_SUBREG v0, sub0

  // Bails if we have to compose sub registers.
  // Indeed, if DefSubReg != 0, we would have to compose it with sub0.
  if (DefSubReg)
    return false;

  // Bails if we have to compose sub registers.
  // Likewise, if v0.subreg != 0, we would have to compose v0.subreg with sub0.
  if (Def->getOperand(1).getSubReg())
    return false;
  // Otherwise, the value is available in the v0.sub0.
  SrcIdx = 1;
  SrcSubReg = Def->getOperand(2).getImm();
  return true;
}

bool ValueTracker::getNextSourceFromSubregToReg(unsigned &SrcIdx,
                                                unsigned &SrcSubReg) {
  assert(Def->isSubregToReg() && "Invalid definition");
  // We are looking at:
  // Def = SUBREG_TO_REG Imm, v0, sub0

  // Bails if we have to compose sub registers.
  // If DefSubReg != sub0, we would have to check that all the bits
  // we track are included in sub0 and if yes, we would have to
  // determine the right subreg in v0.
  if (DefSubReg != Def->getOperand(3).getImm())
    return false;
  // Bails if we have to compose sub registers.
  // Likewise, if v0.subreg != 0, we would have to compose it with sub0.
  if (Def->getOperand(2).getSubReg())
    return false;

  SrcIdx = 2;
  SrcSubReg = Def->getOperand(3).getImm();
  return true;
}

bool ValueTracker::getNextSourceImpl(unsigned &SrcIdx, unsigned &SrcSubReg) {
  assert(Def && "This method needs a valid definition");

  assert(
      (DefIdx < Def->getDesc().getNumDefs() || Def->getDesc().isVariadic()) &&
      Def->getOperand(DefIdx).isDef() && "Invalid DefIdx");
  if (Def->isCopy())
    return getNextSourceFromCopy(SrcIdx, SrcSubReg);
  if (Def->isBitcast())
    return getNextSourceFromBitcast(SrcIdx, SrcSubReg);
  // All the remaining cases involve "complex" instructions.
  // Bails if we did not ask for the advanced tracking.
  if (!UseAdvancedTracking)
    return false;
  if (Def->isRegSequence())
    return getNextSourceFromRegSequence(SrcIdx, SrcSubReg);
  if (Def->isInsertSubreg())
    return getNextSourceFromInsertSubreg(SrcIdx, SrcSubReg);
  if (Def->isExtractSubreg())
    return getNextSourceFromExtractSubreg(SrcIdx, SrcSubReg);
  if (Def->isSubregToReg())
    return getNextSourceFromSubregToReg(SrcIdx, SrcSubReg);
  return false;
}

const MachineInstr *ValueTracker::getNextSource(unsigned &SrcIdx,
                                                unsigned &SrcSubReg) {
  // If we reach a point where we cannot move up in the use-def chain,
  // there is nothing we can get.
  if (!Def)
    return nullptr;

  const MachineInstr *PrevDef = nullptr;
  // Try to find the next source.
  if (getNextSourceImpl(SrcIdx, SrcSubReg)) {
    // Update definition, definition index, and subregister for the
    // next call of getNextSource.
    const MachineOperand &MO = Def->getOperand(SrcIdx);
    assert(MO.isReg() && !MO.isDef() && "Source is invalid");
    // Update the current register.
    Reg = MO.getReg();
    // Update the return value before moving up in the use-def chain.
    PrevDef = Def;
    // If we can still move up in the use-def chain, move to the next
    // defintion.
    if (!TargetRegisterInfo::isPhysicalRegister(Reg)) {
      Def = MRI->getVRegDef(Reg);
      DefIdx = MRI->def_begin(Reg).getOperandNo();
      DefSubReg = SrcSubReg;
      return PrevDef;
    }
  }
  // If we end up here, this means we will not be able to find another source
  // for the next iteration.
  // Make sure any new call to getNextSource bails out early by cutting the
  // use-def chain.
  Def = nullptr;
  return PrevDef;
}
