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
// - Optimize Bitcast pairs:
//
//     v1 = bitcast v0
//     v2 = bitcast v1
//        = v2
//   =>
//     v1 = bitcast v0
//        = v0
// 
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "peephole-opt"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
using namespace llvm;

// Optimize Extensions
static cl::opt<bool>
Aggressive("aggressive-ext-opt", cl::Hidden,
           cl::desc("Aggressive extension optimization"));

static cl::opt<bool>
DisablePeephole("disable-peephole", cl::Hidden, cl::init(false),
                cl::desc("Disable the peephole optimizer"));

STATISTIC(NumReuse,      "Number of extension results reused");
STATISTIC(NumBitcasts,   "Number of bitcasts eliminated");
STATISTIC(NumCmps,       "Number of compares eliminated");
STATISTIC(NumImmFold,    "Number of move immediate foled");

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

    virtual bool runOnMachineFunction(MachineFunction &MF);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      if (Aggressive) {
        AU.addRequired<MachineDominatorTree>();
        AU.addPreserved<MachineDominatorTree>();
      }
    }

  private:
    bool OptimizeBitcastInstr(MachineInstr *MI, MachineBasicBlock *MBB);
    bool OptimizeCmpInstr(MachineInstr *MI, MachineBasicBlock *MBB);
    bool OptimizeExtInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                          SmallPtrSet<MachineInstr*, 8> &LocalMIs);
    bool isMoveImmediate(MachineInstr *MI,
                         SmallSet<unsigned, 4> &ImmDefRegs,
                         DenseMap<unsigned, MachineInstr*> &ImmDefMIs);
    bool FoldImmediate(MachineInstr *MI, MachineBasicBlock *MBB,
                       SmallSet<unsigned, 4> &ImmDefRegs,
                       DenseMap<unsigned, MachineInstr*> &ImmDefMIs);
  };
}

char PeepholeOptimizer::ID = 0;
INITIALIZE_PASS_BEGIN(PeepholeOptimizer, "peephole-opts",
                "Peephole Optimizations", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_END(PeepholeOptimizer, "peephole-opts",
                "Peephole Optimizations", false, false)

FunctionPass *llvm::createPeepholeOptimizerPass() {
  return new PeepholeOptimizer();
}

/// OptimizeExtInstr - If instruction is a copy-like instruction, i.e. it reads
/// a single register and writes a single register and it does not modify the
/// source, and if the source value is preserved as a sub-register of the
/// result, then replace all reachable uses of the source with the subreg of the
/// result.
/// 
/// Do not generate an EXTRACT that is used only in a debug use, as this changes
/// the code. Since this code does not currently share EXTRACTs, just ignore all
/// debug uses.
bool PeepholeOptimizer::
OptimizeExtInstr(MachineInstr *MI, MachineBasicBlock *MBB,
                 SmallPtrSet<MachineInstr*, 8> &LocalMIs) {
  unsigned SrcReg, DstReg, SubIdx;
  if (!TII->isCoalescableExtInstr(*MI, SrcReg, DstReg, SubIdx))
    return false;
  
  if (TargetRegisterInfo::isPhysicalRegister(DstReg) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg))
    return false;

  MachineRegisterInfo::use_nodbg_iterator UI = MRI->use_nodbg_begin(SrcReg);
  if (++UI == MRI->use_nodbg_end())
    // No other uses.
    return false;

  // The source has other uses. See if we can replace the other uses with use of
  // the result of the extension.
  SmallPtrSet<MachineBasicBlock*, 4> ReachedBBs;
  UI = MRI->use_nodbg_begin(DstReg);
  for (MachineRegisterInfo::use_nodbg_iterator UE = MRI->use_nodbg_end();
       UI != UE; ++UI)
    ReachedBBs.insert(UI->getParent());

  // Uses that are in the same BB of uses of the result of the instruction.
  SmallVector<MachineOperand*, 8> Uses;

  // Uses that the result of the instruction can reach.
  SmallVector<MachineOperand*, 8> ExtendedUses;

  bool ExtendLife = true;
  UI = MRI->use_nodbg_begin(SrcReg);
  for (MachineRegisterInfo::use_nodbg_iterator UE = MRI->use_nodbg_end();
       UI != UE; ++UI) {
    MachineOperand &UseMO = UI.getOperand();
    MachineInstr *UseMI = &*UI;
    if (UseMI == MI)
      continue;

    if (UseMI->isPHI()) {
      ExtendLife = false;
      continue;
    }

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
    UI = MRI->use_nodbg_begin(DstReg);
    for (MachineRegisterInfo::use_nodbg_iterator
           UE = MRI->use_nodbg_end(); UI != UE; ++UI)
      if (UI->isPHI())
        PHIBBs.insert(UI->getParent());

    const TargetRegisterClass *RC = MRI->getRegClass(SrcReg);
    for (unsigned i = 0, e = Uses.size(); i != e; ++i) {
      MachineOperand *UseMO = Uses[i];
      MachineInstr *UseMI = UseMO->getParent();
      MachineBasicBlock *UseMBB = UseMI->getParent();
      if (PHIBBs.count(UseMBB))
        continue;

      unsigned NewVR = MRI->createVirtualRegister(RC);
      BuildMI(*UseMBB, UseMI, UseMI->getDebugLoc(),
              TII->get(TargetOpcode::COPY), NewVR)
        .addReg(DstReg, 0, SubIdx);

      UseMO->setReg(NewVR);
      ++NumReuse;
      Changed = true;
    }
  }

  return Changed;
}

/// OptimizeBitcastInstr - If the instruction is a bitcast instruction A that
/// cannot be optimized away during isel (e.g. ARM::VMOVSR, which bitcast
/// a value cross register classes), and the source is defined by another
/// bitcast instruction B. And if the register class of source of B matches
/// the register class of instruction A, then it is legal to replace all uses
/// of the def of A with source of B. e.g.
///   %vreg0<def> = VMOVSR %vreg1
///   %vreg3<def> = VMOVRS %vreg0
///   Replace all uses of vreg3 with vreg1.

bool PeepholeOptimizer::OptimizeBitcastInstr(MachineInstr *MI,
                                             MachineBasicBlock *MBB) {
  unsigned NumDefs = MI->getDesc().getNumDefs();
  unsigned NumSrcs = MI->getDesc().getNumOperands() - NumDefs;
  if (NumDefs != 1)
    return false;

  unsigned Def = 0;
  unsigned Src = 0;
  for (unsigned i = 0, e = NumDefs + NumSrcs; i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (MO.isDef())
      Def = Reg;
    else if (Src)
      // Multiple sources?
      return false;
    else
      Src = Reg;
  }

  assert(Def && Src && "Malformed bitcast instruction!");

  MachineInstr *DefMI = MRI->getVRegDef(Src);
  if (!DefMI || !DefMI->getDesc().isBitcast())
    return false;

  unsigned SrcDef = 0;
  unsigned SrcSrc = 0;
  NumDefs = DefMI->getDesc().getNumDefs();
  NumSrcs = DefMI->getDesc().getNumOperands() - NumDefs;
  if (NumDefs != 1)
    return false;
  for (unsigned i = 0, e = NumDefs + NumSrcs; i != e; ++i) {
    const MachineOperand &MO = DefMI->getOperand(i);
    if (!MO.isReg() || MO.isDef())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (MO.isDef())
      SrcDef = Reg;
    else if (SrcSrc)
      // Multiple sources?
      return false;
    else
      SrcSrc = Reg;
  }

  if (MRI->getRegClass(SrcSrc) != MRI->getRegClass(Def))
    return false;

  MRI->replaceRegWith(Def, SrcSrc);
  MRI->clearKillFlags(SrcSrc);
  MI->eraseFromParent();
  ++NumBitcasts;
  return true;
}

/// OptimizeCmpInstr - If the instruction is a compare and the previous
/// instruction it's comparing against all ready sets (or could be modified to
/// set) the same flag as the compare, then we can remove the comparison and use
/// the flag from the previous instruction.
bool PeepholeOptimizer::OptimizeCmpInstr(MachineInstr *MI,
                                         MachineBasicBlock *MBB) {
  // If this instruction is a comparison against zero and isn't comparing a
  // physical register, we can try to optimize it.
  unsigned SrcReg;
  int CmpMask, CmpValue;
  if (!TII->AnalyzeCompare(MI, SrcReg, CmpMask, CmpValue) ||
      TargetRegisterInfo::isPhysicalRegister(SrcReg))
    return false;

  // Attempt to optimize the comparison instruction.
  if (TII->OptimizeCompareInstr(MI, SrcReg, CmpMask, CmpValue, MRI)) {
    ++NumCmps;
    return true;
  }

  return false;
}

bool PeepholeOptimizer::isMoveImmediate(MachineInstr *MI,
                                        SmallSet<unsigned, 4> &ImmDefRegs,
                                 DenseMap<unsigned, MachineInstr*> &ImmDefMIs) {
  const MCInstrDesc &MCID = MI->getDesc();
  if (!MCID.isMoveImmediate())
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

/// FoldImmediate - Try folding register operands that are defined by move
/// immediate instructions, i.e. a trivial constant folding optimization, if
/// and only if the def and use are in the same BB.
bool PeepholeOptimizer::FoldImmediate(MachineInstr *MI, MachineBasicBlock *MBB,
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
  if (DisablePeephole)
    return false;
  
  TM  = &MF.getTarget();
  TII = TM->getInstrInfo();
  MRI = &MF.getRegInfo();
  DT  = Aggressive ? &getAnalysis<MachineDominatorTree>() : 0;

  bool Changed = false;

  SmallPtrSet<MachineInstr*, 8> LocalMIs;
  SmallSet<unsigned, 4> ImmDefRegs;
  DenseMap<unsigned, MachineInstr*> ImmDefMIs;
  for (MachineFunction::iterator I = MF.begin(), E = MF.end(); I != E; ++I) {
    MachineBasicBlock *MBB = &*I;
    
    bool SeenMoveImm = false;
    LocalMIs.clear();
    ImmDefRegs.clear();
    ImmDefMIs.clear();

    bool First = true;
    MachineBasicBlock::iterator PMII;
    for (MachineBasicBlock::iterator
           MII = I->begin(), MIE = I->end(); MII != MIE; ) {
      MachineInstr *MI = &*MII;
      LocalMIs.insert(MI);

      if (MI->isLabel() || MI->isPHI() || MI->isImplicitDef() ||
          MI->isKill() || MI->isInlineAsm() || MI->isDebugValue() ||
          MI->hasUnmodeledSideEffects()) {
        ++MII;
        continue;
      }

      const MCInstrDesc &MCID = MI->getDesc();

      if (MCID.isBitcast()) {
        if (OptimizeBitcastInstr(MI, MBB)) {
          // MI is deleted.
          Changed = true;
          MII = First ? I->begin() : llvm::next(PMII);
          continue;
        }        
      } else if (MCID.isCompare()) {
        if (OptimizeCmpInstr(MI, MBB)) {
          // MI is deleted.
          Changed = true;
          MII = First ? I->begin() : llvm::next(PMII);
          continue;
        }
      }

      if (isMoveImmediate(MI, ImmDefRegs, ImmDefMIs)) {
        SeenMoveImm = true;
      } else {
        Changed |= OptimizeExtInstr(MI, MBB, LocalMIs);
        if (SeenMoveImm)
          Changed |= FoldImmediate(MI, MBB, ImmDefRegs, ImmDefMIs);
      }

      First = false;
      PMII = MII;
      ++MII;
    }
  }

  return Changed;
}
