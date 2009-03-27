//===-- PrologEpilogInserter.cpp - Insert Prolog/Epilog code in function --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is responsible for finalizing the functions frame layout, saving
// callee saved registers, and for emitting prolog & epilog code for the
// function.
//
// This pass must be run after register allocation.  After this pass is
// executed, it is illegal to construct MO_FrameIndex operands.
//
// This pass implements a shrink wrapping variant of prolog/epilog insertion:
// - Places callee saved register (CSR) spills and restores in the CFG to
//   tightly surround uses so that execution paths that do not use CSRs do not
//   pay the spill/restore penalty.
//
// - Avoiding placment of spills/restores in loops: if a CSR is used inside a
//   loop(nest), the spills are placed in the loop preheader, and restores are
//   placed in the loop exit nodes (the successors of the loop _exiting_ nodes).
//
// - Covering paths without CSR uses: e.g. if a restore is placed in a join
//   block, a matching spill is added to the end of all immediate predecessor
//   blocks that are not reached by a spill. Similarly for saves placed in
//   branch blocks.
//
// Shrink wrapping uses an analysis similar to the one in GVNPRE to determine
// which basic blocks require callee-saved register save/restore code.
//
// This pass uses MachineDominators and MachineLoopInfo. Loop information
// is used to prevent shrink wrapping of callee-saved register save/restore
// code into loops.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "shrink-wrap"

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/RegisterScavenging.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetFrameInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/SparseBitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/STLExtras.h"
#include <climits>
#include <sstream>

using namespace llvm;

// Shrink Wrapping:
static cl::opt<bool>
ShrinkWrapping("shrink-wrap",
  cl::desc("Apply shrink wrapping to callee-saved register spills/restores"));

namespace {
  struct VISIBILITY_HIDDEN PEI : public MachineFunctionPass {
    static char ID;
    PEI() : MachineFunctionPass(&ID) {}

    const char *getPassName() const {
      return "Prolog/Epilog Insertion & Frame Finalization";
    }

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      if (ShrinkWrapping) {
        AU.addRequired<MachineLoopInfo>();
        AU.addRequired<MachineDominatorTree>();
      }
      AU.addPreserved<MachineLoopInfo>();
      AU.addPreserved<MachineDominatorTree>();
      MachineFunctionPass::getAnalysisUsage(AU);
    }

    /// runOnMachineFunction - Insert prolog/epilog code and replace abstract
    /// frame indexes with appropriate references.
    ///
    bool runOnMachineFunction(MachineFunction &Fn) {
      const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();
      RS = TRI->requiresRegisterScavenging(Fn) ? new RegScavenger() : NULL;

      // Get MachineModuleInfo so that we can track the construction of the
      // frame.
      if (MachineModuleInfo *MMI = getAnalysisIfAvailable<MachineModuleInfo>())
        Fn.getFrameInfo()->setMachineModuleInfo(MMI);

      // Allow the target machine to make some adjustments to the function
      // e.g. UsedPhysRegs before calculateCalleeSavedRegisters.
      TRI->processFunctionBeforeCalleeSavedScan(Fn, RS);

      // Scan the function for modified callee saved registers and insert spill
      // code for any callee saved registers that are modified.  Also calculate
      // the MaxCallFrameSize and HasCalls variables for the function's frame
      // information and eliminates call frame pseudo instructions.
      calculateCalleeSavedRegisters(Fn);

      // Determine placement of CSR spill/restore code:
      //  - with shrink wrapping, place spills and restores to tightly
      //    enclose regions in the Machine CFG of the function where
      //    they are used. Without shrink wrapping
      //  - default (no shrink wrapping), place all spills in the
      //    entry block, all restores in return blocks.
      placeCSRSpillsAndRestores(Fn);

      // Add the code to save and restore the callee saved registers
      insertCSRSpillsAndRestores(Fn);

      // Allow the target machine to make final modifications to the function
      // before the frame layout is finalized.
      TRI->processFunctionBeforeFrameFinalized(Fn);

      // Calculate actual frame offsets for all of the abstract stack objects...
      calculateFrameObjectOffsets(Fn);

      // Add prolog and epilog code to the function.  This function is required
      // to align the stack frame as necessary for any stack variables or
      // called functions.  Because of this, calculateCalleeSavedRegisters
      // must be called before this function in order to set the HasCalls
      // and MaxCallFrameSize variables.
      insertPrologEpilogCode(Fn);

      // Replace all MO_FrameIndex operands with physical register references
      // and actual offsets.
      //
      replaceFrameIndices(Fn);

      delete RS;
      return true;
    }

  private:
    RegScavenger *RS;

    // MinCSFrameIndex, MaxCSFrameIndex - Keeps the range of callee saved
    // stack frame indexes.
    unsigned MinCSFrameIndex, MaxCSFrameIndex;

    // Analysis info for spill/restore placement.
    // "CSR": "callee saved register".

    // CSRegSet contains indices into the Callee Saved Register Info
    // vector built by calculateCalleeSavedRegisters() and accessed
    // via MF.getFrameInfo()->getCalleeSavedInfo().
    typedef SparseBitVector<> CSRegSet;

    // CSRegBlockMap maps MachineBasicBlocks to sets of callee
    // saved register indices.
    typedef DenseMap<MachineBasicBlock*, CSRegSet> CSRegBlockMap;

    // Set and maps for computing CSR spill/restore placement:
    //  used in function (UsedCSRegs)
    //  used in a basic block (CSRUsed)
    //  anticipatable in a basic block (Antic{In,Out})
    //  available in a basic block (Avail{In,Out})
    //  to be spilled at the entry to a basic block (CSRSave)
    //  to be restored at the end of a basic block (CSRRestore)

    CSRegSet UsedCSRegs;
    CSRegBlockMap CSRUsed;
    CSRegBlockMap AnticIn, AnticOut;
    CSRegBlockMap AvailIn, AvailOut;
    CSRegBlockMap CSRSave;
    CSRegBlockMap CSRRestore;

    // Entry and return blocks of the current function.
    MachineBasicBlock* EntryBlock;
    SmallVector<MachineBasicBlock*, 4> ReturnBlocks;

    // Flag to control shrink wrapping per-function:
    // may choose to skip shrink wrapping for certain
    // functions.
    bool ShrinkWrapThisFunction;

    bool calculateSets(MachineFunction &Fn);
    void calculateAnticAvail(MachineFunction &Fn);
    MachineBasicBlock* moveSpillsOutOfLoops(MachineFunction &Fn,
                                            MachineBasicBlock* MBB);
    void addRestoresForSBranchBlock(MachineFunction &Fn,
                                    MachineBasicBlock* MBB);
    void moveRestoresOutOfLoops(MachineFunction& Fn,
                                MachineBasicBlock* MBB,
                                std::vector<MachineBasicBlock*>& SBLKS);
    void addSavesForRJoinBlocks(MachineFunction& Fn,
                                std::vector<MachineBasicBlock*>& SBLKS);
    void placeSpillsAndRestores(MachineFunction &Fn);
    void placeCSRSpillsAndRestores(MachineFunction &Fn);
    void calculateCalleeSavedRegisters(MachineFunction &Fn);
    void insertCSRSpillsAndRestores(MachineFunction &Fn);
    void calculateFrameObjectOffsets(MachineFunction &Fn);
    void replaceFrameIndices(MachineFunction &Fn);
    void insertPrologEpilogCode(MachineFunction &Fn);

    // Initialize all shrink wrapping data.
    void initShrinkWrappingInfo() {
      UsedCSRegs.clear();
      CSRUsed.clear();
      AnticIn.clear();
      AnticOut.clear();
      AvailIn.clear();
      AvailOut.clear();
      CSRSave.clear();
      CSRRestore.clear();
      EntryBlock = 0;
      if (! ReturnBlocks.empty())
        ReturnBlocks.clear();
      ShrinkWrapThisFunction = ShrinkWrapping;
    }

    // Convienences for dealing with machine loops.
    MachineBasicBlock* getTopLevelLoopPreheader(MachineLoop* LP) {
      assert(LP && "Machine loop is NULL.");
      MachineBasicBlock* PHDR = LP->getLoopPreheader();
      MachineLoop* PLP = LP->getParentLoop();
      while (PLP) {
        PHDR = PLP->getLoopPreheader();
        PLP = PLP->getParentLoop();
      }
      return PHDR;
    }

    MachineLoop* getTopLevelLoopParent(MachineLoop *LP) {
      if (LP == 0)
        return 0;
      MachineLoop* PLP = LP->getParentLoop();
      while (PLP) {
        LP = PLP;
        PLP = PLP->getParentLoop();
      }
      return LP;
    }

#ifndef NDEBUG
    // Debugging methods.
    static std::string getBasicBlockName(const MachineBasicBlock* MBB) {
      std::ostringstream name;
      if (MBB) {
        if (MBB->getBasicBlock())
          name << MBB->getBasicBlock()->getName();
        else
          name << "_MBB_" << MBB->getNumber();
      }
      return name.str();
    }

    static std::string stringifyCSRegSet(const CSRegSet& s,
                                         MachineFunction &Fn) {
      const TargetRegisterInfo* TRI = Fn.getTarget().getRegisterInfo();
      const std::vector<CalleeSavedInfo> CSI =
        Fn.getFrameInfo()->getCalleeSavedInfo();

      std::ostringstream srep;
      if (CSI.size() == 0) {
        srep << "[]";
        return srep.str();
      }
      srep << "[";
      CSRegSet::iterator I = s.begin(), E = s.end();
      if (I != E) {
        unsigned reg = CSI[*I].getReg();
        srep << TRI->getName(reg);
        for (++I; I != E; ++I) {
          reg = CSI[*I].getReg();
          srep << ",";
          srep << TRI->getName(reg);
        }
      }
      srep << "]";
      return srep.str();
    }

    static void dumpSet(const CSRegSet& s, MachineFunction &Fn) {
      DOUT << stringifyCSRegSet(s, Fn) << "\n";
    }
#endif

  };
  char PEI::ID = 0;
}

/// createPrologEpilogCodeInserter - This function returns a pass that inserts
/// prolog and epilog code, and eliminates abstract frame references.
///
FunctionPass *llvm::createPrologEpilogCodeInserter() { return new PEI(); }


/// placeCSRSpillsAndRestores - determine which MBBs of the function
/// need save, restore code for callee-saved registers by doing a DF analysis
/// similar to the one used in code motion (GVNPRE). This produces maps of MBBs
/// to sets of registers (CSRs) for saves and restores. MachineLoopInfo
/// is used to ensure that CSR save/restore code is not placed inside loops.
/// This function computes the maps of MBBs -> CSRs to spill and restore
/// in CSRSave, CSRRestore.
///
/// If shrink wrapping is not being performed, place all spills in
/// the entry block, all restores in return blocks. In this case,
/// CSRSave has a single mapping, CSRRestore has mappings for each
/// return block.
///
void PEI::placeCSRSpillsAndRestores(MachineFunction &Fn) {

#ifndef NDEBUG
  DOUT << "Place CSR spills/restores for "
       << Fn.getFunction()->getName() << "\n";
#endif

  initShrinkWrappingInfo();

  if (calculateSets(Fn))
    placeSpillsAndRestores(Fn);
}

/// calculateAnticAvail - helper for computing the data flow
/// sets required for determining spill/restore placements.
///
void PEI::calculateAnticAvail(MachineFunction &Fn) {

  // Calulate Antic{In,Out} and Avail{In,Out} iteratively on the MCFG.
  bool changed = true;
  unsigned iterations = 0;
  while (changed) {
    changed = false;
    for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
         MBBI != MBBE; ++MBBI) {
      MachineBasicBlock* MBB = MBBI;

      // AnticOut[MBB] = INTERSECT(AnticIn[S] for S in SUCC(MBB))
      MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
        SE = MBB->succ_end();
      if (SI != SE) {
        CSRegSet prevAnticOut = AnticOut[MBB];
        MachineBasicBlock* SUCC = *SI;
        AnticOut[MBB] = AnticIn[SUCC];
        for (++SI; SI != SE; ++SI) {
          SUCC = *SI;
          AnticOut[MBB] &= AnticIn[SUCC];
        }
        if (prevAnticOut != AnticOut[MBB])
          changed = true;
      }
      // AnticIn[MBB] = CSRUsed[MBB] | AnticOut[MBB];
      CSRegSet prevAnticIn = AnticIn[MBB];
      AnticIn[MBB] = CSRUsed[MBB] | AnticOut[MBB];
      if (prevAnticIn |= AnticIn[MBB])
        changed = true;

      // AvailIn[MBB] = INTERSECT(AvailOut[S] for S in PRED(MBB))
      MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
        PE = MBB->pred_end();
      if (PI != PE) {
        CSRegSet prevAvailIn = AvailIn[MBB];
        MachineBasicBlock* PRED = *PI;
        AvailIn[MBB] = AvailOut[PRED];
        for (++PI; PI != PE; ++PI) {
          PRED = *PI;
          AvailIn[MBB] &= AvailOut[PRED];
        }
        if (prevAvailIn != AvailIn[MBB])
          changed = true;
      }
      // AvailOut[MBB] = CSRUsed[MBB] | AvailIn[MBB];
      CSRegSet prevAvailOut = AvailOut[MBB];
      AvailOut[MBB] = CSRUsed[MBB] | AvailIn[MBB];
      if (prevAvailOut |= AvailOut[MBB])
        changed = true;
    }
    ++iterations;
  }

  // EXP
  AnticIn[EntryBlock].clear();
  AnticOut[EntryBlock].clear();

#ifndef NDEBUG
  DOUT << "-----------------------------------------------------------\n";
  DOUT << "iterations = " << iterations << "\n";
  DOUT << "-----------------------------------------------------------\n";
  DOUT << "MBB | ANTIC_IN | ANTIC_OUT | AVAIL_IN | AVAIL_OUT\n";
  DOUT << "-----------------------------------------------------------\n";
  for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
       MBBI != MBBE; ++MBBI) {
    MachineBasicBlock* MBB = MBBI;

    DOUT << getBasicBlockName(MBB) << " | "
         << stringifyCSRegSet(AnticIn[MBB], Fn)
         << " | "
         << stringifyCSRegSet(AnticOut[MBB], Fn)
         << " | "
         << stringifyCSRegSet(AvailIn[MBB], Fn)
         << " | "
         << stringifyCSRegSet(AvailOut[MBB], Fn)
         << "\n";
  }
#endif
}

/// calculateSets - helper function for placeCSRSpillsAndRestores,
/// collect the CSRs used in this function, develop the DF sets that
/// describe the minimal regions in the Machine CFG around which spills,
/// restores must be placed.
///
/// This function decides if shrink wrapping should actually be done:
///   if all CSR uses are in the entry block, no shrink wrapping is possible,
///   so ShrinkWrapping is turned off (for the current function) and the
///   function returns false.
///
bool PEI::calculateSets(MachineFunction &Fn) {

  // Sets used to compute spill, restore placement sets.
  const std::vector<CalleeSavedInfo> CSI =
    Fn.getFrameInfo()->getCalleeSavedInfo();

  // If no CSRs used, we are done.
  if (CSI.empty()) {
#ifndef NDEBUG
    DOUT << Fn.getFunction()->getName()
         << " uses no callee-saved registers.\n";
#endif
    return false;
  }

#ifndef NDEBUG
  DOUT << "-----------------------------------------------------------\n";
#endif

  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();
  bool allCSRUsesInEntryBlock = true;

  // Initialize UsedCSRegs set, CSRUsed map.
  // At the same time, put entry block directly into
  // CSRSave, CSRRestore sets if any CSRs are used.
  //
  // Quick exit option (not implemented):
  //   Given N CSR uses in entry block,
  //   revert to default behavior, skip the placement
  //   step and put all saves in entry, restores in
  //   return blocks.

  // Set up entry and return blocks.
  EntryBlock = Fn.begin();
  for (MachineFunction::iterator MBB = Fn.begin(), E = Fn.end();
       MBB != E; ++MBB)
    if (!MBB->empty() && MBB->back().getDesc().isReturn())
      ReturnBlocks.push_back(MBB);

  // TODO -- check for a use of a CSR in each imm. successor of EntryBlock,
  // do not shrink wrap this function if this is the case.

  // If not shrink wrapping (this function) at this point, set bits in
  // CSR{Save,Restore}[] and UsedCSRegs, then return.
  if (! ShrinkWrapThisFunction) {
    for (unsigned inx = 0, e = CSI.size(); inx != e; ++inx) {
      UsedCSRegs.set(inx);
      CSRSave[EntryBlock].set(inx);
      for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
        CSRRestore[ReturnBlocks[ri]].set(inx);
    }
    return false;
  }

  // Walk instructions in all MBBs, create basic sets, choose
  // whether or not to shrink wrap this function.
  for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
       MBBI != MBBE; ++MBBI) {
    MachineBasicBlock* MBB = MBBI;
    for (MachineBasicBlock::iterator I = MBB->begin(); I != MBB->end(); ++I) {
      for (unsigned inx = 0, e = CSI.size(); inx != e; ++inx) {
        unsigned Reg = CSI[inx].getReg();
        // If instruction I reads or modifies Reg, add it to UsedCSRegs,
        // CSRUsed map for the current block.
        for (unsigned opInx = 0, opEnd = I->getNumOperands();
             opInx != opEnd; ++opInx) {
          const MachineOperand &MO = I->getOperand(opInx);
          if (! (MO.isReg() && (MO.isUse() || MO.isDef())))
            continue;
          unsigned MOReg = MO.getReg();
          if (!MOReg)
            continue;
          if (MOReg == Reg ||
              (TargetRegisterInfo::isPhysicalRegister(MOReg) &&
               TargetRegisterInfo::isPhysicalRegister(Reg) &&
               TRI->isSubRegister(MOReg, Reg))) {
            // CSR Reg is defined/used in block MBB.
            UsedCSRegs.set(inx);
            CSRUsed[MBB].set(inx);
            // Short-circuit analysis for entry, return blocks:
            // if a CSR is used in the entry block, add it directly
            // to CSRSave[EntryBlock] and to CSRRestore[R] for R
            // in ReturnBlocks. Note CSR uses in non-entry blocks.
            if (ShrinkWrapThisFunction) {
              if (MBB == EntryBlock) {
                CSRSave[MBB].set(inx);
                for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
                  CSRRestore[ReturnBlocks[ri]].set(inx);
              } else
                allCSRUsesInEntryBlock = false;
            } else {
              // Not shrink wrapping => ensure saves/restores are correctly
              // added for entry, return blocks.
              CSRSave[EntryBlock].set(inx);
              for (unsigned ri = 0, re = ReturnBlocks.size(); ri != re; ++ri)
                CSRRestore[ReturnBlocks[ri]].set(inx);
            }
          }
        }
      }
    }
#ifndef NDEBUG
    DOUT << "CSRUsed[" << getBasicBlockName(MBB) << "] = "
         << stringifyCSRegSet(CSRUsed[MBB], Fn) << "\n";
#endif
  }

#ifndef NDEBUG
  DOUT << "UsedCSRegs = " << stringifyCSRegSet(UsedCSRegs, Fn) << "\n";
#endif

  // Early exit:
  // 1. Not asked to do shrink wrapping => just "place" all spills(restores)
  //    in the entry(return) block(s), already done above.
  // 2. All CSR uses in entry block => same as case 1, but say we will
  //    not shrink wrap the current function.
  ShrinkWrapThisFunction = (ShrinkWrapping &&
                            ShrinkWrapThisFunction &&
                            ! allCSRUsesInEntryBlock);
  if (! ShrinkWrapThisFunction) {
    return false;
  }

  calculateAnticAvail(Fn);

  return true;
}

/// moveSpillsOutOfLoops - helper for placeSpillsAndRestores() which
/// relocates a spill from a subgraph in a loop to the loop preheader.
/// Returns the MBB to which saves have been moved, or the given MBB
/// if it is a branch point.
///
MachineBasicBlock* PEI::moveSpillsOutOfLoops(MachineFunction &Fn,
                                             MachineBasicBlock* MBB) {
  if (MBB == 0 || CSRSave[MBB].empty())
    return 0;

  // Block to which saves are moved.
  MachineBasicBlock* DEST = 0;
  MachineLoopInfo &LI = getAnalysis<MachineLoopInfo>();

  if (MachineLoop* LP = LI.getLoopFor(MBB)) {
    MachineBasicBlock* LPH = getTopLevelLoopPreheader(LP);
    assert(LPH && "Loop has no top level preheader?");

#ifndef NDEBUG
    DOUT << "Moving saves of "
         << stringifyCSRegSet(CSRSave[MBB], Fn)
         << " from " << getBasicBlockName(MBB)
         << " to " << getBasicBlockName(LPH) << "\n";
#endif
    // Add CSRegSet from MBB to LPH, empty out MBB's CSRegSet.
    CSRSave[LPH] |= CSRSave[MBB];
    // If saves moved to entry block, add restores to returns.
    if (LPH == EntryBlock) {
      for (unsigned i = 0, e = ReturnBlocks.size(); i != e; ++i)
        CSRRestore[ReturnBlocks[i]] |= CSRSave[MBB];
    } else {
      // Remember where we moved the save so we can add
      // restores on successor paths if necessary.
      if (LPH->succ_size() > 1)
        DEST = LPH;
    }
    CSRSave[MBB].clear();
  } else if (MBB->succ_size() > 1)
    DEST = MBB;
  return DEST;
}

/// addRestoresForSBranchBlock - helper for placeSpillsAndRestores() which
/// adds restores of CSRs saved in branch point MBBs to the front of any
/// successor blocks connected to regions with no uses of the saved CSRs.
///
void PEI::addRestoresForSBranchBlock(MachineFunction &Fn,
                                     MachineBasicBlock* MBB) {

  if (MBB == 0 || CSRSave[MBB].empty() || MBB->succ_size() < 2)
    return;

  // Add restores of CSRs saved in branch point MBBs to the
  // front of any succ blocks flowing into regions that
  // have no uses of MBB's CSRs.
  bool hasCSRUses = false;
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI) {
    MachineBasicBlock* SUCC = *SI;
    bool needsRestore = false;
    if (CSRUsed[SUCC].intersects(CSRSave[MBB])) {
      hasCSRUses = true;
      continue;
    }
    needsRestore = true;
    for (df_iterator<MachineBasicBlock*> BI = df_begin(SUCC),
           BE = df_end(SUCC); BI != BE; ++BI) {
      MachineBasicBlock* SBB = *BI;
      if (CSRUsed[SBB].intersects(CSRSave[MBB])) {
        hasCSRUses = true;
        needsRestore = false;
        break;
      }
    }
    // Additional restores are needed for SUCC iff there is at least
    // one CSR use reachable from the successors of MBB and there
    // are no uses in or below SUCC.
    if (needsRestore && hasCSRUses) {
#ifndef NDEBUG
      DOUT << "MBB " << getBasicBlockName(MBB)
           << " needs a restore on path to successor "
           << getBasicBlockName(SUCC) << "\n";
#endif
      // Add restores to SUCC for all CSRs saved in MBB...
      CSRRestore[SUCC] = CSRSave[MBB];
    }
  }
}

/// moveRestoresOutOfLoops - helper for placeSpillsAndRestores() which
/// relocates restores from a subgraph in a loop to the loop exit blocks.
/// This function records the MBBs to which restores have been moved in
/// SBLKS. If no restores are moved, SBLKS contains the input MBB if it
/// is a join point in the Machine CFG.
///
void PEI::moveRestoresOutOfLoops(MachineFunction& Fn,
                                 MachineBasicBlock* MBB,
                                 std::vector<MachineBasicBlock*>& SBLKS) {

  SBLKS.clear();
  if (MBB == 0 || CSRRestore[MBB].empty())
    return;

  MachineLoopInfo &LI = getAnalysis<MachineLoopInfo>();

  if (MachineLoop* LP = LI.getLoopFor(MBB)) {
    LP = getTopLevelLoopParent(LP);
    assert(LP && "Loop with no top level parent?");

    SmallVector<MachineBasicBlock*, 4> exitBlocks;

    LP->getExitBlocks(exitBlocks);
    assert(exitBlocks.size() > 0 &&
           "Loop has no top level exit blocks?");
    for (unsigned i = 0, e = exitBlocks.size(); i != e; ++i) {
      MachineBasicBlock* EXB = exitBlocks[i];

#ifndef NDEBUG
      DOUT << "Moving restores of "
           << stringifyCSRegSet(CSRRestore[MBB], Fn)
           << " from " << getBasicBlockName(MBB)
           << " to " << getBasicBlockName(EXB) << "\n";
#endif

      // Add CSRegSet from MBB to LPE, empty out MBB's CSRegSet.
      CSRRestore[EXB] |= CSRRestore[MBB];
      if (EXB->pred_size() > 1)
        SBLKS.push_back(EXB);
    }
    CSRRestore[MBB].clear();
  } else if (MBB->pred_size() > 1)
    SBLKS.push_back(MBB);
}

/// addSavesForRJoinBlocks - Add saves of CSRs restored in join point MBBs
/// to the ends of any pred blocks that flow into MBB from regions that
/// have no uses of MBB's CSRs.
///
void PEI::addSavesForRJoinBlocks(MachineFunction& Fn,
                                 std::vector<MachineBasicBlock*>& SBLKS) {

  if (SBLKS.empty())
    return;

  for (unsigned i = 0, e = SBLKS.size(); i != e; ++i) {
    MachineBasicBlock* MBB = SBLKS[i];
    if (MBB->pred_size() > 1) {
      bool needsSave = false;
      for (MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
             PE = MBB->pred_end(); PI != PE; ++PI) {
        MachineBasicBlock* PRED = *PI;

        // Walk back up in the CFG from the preds of MBB, look for
        // a block that uses any CSR that is restored in MBB.
        if (CSRUsed[PRED].intersects(CSRRestore[MBB]))
          continue;
        needsSave = true;
        for (idf_iterator<MachineBasicBlock*> PPI = idf_begin(PRED),
               PPE = idf_end(PRED); PPI != PPE; ++PPI) {
          MachineBasicBlock* PBB = *PPI;
          if (CSRUsed[PBB].intersects(CSRRestore[MBB])) {
            needsSave = false;
            break;
          }
        }
        if (needsSave) {
          // Add saves to PRED for all CSRs restored in MBB...
#ifndef NDEBUG
          DOUT << "MBB " << getBasicBlockName(MBB)
               << " needs a save on path from predecessor "
               << getBasicBlockName(PRED) << "\n";
#endif
          CSRSave[PRED] = CSRRestore[MBB];
        }
      }
    }
  }
}

/// placeSpillsAndRestores - decide which MBBs need spills, restores
/// of CSRs.
///
void PEI::placeSpillsAndRestores(MachineFunction &Fn) {

#ifndef NDEBUG
  DOUT << "-----------------------------------------------------------\n";
#endif

  // Calculate CSR{Save,Restore} using Antic, Avail on the Machine-CFG.
  for (MachineFunction::iterator MBBI = Fn.begin(), MBBE = Fn.end();
       MBBI != MBBE; ++MBBI) {
    MachineBasicBlock* MBB = MBBI;
    // Entry block saves are recorded in UsedCSRegs pass above.
    if (MBB != EntryBlock) {
      // Intersect (CSRegs - AnticIn[P]) for all predecessors P of MBB
      CSRegSet anticInPreds;
      MachineBasicBlock::pred_iterator PI = MBB->pred_begin(),
        PE = MBB->pred_end();
      if (PI != PE) {
        MachineBasicBlock* PRED = *PI;
        anticInPreds = UsedCSRegs - AnticIn[PRED];
        for (++PI; PI != PE; ++PI) {
          PRED = *PI;
          // Handle self loop.
          if (PRED != MBB)
            anticInPreds &= (UsedCSRegs - AnticIn[PRED]);
        }
      }
      // CSRSave[MBB] = (AnticIn[MBB] - AvailIn[MBB]) & anticInPreds
      CSRSave[MBB] = (AnticIn[MBB] - AvailIn[MBB]) & anticInPreds;

      // Remove the CSRs that are saved in the entry block
      if (! CSRSave[MBB].empty() && ! CSRSave[EntryBlock].empty())
        CSRSave[MBB] = CSRSave[MBB] - CSRSave[EntryBlock];

      // Move saves inside loops to the preheaders of the outermost
      // containing loops, add restores to blocks reached by saves
      // placed at branch points where necessary.
      if (MachineBasicBlock* DESTBB = moveSpillsOutOfLoops(Fn, MBB)) {
        // Add restores to blocks reached by saves placed at branch
        // points where necessary.
        addRestoresForSBranchBlock(Fn, DESTBB);
      }
    }

#ifndef NDEBUG
    if (! CSRSave[MBB].empty())
      DOUT << "SAVE[" << getBasicBlockName(MBB) << "] = "
           << stringifyCSRegSet(CSRSave[MBB], Fn) << "\n";
#endif

    // Compute CSRRestore, which may already be set for return blocks.
    if (! CSRRestore[MBB].empty() || MBB->pred_size() == 0)
      continue;

    // Intersect (CSRegs - AvailOut[S]) for all successors S of MBB
    CSRegSet availOutSucc;
    MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
      SE = MBB->succ_end();
    if (SI != SE) {
      MachineBasicBlock* SUCC = *SI;
      availOutSucc = UsedCSRegs - AvailOut[SUCC];
      for (++SI; SI != SE; ++SI) {
        SUCC = *SI;
        // Handle self loop.
        if (SUCC != MBB)
          availOutSucc &= (UsedCSRegs - AvailOut[SUCC]);
      }
    } else if (! CSRUsed[MBB].empty()) {
      // Take care of uses in return blocks (which have no successors).
      availOutSucc = UsedCSRegs;
    }
    // CSRRestore[MBB] = (AvailOut[MBB] - AnticOut[MBB]) & availOutSucc
    CSRRestore[MBB] = (AvailOut[MBB] - AnticOut[MBB]) & availOutSucc;

    // Remove the CSRs that are restored in the return blocks.
    // Lest this be confusing, note that:
    // CSRSave[EntryBlock] == CSRRestore[B] for all B in ReturnBlocks.
    if (! CSRRestore[MBB].empty() && ! CSRSave[EntryBlock].empty())
      CSRRestore[MBB] = CSRRestore[MBB] - CSRSave[EntryBlock];

    // Move restores inside loops to the exits of the outermost (top level)
    // containing loops.
    std::vector<MachineBasicBlock*> saveBlocks;
    moveRestoresOutOfLoops(Fn, MBB, saveBlocks);

    // Add saves of CSRs restored in join point MBBs to the ends
    // of any pred blocks that flow into MBB from regions that
    // have no uses of MBB's CSRs.
    addSavesForRJoinBlocks(Fn, saveBlocks);

#ifndef NDEBUG
    if (! CSRRestore[MBB].empty())
      DOUT << "RESTORE[" << getBasicBlockName(MBB) << "] = "
           << stringifyCSRegSet(CSRRestore[MBB], Fn) << "\n";
#endif
  }

#ifndef NDEBUG
  DOUT << "-----------------------------------------------------------\n";
  DOUT << "Final SAVE, RESTORE:\n";
  DOUT << "-----------------------------------------------------------\n";
  for (MachineFunction::iterator MBB = Fn.begin(), E = Fn.end();
       MBB != E; ++MBB) {
    if (! CSRSave[MBB].empty()) {
      DOUT << "SAVE[" << getBasicBlockName(MBB) << "] = "
           << stringifyCSRegSet(CSRSave[MBB], Fn);
      if (CSRRestore[MBB].empty())
        DOUT << "\n";
    }
    if (! CSRRestore[MBB].empty()) {
      if (! CSRSave[MBB].empty())
        DOUT << "    ";
      DOUT << "RESTORE[" << getBasicBlockName(MBB) << "] = "
           << stringifyCSRegSet(CSRRestore[MBB], Fn) << "\n";
    }
  }
#endif
}

/// calculateCalleeSavedRegisters - Scan the function for modified callee saved
/// registers.  Also calculate the MaxCallFrameSize and HasCalls variables for
/// the function's frame information and eliminates call frame pseudo
/// instructions.
///
void PEI::calculateCalleeSavedRegisters(MachineFunction &Fn) {
  const TargetRegisterInfo *RegInfo = Fn.getTarget().getRegisterInfo();
  const TargetFrameInfo *TFI = Fn.getTarget().getFrameInfo();

  // Get the callee saved register list...
  const unsigned *CSRegs = RegInfo->getCalleeSavedRegs(&Fn);

  // Get the function call frame set-up and tear-down instruction opcode
  int FrameSetupOpcode   = RegInfo->getCallFrameSetupOpcode();
  int FrameDestroyOpcode = RegInfo->getCallFrameDestroyOpcode();

  // These are used to keep track the callee-save area. Initialize them.
  MinCSFrameIndex = INT_MAX;
  MaxCSFrameIndex = 0;

  // Early exit for targets which have no callee saved registers and no call
  // frame setup/destroy pseudo instructions.
  if ((CSRegs == 0 || CSRegs[0] == 0) &&
      FrameSetupOpcode == -1 && FrameDestroyOpcode == -1)
    return;

  unsigned MaxCallFrameSize = 0;
  bool HasCalls = false;

  std::vector<MachineBasicBlock::iterator> FrameSDOps;
  for (MachineFunction::iterator BB = Fn.begin(), E = Fn.end(); BB != E; ++BB)
    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ++I)
      if (I->getOpcode() == FrameSetupOpcode ||
          I->getOpcode() == FrameDestroyOpcode) {
        assert(I->getNumOperands() >= 1 && "Call Frame Setup/Destroy Pseudo"
               " instructions should have a single immediate argument!");
        unsigned Size = I->getOperand(0).getImm();
        if (Size > MaxCallFrameSize) MaxCallFrameSize = Size;
        HasCalls = true;
        FrameSDOps.push_back(I);
      }

  MachineFrameInfo *FFI = Fn.getFrameInfo();
  FFI->setHasCalls(HasCalls);
  FFI->setMaxCallFrameSize(MaxCallFrameSize);

  for (unsigned i = 0, e = FrameSDOps.size(); i != e; ++i) {
    MachineBasicBlock::iterator I = FrameSDOps[i];
    // If call frames are not being included as part of the stack frame,
    // and there is no dynamic allocation (therefore referencing frame slots
    // off sp), leave the pseudo ops alone. We'll eliminate them later.
    if (RegInfo->hasReservedCallFrame(Fn) || RegInfo->hasFP(Fn))
      RegInfo->eliminateCallFramePseudoInstr(Fn, *I->getParent(), I);
  }

  // Now figure out which *callee saved* registers are modified by the current
  // function, thus needing to be saved and restored in the prolog/epilog.
  //
  const TargetRegisterClass* const *CSRegClasses =
    RegInfo->getCalleeSavedRegClasses(&Fn);
  std::vector<CalleeSavedInfo> CSI;
  for (unsigned i = 0; CSRegs[i]; ++i) {
    unsigned Reg = CSRegs[i];
    if (Fn.getRegInfo().isPhysRegUsed(Reg)) {
        // If the reg is modified, save it!
      CSI.push_back(CalleeSavedInfo(Reg, CSRegClasses[i]));
    } else {
      for (const unsigned *AliasSet = RegInfo->getAliasSet(Reg);
           *AliasSet; ++AliasSet) {  // Check alias registers too.
        if (Fn.getRegInfo().isPhysRegUsed(*AliasSet)) {
          CSI.push_back(CalleeSavedInfo(Reg, CSRegClasses[i]));
          break;
        }
      }
    }
  }

  if (CSI.empty())
    return;   // Early exit if no callee saved registers are modified!

  unsigned NumFixedSpillSlots;
  const std::pair<unsigned,int> *FixedSpillSlots =
    TFI->getCalleeSavedSpillSlots(NumFixedSpillSlots);

  // Now that we know which registers need to be saved and restored, allocate
  // stack slots for them.
  for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
    unsigned Reg = CSI[i].getReg();
    const TargetRegisterClass *RC = CSI[i].getRegClass();

    // Check to see if this physreg must be spilled to a particular stack slot
    // on this target.
    const std::pair<unsigned,int> *FixedSlot = FixedSpillSlots;
    while (FixedSlot != FixedSpillSlots+NumFixedSpillSlots &&
           FixedSlot->first != Reg)
      ++FixedSlot;

    int FrameIdx;
    if (FixedSlot == FixedSpillSlots+NumFixedSpillSlots) {
      // Nope, just spill it anywhere convenient.
      unsigned Align = RC->getAlignment();
      unsigned StackAlign = TFI->getStackAlignment();
      // We may not be able to sastify the desired alignment specification of
      // the TargetRegisterClass if the stack alignment is smaller.
      // Use the min.
      Align = std::min(Align, StackAlign);
      FrameIdx = FFI->CreateStackObject(RC->getSize(), Align);
      if ((unsigned)FrameIdx < MinCSFrameIndex) MinCSFrameIndex = FrameIdx;
      if ((unsigned)FrameIdx > MaxCSFrameIndex) MaxCSFrameIndex = FrameIdx;
    } else {
      // Spill it to the stack where we must.
      FrameIdx = FFI->CreateFixedObject(RC->getSize(), FixedSlot->second);
    }
    CSI[i].setFrameIdx(FrameIdx);
  }

  FFI->setCalleeSavedInfo(CSI);
}

/// insertCSRSpillsAndRestores - Insert spill and restore code for
/// callee saved registers used in the function, handling shrink wrapping.
///
void PEI::insertCSRSpillsAndRestores(MachineFunction &Fn) {
  // Get callee saved register information.
  MachineFrameInfo *FFI = Fn.getFrameInfo();
  const std::vector<CalleeSavedInfo> &CSI = FFI->getCalleeSavedInfo();

  // Early exit if no callee saved registers are modified!
  if (CSI.empty())
    return;

  const TargetInstrInfo &TII = *Fn.getTarget().getInstrInfo();
  MachineBasicBlock::iterator I;
  std::vector<CalleeSavedInfo> blockCSI;

#ifndef NDEBUG
  DOUT << "Inserting spill/restore code for CSRs in function "
       << Fn.getFunction()->getName() << "\n";
#endif

  // Insert spills.
  for (CSRegBlockMap::iterator
         BI = CSRSave.begin(), BE = CSRSave.end(); BI != BE; ++BI) {
    MachineBasicBlock* MBB = BI->first;
    CSRegSet save = BI->second;

    if (save.empty())
      continue;

    if (! ShrinkWrapThisFunction) {
      // Spill using target interface.
      I = MBB->begin();
      if (!TII.spillCalleeSavedRegisters(*MBB, I, CSI)) {
        for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
          // Add the callee-saved register as live-in. It's killed at the spill.
          MBB->addLiveIn(CSI[i].getReg());

          // Insert the spill to the stack frame.
          TII.storeRegToStackSlot(*MBB, I, CSI[i].getReg(), true,
                                  CSI[i].getFrameIdx(), CSI[i].getRegClass());
        }
      }
    } else {
#ifndef NDEBUG
      DOUT << "CSRSave[" << getBasicBlockName(MBB) << "] = "
           << stringifyCSRegSet(save, Fn) << "\n";
#endif

      blockCSI.clear();
      for (CSRegSet::iterator RI = save.begin(),
             RE = save.end(); RI != RE; ++RI) {
        blockCSI.push_back(CSI[*RI]);
      }
      assert(blockCSI.size() > 0 &&
             "Could not collect callee saved register info");

      // If MBB has no uses of CSRs being saved, this means saves
      // must be inserted at the _end_.
      if (! MBB->empty() && ! CSRUsed[MBB].intersects(save)) {
        I = MBB->end();
        --I;
        if (I->getDesc().isCall()) {
          ++I;
        } else {
          MachineBasicBlock::iterator I2 = I;
          while (I2 != MBB->begin() && (--I2)->getDesc().isTerminator())
            I = I2;
        }
      } else {
        I = MBB->begin();
      }

      // When shrink wrapping, use stack slot stores/loads.
      for (unsigned i = 0, e = blockCSI.size(); i != e; ++i) {
        // Add the callee-saved register as live-in.
        // It's killed at the spill.
        MBB->addLiveIn(blockCSI[i].getReg());

        // Insert the spill to the stack frame.
        TII.storeRegToStackSlot(*MBB, I, blockCSI[i].getReg(),
                                true,
                                blockCSI[i].getFrameIdx(),
                                blockCSI[i].getRegClass());
      }
    }
  }
  // Use CSRRestore to add code to restore the callee-saved registers in
  // each block.
  for (CSRegBlockMap::iterator
         BI = CSRRestore.begin(), BE = CSRRestore.end(); BI != BE; ++BI) {
    MachineBasicBlock* MBB = BI->first;
    CSRegSet restore = BI->second;

    if (restore.empty())
      continue;
    if (! ShrinkWrapThisFunction) {
      // Restore using target interface.
      I = MBB->end(); --I;

      // Skip over all terminator instructions, which are part of the return
      // sequence.
      MachineBasicBlock::iterator I2 = I;
      while (I2 != MBB->begin() && (--I2)->getDesc().isTerminator())
        I = I2;

      bool AtStart = I == MBB->begin();
      MachineBasicBlock::iterator BeforeI = I;
      if (!AtStart)
        --BeforeI;

      // Restore all registers immediately before the return and any
      // terminators that preceed it.
      if (!TII.restoreCalleeSavedRegisters(*MBB, I, CSI)) {
        for (unsigned i = 0, e = CSI.size(); i != e; ++i) {
          TII.loadRegFromStackSlot(*MBB, I, CSI[i].getReg(),
                                   CSI[i].getFrameIdx(),
                                   CSI[i].getRegClass());
          assert(I != MBB->begin() &&
                 "loadRegFromStackSlot didn't insert any code!");
          // Insert in reverse order.  loadRegFromStackSlot can insert
          // multiple instructions.
          if (AtStart)
            I = MBB->begin();
          else {
            I = BeforeI;
            ++I;
          }
        }
      }
    } else {
#ifndef NDEBUG
      DOUT << "CSRRestore[" << getBasicBlockName(MBB) << "] = "
           << stringifyCSRegSet(restore, Fn) << "\n";
#endif

      blockCSI.clear();
      for (CSRegSet::iterator RI = restore.begin(),
             RE = restore.end(); RI != RE; ++RI) {
        blockCSI.push_back(CSI[*RI]);
      }
      assert(blockCSI.size() > 0 &&
             "Could not find callee saved register info");

      // If MBB uses no CSRs but has restores, this means
      // it must have restores inserted at the _beginning_.
      // N.B. -- not necessary if edge splitting done.
      if (MBB->empty() || ! CSRUsed[MBB].intersects(restore)) {
        I = MBB->begin();
      } else {
        I = MBB->end();
        --I;

        // EXP iff spill/restore implemented with push/pop:
        // append restore to block unless it ends in a
        // barrier terminator instruction.

        // Skip over all terminator instructions, which are part of the
        // return sequence.
        if (I->getDesc().isCall()) {
          ++I;
        } else {
          MachineBasicBlock::iterator I2 = I;
          while (I2 != MBB->begin() && (--I2)->getDesc().isTerminator())
            I = I2;
        }
      }

      bool AtStart = I == MBB->begin();
      MachineBasicBlock::iterator BeforeI = I;
      if (!AtStart)
        --BeforeI;

#ifndef NDEBUG
      if (! MBB->empty() && ! CSRUsed[MBB].intersects(restore)) {
        MachineInstr* MI = BeforeI;
        DOUT << "adding restore after ";
        DEBUG(MI->dump());
      } else {
        DOUT << "adding restore to beginning of "
             << getBasicBlockName(MBB) << "\n";
      }
#endif

      // Restore all registers immediately before the return and any
      // terminators that preceed it.
      for (unsigned i = 0, e = blockCSI.size(); i != e; ++i) {
        TII.loadRegFromStackSlot(*MBB, I, blockCSI[i].getReg(),
                                 blockCSI[i].getFrameIdx(),
                                 blockCSI[i].getRegClass());
        assert(I != MBB->begin() &&
               "loadRegFromStackSlot didn't insert any code!");
        // Insert in reverse order.  loadRegFromStackSlot can insert
        // multiple instructions.
        if (AtStart)
          I = MBB->begin();
        else {
          I = BeforeI;
          ++I;
        }
      }
    }
  }
}

/// AdjustStackOffset - Helper function used to adjust the stack frame offset.
static inline void
AdjustStackOffset(MachineFrameInfo *FFI, int FrameIdx,
                  bool StackGrowsDown, int64_t &Offset,
                  unsigned &MaxAlign) {
  // If stack grows down, we need to add size of find the lowest address of the
  // object.
  if (StackGrowsDown)
    Offset += FFI->getObjectSize(FrameIdx);

  unsigned Align = FFI->getObjectAlignment(FrameIdx);

  // If the alignment of this object is greater than that of the stack, then
  // increase the stack alignment to match.
  MaxAlign = std::max(MaxAlign, Align);

  // Adjust to alignment boundary.
  Offset = (Offset + Align - 1) / Align * Align;

  if (StackGrowsDown) {
    FFI->setObjectOffset(FrameIdx, -Offset); // Set the computed offset
  } else {
    FFI->setObjectOffset(FrameIdx, Offset);
    Offset += FFI->getObjectSize(FrameIdx);
  }
}

/// calculateFrameObjectOffsets - Calculate actual frame offsets for all of the
/// abstract stack objects.
///
void PEI::calculateFrameObjectOffsets(MachineFunction &Fn) {
  const TargetFrameInfo &TFI = *Fn.getTarget().getFrameInfo();

  bool StackGrowsDown =
    TFI.getStackGrowthDirection() == TargetFrameInfo::StackGrowsDown;

  // Loop over all of the stack objects, assigning sequential addresses...
  MachineFrameInfo *FFI = Fn.getFrameInfo();

  unsigned MaxAlign = FFI->getMaxAlignment();

  // Start at the beginning of the local area.
  // The Offset is the distance from the stack top in the direction
  // of stack growth -- so it's always nonnegative.
  int64_t Offset = TFI.getOffsetOfLocalArea();
  if (StackGrowsDown)
    Offset = -Offset;
  assert(Offset >= 0
         && "Local area offset should be in direction of stack growth");

  // If there are fixed sized objects that are preallocated in the local area,
  // non-fixed objects can't be allocated right at the start of local area.
  // We currently don't support filling in holes in between fixed sized
  // objects, so we adjust 'Offset' to point to the end of last fixed sized
  // preallocated object.
  for (int i = FFI->getObjectIndexBegin(); i != 0; ++i) {
    int64_t FixedOff;
    if (StackGrowsDown) {
      // The maximum distance from the stack pointer is at lower address of
      // the object -- which is given by offset. For down growing stack
      // the offset is negative, so we negate the offset to get the distance.
      FixedOff = -FFI->getObjectOffset(i);
    } else {
      // The maximum distance from the start pointer is at the upper
      // address of the object.
      FixedOff = FFI->getObjectOffset(i) + FFI->getObjectSize(i);
    }
    if (FixedOff > Offset) Offset = FixedOff;
  }

  // First assign frame offsets to stack objects that are used to spill
  // callee saved registers.
  if (StackGrowsDown) {
    for (unsigned i = MinCSFrameIndex; i <= MaxCSFrameIndex; ++i) {
      // If stack grows down, we need to add size of find the lowest
      // address of the object.
      Offset += FFI->getObjectSize(i);

      unsigned Align = FFI->getObjectAlignment(i);
      // If the alignment of this object is greater than that of the stack,
      // then increase the stack alignment to match.
      MaxAlign = std::max(MaxAlign, Align);
      // Adjust to alignment boundary
      Offset = (Offset+Align-1)/Align*Align;

      FFI->setObjectOffset(i, -Offset);        // Set the computed offset
    }
  } else {
    int MaxCSFI = MaxCSFrameIndex, MinCSFI = MinCSFrameIndex;
    for (int i = MaxCSFI; i >= MinCSFI ; --i) {
      unsigned Align = FFI->getObjectAlignment(i);
      // If the alignment of this object is greater than that of the stack,
      // then increase the stack alignment to match.
      MaxAlign = std::max(MaxAlign, Align);
      // Adjust to alignment boundary
      Offset = (Offset+Align-1)/Align*Align;

      FFI->setObjectOffset(i, Offset);
      Offset += FFI->getObjectSize(i);
    }
  }

  // Make sure the special register scavenging spill slot is closest to the
  // frame pointer if a frame pointer is required.
  const TargetRegisterInfo *RegInfo = Fn.getTarget().getRegisterInfo();
  if (RS && RegInfo->hasFP(Fn)) {
    int SFI = RS->getScavengingFrameIndex();
    if (SFI >= 0)
      AdjustStackOffset(FFI, SFI, StackGrowsDown, Offset, MaxAlign);
  }

  // Make sure that the stack protector comes before the local variables on the
  // stack.
  if (FFI->getStackProtectorIndex() >= 0)
    AdjustStackOffset(FFI, FFI->getStackProtectorIndex(), StackGrowsDown,
                      Offset, MaxAlign);

  // Then assign frame offsets to stack objects that are not used to spill
  // callee saved registers.
  for (unsigned i = 0, e = FFI->getObjectIndexEnd(); i != e; ++i) {
    if (i >= MinCSFrameIndex && i <= MaxCSFrameIndex)
      continue;
    if (RS && (int)i == RS->getScavengingFrameIndex())
      continue;
    if (FFI->isDeadObjectIndex(i))
      continue;
    if (FFI->getStackProtectorIndex() == (int)i)
      continue;

    AdjustStackOffset(FFI, i, StackGrowsDown, Offset, MaxAlign);
  }

  // Make sure the special register scavenging spill slot is closest to the
  // stack pointer.
  if (RS && !RegInfo->hasFP(Fn)) {
    int SFI = RS->getScavengingFrameIndex();
    if (SFI >= 0)
      AdjustStackOffset(FFI, SFI, StackGrowsDown, Offset, MaxAlign);
  }

  // Round up the size to a multiple of the alignment, but only if there are
  // calls or alloca's in the function.  This ensures that any calls to
  // subroutines have their stack frames suitable aligned.
  // Also do this if we need runtime alignment of the stack.  In this case
  // offsets will be relative to SP not FP; round up the stack size so this
  // works.
  if (!RegInfo->targetHandlesStackFrameRounding() &&
      (FFI->hasCalls() || FFI->hasVarSizedObjects() ||
       (RegInfo->needsStackRealignment(Fn) &&
        FFI->getObjectIndexEnd() != 0))) {
    // If we have reserved argument space for call sites in the function
    // immediately on entry to the current function, count it as part of the
    // overall stack size.
    if (RegInfo->hasReservedCallFrame(Fn))
      Offset += FFI->getMaxCallFrameSize();

    unsigned AlignMask = std::max(TFI.getStackAlignment(),MaxAlign) - 1;
    Offset = (Offset + AlignMask) & ~uint64_t(AlignMask);
  }

  // Update frame info to pretend that this is part of the stack...
  FFI->setStackSize(Offset+TFI.getOffsetOfLocalArea());

  // Remember the required stack alignment in case targets need it to perform
  // dynamic stack alignment.
  FFI->setMaxAlignment(MaxAlign);
}


/// insertPrologEpilogCode - Scan the function for modified callee saved
/// registers, insert spill code for these callee saved registers, then add
/// prolog and epilog code to the function.
///
void PEI::insertPrologEpilogCode(MachineFunction &Fn) {
  const TargetRegisterInfo *TRI = Fn.getTarget().getRegisterInfo();

  // Add prologue to the function...
  TRI->emitPrologue(Fn);

  // Add epilogue to restore the callee-save registers in each exiting block
  for (MachineFunction::iterator I = Fn.begin(), E = Fn.end(); I != E; ++I) {
    // If last instruction is a return instruction, add an epilogue
    if (!I->empty() && I->back().getDesc().isReturn())
      TRI->emitEpilogue(Fn, *I);
  }
}


/// replaceFrameIndices - Replace all MO_FrameIndex operands with physical
/// register references and actual offsets.
///
void PEI::replaceFrameIndices(MachineFunction &Fn) {
  if (!Fn.getFrameInfo()->hasStackObjects()) return; // Nothing to do?

  const TargetMachine &TM = Fn.getTarget();
  assert(TM.getRegisterInfo() && "TM::getRegisterInfo() must be implemented!");
  const TargetRegisterInfo &TRI = *TM.getRegisterInfo();
  const TargetFrameInfo *TFI = TM.getFrameInfo();
  bool StackGrowsDown =
    TFI->getStackGrowthDirection() == TargetFrameInfo::StackGrowsDown;
  int FrameSetupOpcode   = TRI.getCallFrameSetupOpcode();
  int FrameDestroyOpcode = TRI.getCallFrameDestroyOpcode();

  for (MachineFunction::iterator BB = Fn.begin(),
         E = Fn.end(); BB != E; ++BB) {
    int SPAdj = 0;  // SP offset due to call frame setup / destroy.
    if (RS) RS->enterBasicBlock(BB);

    for (MachineBasicBlock::iterator I = BB->begin(); I != BB->end(); ) {
      if (I->getOpcode() == TargetInstrInfo::DECLARE) {
        // Ignore it.
        ++I;
        continue;
      }

      if (I->getOpcode() == FrameSetupOpcode ||
          I->getOpcode() == FrameDestroyOpcode) {
        // Remember how much SP has been adjusted to create the call
        // frame.
        int Size = I->getOperand(0).getImm();

        if ((!StackGrowsDown && I->getOpcode() == FrameSetupOpcode) ||
            (StackGrowsDown && I->getOpcode() == FrameDestroyOpcode))
          Size = -Size;

        SPAdj += Size;

        MachineBasicBlock::iterator PrevI = BB->end();
        if (I != BB->begin()) PrevI = prior(I);
        TRI.eliminateCallFramePseudoInstr(Fn, *BB, I);

        // Visit the instructions created by eliminateCallFramePseudoInstr().
        if (PrevI == BB->end())
          I = BB->begin();     // The replaced instr was the first in the block.
        else
          I = next(PrevI);
        continue;
      }

      MachineInstr *MI = I;
      bool DoIncr = true;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i)
        if (MI->getOperand(i).isFI()) {
          // Some instructions (e.g. inline asm instructions) can have
          // multiple frame indices and/or cause eliminateFrameIndex
          // to insert more than one instruction. We need the register
          // scavenger to go through all of these instructions so that
          // it can update its register information. We keep the
          // iterator at the point before insertion so that we can
          // revisit them in full.
          bool AtBeginning = (I == BB->begin());
          if (!AtBeginning) --I;

          // If this instruction has a FrameIndex operand, we need to
          // use that target machine register info object to eliminate
          // it.

          TRI.eliminateFrameIndex(MI, SPAdj, RS);

          // Reset the iterator if we were at the beginning of the BB.
          if (AtBeginning) {
            I = BB->begin();
            DoIncr = false;
          }

          MI = 0;
          break;
        }

      if (DoIncr && I != BB->end()) ++I;

      // Update register states.
      if (RS && MI) RS->forward(MI);
    }

    assert(SPAdj == 0 && "Unbalanced call frame setup / destroy pairs?");
  }
}
