//===-- LiveIntervalAnalysis.cpp - Live Interval Analysis -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the LiveInterval analysis pass which is used
// by the Linear Scan Register allocator. This pass linearizes the
// basic blocks of the function in DFS order and uses the
// LiveVariables pass to conservatively compute live intervals for
// each virtual and physical register.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "liveintervals"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "VirtRegMap.h"
#include "llvm/Value.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/CodeGen/LiveVariables.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include <algorithm>
#include <limits>
#include <cmath>
using namespace llvm;

// Hidden options for help debugging.
static cl::opt<bool> DisableReMat("disable-rematerialization", 
                                  cl::init(false), cl::Hidden);

static cl::opt<bool> EnableAggressiveRemat("aggressive-remat", cl::Hidden);

static cl::opt<bool> EnableFastSpilling("fast-spill",
                                        cl::init(false), cl::Hidden);

static cl::opt<bool> EarlyCoalescing("early-coalescing", cl::init(false));

static cl::opt<int> CoalescingLimit("early-coalescing-limit",
                                    cl::init(-1), cl::Hidden);

STATISTIC(numIntervals , "Number of original intervals");
STATISTIC(numFolds     , "Number of loads/stores folded into instructions");
STATISTIC(numSplits    , "Number of intervals split");
STATISTIC(numCoalescing, "Number of early coalescing performed");

char LiveIntervals::ID = 0;
static RegisterPass<LiveIntervals> X("liveintervals", "Live Interval Analysis");

void LiveIntervals::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<AliasAnalysis>();
  AU.addPreserved<AliasAnalysis>();
  AU.addPreserved<LiveVariables>();
  AU.addRequired<LiveVariables>();
  AU.addPreservedID(MachineLoopInfoID);
  AU.addPreservedID(MachineDominatorsID);
  
  if (!StrongPHIElim) {
    AU.addPreservedID(PHIEliminationID);
    AU.addRequiredID(PHIEliminationID);
  }
  
  AU.addRequiredID(TwoAddressInstructionPassID);
  MachineFunctionPass::getAnalysisUsage(AU);
}

void LiveIntervals::releaseMemory() {
  // Free the live intervals themselves.
  for (DenseMap<unsigned, LiveInterval*>::iterator I = r2iMap_.begin(),
       E = r2iMap_.end(); I != E; ++I)
    delete I->second;
  
  MBB2IdxMap.clear();
  Idx2MBBMap.clear();
  mi2iMap_.clear();
  i2miMap_.clear();
  r2iMap_.clear();
  terminatorGaps.clear();
  phiJoinCopies.clear();

  // Release VNInfo memroy regions after all VNInfo objects are dtor'd.
  VNInfoAllocator.Reset();
  while (!CloneMIs.empty()) {
    MachineInstr *MI = CloneMIs.back();
    CloneMIs.pop_back();
    mf_->DeleteMachineInstr(MI);
  }
}

static bool CanTurnIntoImplicitDef(MachineInstr *MI, unsigned Reg,
                                   unsigned OpIdx, const TargetInstrInfo *tii_){
  unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
  if (tii_->isMoveInstr(*MI, SrcReg, DstReg, SrcSubReg, DstSubReg) &&
      Reg == SrcReg)
    return true;

  if (OpIdx == 2 && MI->getOpcode() == TargetInstrInfo::SUBREG_TO_REG)
    return true;
  if (OpIdx == 1 && MI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG)
    return true;
  return false;
}

/// processImplicitDefs - Process IMPLICIT_DEF instructions and make sure
/// there is one implicit_def for each use. Add isUndef marker to
/// implicit_def defs and their uses.
void LiveIntervals::processImplicitDefs() {
  SmallSet<unsigned, 8> ImpDefRegs;
  SmallVector<MachineInstr*, 8> ImpDefMIs;
  MachineBasicBlock *Entry = mf_->begin();
  SmallPtrSet<MachineBasicBlock*,16> Visited;
  for (df_ext_iterator<MachineBasicBlock*, SmallPtrSet<MachineBasicBlock*,16> >
         DFI = df_ext_begin(Entry, Visited), E = df_ext_end(Entry, Visited);
       DFI != E; ++DFI) {
    MachineBasicBlock *MBB = *DFI;
    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ) {
      MachineInstr *MI = &*I;
      ++I;
      if (MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF) {
        unsigned Reg = MI->getOperand(0).getReg();
        ImpDefRegs.insert(Reg);
        if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
          for (const unsigned *SS = tri_->getSubRegisters(Reg); *SS; ++SS)
            ImpDefRegs.insert(*SS);
        }
        ImpDefMIs.push_back(MI);
        continue;
      }

      if (MI->getOpcode() == TargetInstrInfo::INSERT_SUBREG) {
        MachineOperand &MO = MI->getOperand(2);
        if (ImpDefRegs.count(MO.getReg())) {
          // %reg1032<def> = INSERT_SUBREG %reg1032, undef, 2
          // This is an identity copy, eliminate it now.
          if (MO.isKill()) {
            LiveVariables::VarInfo& vi = lv_->getVarInfo(MO.getReg());
            vi.removeKill(MI);
          }
          MI->eraseFromParent();
          continue;
        }
      }

      bool ChangedToImpDef = false;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand& MO = MI->getOperand(i);
        if (!MO.isReg() || !MO.isUse() || MO.isUndef())
          continue;
        unsigned Reg = MO.getReg();
        if (!Reg)
          continue;
        if (!ImpDefRegs.count(Reg))
          continue;
        // Use is a copy, just turn it into an implicit_def.
        if (CanTurnIntoImplicitDef(MI, Reg, i, tii_)) {
          bool isKill = MO.isKill();
          MI->setDesc(tii_->get(TargetInstrInfo::IMPLICIT_DEF));
          for (int j = MI->getNumOperands() - 1, ee = 0; j > ee; --j)
            MI->RemoveOperand(j);
          if (isKill) {
            ImpDefRegs.erase(Reg);
            LiveVariables::VarInfo& vi = lv_->getVarInfo(Reg);
            vi.removeKill(MI);
          }
          ChangedToImpDef = true;
          break;
        }

        MO.setIsUndef();
        if (MO.isKill() || MI->isRegTiedToDefOperand(i)) {
          // Make sure other uses of 
          for (unsigned j = i+1; j != e; ++j) {
            MachineOperand &MOJ = MI->getOperand(j);
            if (MOJ.isReg() && MOJ.isUse() && MOJ.getReg() == Reg)
              MOJ.setIsUndef();
          }
          ImpDefRegs.erase(Reg);
        }
      }

      if (ChangedToImpDef) {
        // Backtrack to process this new implicit_def.
        --I;
      } else {
        for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
          MachineOperand& MO = MI->getOperand(i);
          if (!MO.isReg() || !MO.isDef())
            continue;
          ImpDefRegs.erase(MO.getReg());
        }
      }
    }

    // Any outstanding liveout implicit_def's?
    for (unsigned i = 0, e = ImpDefMIs.size(); i != e; ++i) {
      MachineInstr *MI = ImpDefMIs[i];
      unsigned Reg = MI->getOperand(0).getReg();
      if (TargetRegisterInfo::isPhysicalRegister(Reg) ||
          !ImpDefRegs.count(Reg)) {
        // Delete all "local" implicit_def's. That include those which define
        // physical registers since they cannot be liveout.
        MI->eraseFromParent();
        continue;
      }

      // If there are multiple defs of the same register and at least one
      // is not an implicit_def, do not insert implicit_def's before the
      // uses.
      bool Skip = false;
      for (MachineRegisterInfo::def_iterator DI = mri_->def_begin(Reg),
             DE = mri_->def_end(); DI != DE; ++DI) {
        if (DI->getOpcode() != TargetInstrInfo::IMPLICIT_DEF) {
          Skip = true;
          break;
        }
      }
      if (Skip)
        continue;

      // The only implicit_def which we want to keep are those that are live
      // out of its block.
      MI->eraseFromParent();

      for (MachineRegisterInfo::use_iterator UI = mri_->use_begin(Reg),
             UE = mri_->use_end(); UI != UE; ) {
        MachineOperand &RMO = UI.getOperand();
        MachineInstr *RMI = &*UI;
        ++UI;
        MachineBasicBlock *RMBB = RMI->getParent();
        if (RMBB == MBB)
          continue;

        // Turn a copy use into an implicit_def.
        unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
        if (tii_->isMoveInstr(*RMI, SrcReg, DstReg, SrcSubReg, DstSubReg) &&
            Reg == SrcReg) {
          RMI->setDesc(tii_->get(TargetInstrInfo::IMPLICIT_DEF));
          for (int j = RMI->getNumOperands() - 1, ee = 0; j > ee; --j)
            RMI->RemoveOperand(j);
          continue;
        }

        const TargetRegisterClass* RC = mri_->getRegClass(Reg);
        unsigned NewVReg = mri_->createVirtualRegister(RC);
        RMO.setReg(NewVReg);
        RMO.setIsUndef();
        RMO.setIsKill();
      }
    }
    ImpDefRegs.clear();
    ImpDefMIs.clear();
  }
}


void LiveIntervals::computeNumbering() {
  Index2MiMap OldI2MI = i2miMap_;
  std::vector<IdxMBBPair> OldI2MBB = Idx2MBBMap;
  
  Idx2MBBMap.clear();
  MBB2IdxMap.clear();
  mi2iMap_.clear();
  i2miMap_.clear();
  terminatorGaps.clear();
  phiJoinCopies.clear();
  
  FunctionSize = 0;
  
  // Number MachineInstrs and MachineBasicBlocks.
  // Initialize MBB indexes to a sentinal.
  MBB2IdxMap.resize(mf_->getNumBlockIDs(),
                    std::make_pair(LiveIndex(),LiveIndex()));
  
  LiveIndex MIIndex;
  for (MachineFunction::iterator MBB = mf_->begin(), E = mf_->end();
       MBB != E; ++MBB) {
    LiveIndex StartIdx = MIIndex;

    // Insert an empty slot at the beginning of each block.
    MIIndex = getNextIndex(MIIndex);
    i2miMap_.push_back(0);

    for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end();
         I != E; ++I) {
      
      if (I == MBB->getFirstTerminator()) {
        // Leave a gap for before terminators, this is where we will point
        // PHI kills.
        LiveIndex tGap(true, MIIndex);
        bool inserted =
          terminatorGaps.insert(std::make_pair(&*MBB, tGap)).second;
        assert(inserted && 
               "Multiple 'first' terminators encountered during numbering.");
        inserted = inserted; // Avoid compiler warning if assertions turned off.
        i2miMap_.push_back(0);

        MIIndex = getNextIndex(MIIndex);
      }

      bool inserted = mi2iMap_.insert(std::make_pair(I, MIIndex)).second;
      assert(inserted && "multiple MachineInstr -> index mappings");
      inserted = true;
      i2miMap_.push_back(I);
      MIIndex = getNextIndex(MIIndex);
      FunctionSize++;
      
      // Insert max(1, numdefs) empty slots after every instruction.
      unsigned Slots = I->getDesc().getNumDefs();
      if (Slots == 0)
        Slots = 1;
      while (Slots--) {
        MIIndex = getNextIndex(MIIndex);
        i2miMap_.push_back(0);
      }

    }
  
    if (MBB->getFirstTerminator() == MBB->end()) {
      // Leave a gap for before terminators, this is where we will point
      // PHI kills.
      LiveIndex tGap(true, MIIndex);
      bool inserted =
        terminatorGaps.insert(std::make_pair(&*MBB, tGap)).second;
      assert(inserted && 
             "Multiple 'first' terminators encountered during numbering.");
      inserted = inserted; // Avoid compiler warning if assertions turned off.
      i2miMap_.push_back(0);
 
      MIIndex = getNextIndex(MIIndex);
    }
    
    // Set the MBB2IdxMap entry for this MBB.
    MBB2IdxMap[MBB->getNumber()] = std::make_pair(StartIdx, getPrevSlot(MIIndex));
    Idx2MBBMap.push_back(std::make_pair(StartIdx, MBB));
  }

  std::sort(Idx2MBBMap.begin(), Idx2MBBMap.end(), Idx2MBBCompare());
  
  if (!OldI2MI.empty())
    for (iterator OI = begin(), OE = end(); OI != OE; ++OI) {
      for (LiveInterval::iterator LI = OI->second->begin(),
           LE = OI->second->end(); LI != LE; ++LI) {
        
        // Remap the start index of the live range to the corresponding new
        // number, or our best guess at what it _should_ correspond to if the
        // original instruction has been erased.  This is either the following
        // instruction or its predecessor.
        unsigned index = LI->start.getVecIndex();
        LiveIndex::Slot offset = LI->start.getSlot();
        if (LI->start.isLoad()) {
          std::vector<IdxMBBPair>::const_iterator I =
                  std::lower_bound(OldI2MBB.begin(), OldI2MBB.end(), LI->start);
          // Take the pair containing the index
          std::vector<IdxMBBPair>::const_iterator J =
                    (I == OldI2MBB.end() && OldI2MBB.size()>0) ? (I-1): I;
          
          LI->start = getMBBStartIdx(J->second);
        } else {
          LI->start = LiveIndex(
            LiveIndex(mi2iMap_[OldI2MI[index]]), 
                              (LiveIndex::Slot)offset);
        }
        
        // Remap the ending index in the same way that we remapped the start,
        // except for the final step where we always map to the immediately
        // following instruction.
        index = (getPrevSlot(LI->end)).getVecIndex();
        offset  = LI->end.getSlot();
        if (LI->end.isLoad()) {
          // VReg dies at end of block.
          std::vector<IdxMBBPair>::const_iterator I =
                  std::lower_bound(OldI2MBB.begin(), OldI2MBB.end(), LI->end);
          --I;
          
          LI->end = getNextSlot(getMBBEndIdx(I->second));
        } else {
          unsigned idx = index;
          while (index < OldI2MI.size() && !OldI2MI[index]) ++index;
          
          if (index != OldI2MI.size())
            LI->end =
              LiveIndex(mi2iMap_[OldI2MI[index]],
                (idx == index ? offset : LiveIndex::LOAD));
          else
            LI->end =
              LiveIndex(LiveIndex::NUM * i2miMap_.size());
        }
      }
      
      for (LiveInterval::vni_iterator VNI = OI->second->vni_begin(),
           VNE = OI->second->vni_end(); VNI != VNE; ++VNI) { 
        VNInfo* vni = *VNI;
        
        // Remap the VNInfo def index, which works the same as the
        // start indices above. VN's with special sentinel defs
        // don't need to be remapped.
        if (vni->isDefAccurate() && !vni->isUnused()) {
          unsigned index = vni->def.getVecIndex();
          LiveIndex::Slot offset = vni->def.getSlot();
          if (vni->def.isLoad()) {
            std::vector<IdxMBBPair>::const_iterator I =
                  std::lower_bound(OldI2MBB.begin(), OldI2MBB.end(), vni->def);
            // Take the pair containing the index
            std::vector<IdxMBBPair>::const_iterator J =
                    (I == OldI2MBB.end() && OldI2MBB.size()>0) ? (I-1): I;
          
            vni->def = getMBBStartIdx(J->second);
          } else {
            vni->def = LiveIndex(mi2iMap_[OldI2MI[index]], offset);
          }
        }
        
        // Remap the VNInfo kill indices, which works the same as
        // the end indices above.
        for (size_t i = 0; i < vni->kills.size(); ++i) {
          unsigned index = getPrevSlot(vni->kills[i]).getVecIndex();
          LiveIndex::Slot offset = vni->kills[i].getSlot();

          if (vni->kills[i].isLoad()) {
            assert("Value killed at a load slot.");
            /*std::vector<IdxMBBPair>::const_iterator I =
             std::lower_bound(OldI2MBB.begin(), OldI2MBB.end(), vni->kills[i]);
            --I;

            vni->kills[i] = getMBBEndIdx(I->second);*/
          } else {
            if (vni->kills[i].isPHIIndex()) {
              std::vector<IdxMBBPair>::const_iterator I =
                std::lower_bound(OldI2MBB.begin(), OldI2MBB.end(), vni->kills[i]);
              --I;
              vni->kills[i] = terminatorGaps[I->second];  
            } else {
              assert(OldI2MI[index] != 0 &&
                     "Kill refers to instruction not present in index maps.");
              vni->kills[i] = LiveIndex(mi2iMap_[OldI2MI[index]], offset);
            }
           
            /*
            unsigned idx = index;
            while (index < OldI2MI.size() && !OldI2MI[index]) ++index;
            
            if (index != OldI2MI.size())
              vni->kills[i] = mi2iMap_[OldI2MI[index]] + 
                              (idx == index ? offset : 0);
            else
              vni->kills[i] = InstrSlots::NUM * i2miMap_.size();
            */
          }
        }
      }
    }
}

void LiveIntervals::scaleNumbering(int factor) {
  // Need to
  //  * scale MBB begin and end points
  //  * scale all ranges.
  //  * Update VNI structures.
  //  * Scale instruction numberings 

  // Scale the MBB indices.
  Idx2MBBMap.clear();
  for (MachineFunction::iterator MBB = mf_->begin(), MBBE = mf_->end();
       MBB != MBBE; ++MBB) {
    std::pair<LiveIndex, LiveIndex> &mbbIndices = MBB2IdxMap[MBB->getNumber()];
    mbbIndices.first = mbbIndices.first.scale(factor);
    mbbIndices.second = mbbIndices.second.scale(factor);
    Idx2MBBMap.push_back(std::make_pair(mbbIndices.first, MBB)); 
  }
  std::sort(Idx2MBBMap.begin(), Idx2MBBMap.end(), Idx2MBBCompare());

  // Scale terminator gaps.
  for (DenseMap<MachineBasicBlock*, LiveIndex>::iterator
       TGI = terminatorGaps.begin(), TGE = terminatorGaps.end();
       TGI != TGE; ++TGI) {
    terminatorGaps[TGI->first] = TGI->second.scale(factor);
  }

  // Scale the intervals.
  for (iterator LI = begin(), LE = end(); LI != LE; ++LI) {
    LI->second->scaleNumbering(factor);
  }

  // Scale MachineInstrs.
  Mi2IndexMap oldmi2iMap = mi2iMap_;
  LiveIndex highestSlot;
  for (Mi2IndexMap::iterator MI = oldmi2iMap.begin(), ME = oldmi2iMap.end();
       MI != ME; ++MI) {
    LiveIndex newSlot = MI->second.scale(factor);
    mi2iMap_[MI->first] = newSlot;
    highestSlot = std::max(highestSlot, newSlot); 
  }

  unsigned highestVIndex = highestSlot.getVecIndex();
  i2miMap_.clear();
  i2miMap_.resize(highestVIndex + 1);
  for (Mi2IndexMap::iterator MI = mi2iMap_.begin(), ME = mi2iMap_.end();
       MI != ME; ++MI) {
    i2miMap_[MI->second.getVecIndex()] = const_cast<MachineInstr *>(MI->first);
  }

}


/// runOnMachineFunction - Register allocate the whole function
///
bool LiveIntervals::runOnMachineFunction(MachineFunction &fn) {
  mf_ = &fn;
  mri_ = &mf_->getRegInfo();
  tm_ = &fn.getTarget();
  tri_ = tm_->getRegisterInfo();
  tii_ = tm_->getInstrInfo();
  aa_ = &getAnalysis<AliasAnalysis>();
  lv_ = &getAnalysis<LiveVariables>();
  allocatableRegs_ = tri_->getAllocatableSet(fn);

  processImplicitDefs();
  computeNumbering();
  computeIntervals();
  performEarlyCoalescing();

  numIntervals += getNumIntervals();

  DEBUG(dump());
  return true;
}

/// print - Implement the dump method.
void LiveIntervals::print(raw_ostream &OS, const Module* ) const {
  OS << "********** INTERVALS **********\n";
  for (const_iterator I = begin(), E = end(); I != E; ++I) {
    I->second->print(OS, tri_);
    OS << "\n";
  }

  printInstrs(OS);
}

void LiveIntervals::printInstrs(raw_ostream &OS) const {
  OS << "********** MACHINEINSTRS **********\n";

  for (MachineFunction::iterator mbbi = mf_->begin(), mbbe = mf_->end();
       mbbi != mbbe; ++mbbi) {
    OS << ((Value*)mbbi->getBasicBlock())->getName() << ":\n";
    for (MachineBasicBlock::iterator mii = mbbi->begin(),
           mie = mbbi->end(); mii != mie; ++mii) {
      OS << getInstructionIndex(mii) << '\t' << *mii;
    }
  }
}

void LiveIntervals::dumpInstrs() const {
  printInstrs(errs());
}

/// conflictsWithPhysRegDef - Returns true if the specified register
/// is defined during the duration of the specified interval.
bool LiveIntervals::conflictsWithPhysRegDef(const LiveInterval &li,
                                            VirtRegMap &vrm, unsigned reg) {
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    for (LiveIndex index = getBaseIndex(I->start),
           end = getNextIndex(getBaseIndex(getPrevSlot(I->end))); index != end;
         index = getNextIndex(index)) {
      // skip deleted instructions
      while (index != end && !getInstructionFromIndex(index))
        index = getNextIndex(index);
      if (index == end) break;

      MachineInstr *MI = getInstructionFromIndex(index);
      unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
      if (tii_->isMoveInstr(*MI, SrcReg, DstReg, SrcSubReg, DstSubReg))
        if (SrcReg == li.reg || DstReg == li.reg)
          continue;
      for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
        MachineOperand& mop = MI->getOperand(i);
        if (!mop.isReg())
          continue;
        unsigned PhysReg = mop.getReg();
        if (PhysReg == 0 || PhysReg == li.reg)
          continue;
        if (TargetRegisterInfo::isVirtualRegister(PhysReg)) {
          if (!vrm.hasPhys(PhysReg))
            continue;
          PhysReg = vrm.getPhys(PhysReg);
        }
        if (PhysReg && tri_->regsOverlap(PhysReg, reg))
          return true;
      }
    }
  }

  return false;
}

/// conflictsWithPhysRegRef - Similar to conflictsWithPhysRegRef except
/// it can check use as well.
bool LiveIntervals::conflictsWithPhysRegRef(LiveInterval &li,
                                            unsigned Reg, bool CheckUse,
                                  SmallPtrSet<MachineInstr*,32> &JoinedCopies) {
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    for (LiveIndex index = getBaseIndex(I->start),
           end = getNextIndex(getBaseIndex(getPrevSlot(I->end))); index != end;
         index = getNextIndex(index)) {
      // Skip deleted instructions.
      MachineInstr *MI = 0;
      while (index != end) {
        MI = getInstructionFromIndex(index);
        if (MI)
          break;
        index = getNextIndex(index);
      }
      if (index == end) break;

      if (JoinedCopies.count(MI))
        continue;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand& MO = MI->getOperand(i);
        if (!MO.isReg())
          continue;
        if (MO.isUse() && !CheckUse)
          continue;
        unsigned PhysReg = MO.getReg();
        if (PhysReg == 0 || TargetRegisterInfo::isVirtualRegister(PhysReg))
          continue;
        if (tri_->isSubRegister(Reg, PhysReg))
          return true;
      }
    }
  }

  return false;
}

#ifndef NDEBUG
static void printRegName(unsigned reg, const TargetRegisterInfo* tri_) {
  if (TargetRegisterInfo::isPhysicalRegister(reg))
    errs() << tri_->getName(reg);
  else
    errs() << "%reg" << reg;
}
#endif

void LiveIntervals::handleVirtualRegisterDef(MachineBasicBlock *mbb,
                                             MachineBasicBlock::iterator mi,
                                             LiveIndex MIIdx,
                                             MachineOperand& MO,
                                             unsigned MOIdx,
                                             LiveInterval &interval) {
  DEBUG({
      errs() << "\t\tregister: ";
      printRegName(interval.reg, tri_);
    });

  // Virtual registers may be defined multiple times (due to phi
  // elimination and 2-addr elimination).  Much of what we do only has to be
  // done once for the vreg.  We use an empty interval to detect the first
  // time we see a vreg.
  LiveVariables::VarInfo& vi = lv_->getVarInfo(interval.reg);
  if (interval.empty()) {
    // Get the Idx of the defining instructions.
    LiveIndex defIndex = getDefIndex(MIIdx);
    // Earlyclobbers move back one, so that they overlap the live range
    // of inputs.
    if (MO.isEarlyClobber())
      defIndex = getUseIndex(MIIdx);
    VNInfo *ValNo;
    MachineInstr *CopyMI = NULL;
    unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
    if (mi->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG ||
        mi->getOpcode() == TargetInstrInfo::INSERT_SUBREG ||
        mi->getOpcode() == TargetInstrInfo::SUBREG_TO_REG ||
        tii_->isMoveInstr(*mi, SrcReg, DstReg, SrcSubReg, DstSubReg))
      CopyMI = mi;
    // Earlyclobbers move back one.
    ValNo = interval.getNextValue(defIndex, CopyMI, true, VNInfoAllocator);

    assert(ValNo->id == 0 && "First value in interval is not 0?");

    // Loop over all of the blocks that the vreg is defined in.  There are
    // two cases we have to handle here.  The most common case is a vreg
    // whose lifetime is contained within a basic block.  In this case there
    // will be a single kill, in MBB, which comes after the definition.
    if (vi.Kills.size() == 1 && vi.Kills[0]->getParent() == mbb) {
      // FIXME: what about dead vars?
      LiveIndex killIdx;
      if (vi.Kills[0] != mi)
        killIdx = getNextSlot(getUseIndex(getInstructionIndex(vi.Kills[0])));
      else if (MO.isEarlyClobber())
        // Earlyclobbers that die in this instruction move up one extra, to
        // compensate for having the starting point moved back one.  This
        // gets them to overlap the live range of other outputs.
        killIdx = getNextSlot(getNextSlot(defIndex));
      else
        killIdx = getNextSlot(defIndex);

      // If the kill happens after the definition, we have an intra-block
      // live range.
      if (killIdx > defIndex) {
        assert(vi.AliveBlocks.empty() &&
               "Shouldn't be alive across any blocks!");
        LiveRange LR(defIndex, killIdx, ValNo);
        interval.addRange(LR);
        DEBUG(errs() << " +" << LR << "\n");
        ValNo->addKill(killIdx);
        return;
      }
    }

    // The other case we handle is when a virtual register lives to the end
    // of the defining block, potentially live across some blocks, then is
    // live into some number of blocks, but gets killed.  Start by adding a
    // range that goes from this definition to the end of the defining block.
    LiveRange NewLR(defIndex, getNextSlot(getMBBEndIdx(mbb)), ValNo);
    DEBUG(errs() << " +" << NewLR);
    interval.addRange(NewLR);

    // Iterate over all of the blocks that the variable is completely
    // live in, adding [insrtIndex(begin), instrIndex(end)+4) to the
    // live interval.
    for (SparseBitVector<>::iterator I = vi.AliveBlocks.begin(), 
             E = vi.AliveBlocks.end(); I != E; ++I) {
      LiveRange LR(getMBBStartIdx(*I),
                   getNextSlot(getMBBEndIdx(*I)),  // MBB ends at -1.
                   ValNo);
      interval.addRange(LR);
      DEBUG(errs() << " +" << LR);
    }

    // Finally, this virtual register is live from the start of any killing
    // block to the 'use' slot of the killing instruction.
    for (unsigned i = 0, e = vi.Kills.size(); i != e; ++i) {
      MachineInstr *Kill = vi.Kills[i];
      LiveIndex killIdx =
        getNextSlot(getUseIndex(getInstructionIndex(Kill)));
      LiveRange LR(getMBBStartIdx(Kill->getParent()), killIdx, ValNo);
      interval.addRange(LR);
      ValNo->addKill(killIdx);
      DEBUG(errs() << " +" << LR);
    }

  } else {
    // If this is the second time we see a virtual register definition, it
    // must be due to phi elimination or two addr elimination.  If this is
    // the result of two address elimination, then the vreg is one of the
    // def-and-use register operand.
    if (mi->isRegTiedToUseOperand(MOIdx)) {
      // If this is a two-address definition, then we have already processed
      // the live range.  The only problem is that we didn't realize there
      // are actually two values in the live interval.  Because of this we
      // need to take the LiveRegion that defines this register and split it
      // into two values.
      assert(interval.containsOneValue());
      LiveIndex DefIndex = getDefIndex(interval.getValNumInfo(0)->def);
      LiveIndex RedefIndex = getDefIndex(MIIdx);
      if (MO.isEarlyClobber())
        RedefIndex = getUseIndex(MIIdx);

      const LiveRange *OldLR =
        interval.getLiveRangeContaining(getPrevSlot(RedefIndex));
      VNInfo *OldValNo = OldLR->valno;

      // Delete the initial value, which should be short and continuous,
      // because the 2-addr copy must be in the same MBB as the redef.
      interval.removeRange(DefIndex, RedefIndex);

      // Two-address vregs should always only be redefined once.  This means
      // that at this point, there should be exactly one value number in it.
      assert(interval.containsOneValue() && "Unexpected 2-addr liveint!");

      // The new value number (#1) is defined by the instruction we claimed
      // defined value #0.
      VNInfo *ValNo = interval.getNextValue(OldValNo->def, OldValNo->getCopy(),
                                            false, // update at *
                                            VNInfoAllocator);
      ValNo->setFlags(OldValNo->getFlags()); // * <- updating here

      // Value#0 is now defined by the 2-addr instruction.
      OldValNo->def  = RedefIndex;
      OldValNo->setCopy(0);
      if (MO.isEarlyClobber())
        OldValNo->setHasRedefByEC(true);
      
      // Add the new live interval which replaces the range for the input copy.
      LiveRange LR(DefIndex, RedefIndex, ValNo);
      DEBUG(errs() << " replace range with " << LR);
      interval.addRange(LR);
      ValNo->addKill(RedefIndex);

      // If this redefinition is dead, we need to add a dummy unit live
      // range covering the def slot.
      if (MO.isDead())
        interval.addRange(
          LiveRange(RedefIndex, MO.isEarlyClobber() ?
                                getNextSlot(getNextSlot(RedefIndex)) :
                                getNextSlot(RedefIndex), OldValNo));

      DEBUG({
          errs() << " RESULT: ";
          interval.print(errs(), tri_);
        });
    } else {
      // Otherwise, this must be because of phi elimination.  If this is the
      // first redefinition of the vreg that we have seen, go back and change
      // the live range in the PHI block to be a different value number.
      if (interval.containsOneValue()) {
        // Remove the old range that we now know has an incorrect number.
        VNInfo *VNI = interval.getValNumInfo(0);
        MachineInstr *Killer = vi.Kills[0];
        phiJoinCopies.push_back(Killer);
        LiveIndex Start = getMBBStartIdx(Killer->getParent());
        LiveIndex End =
          getNextSlot(getUseIndex(getInstructionIndex(Killer)));
        DEBUG({
            errs() << " Removing [" << Start << "," << End << "] from: ";
            interval.print(errs(), tri_);
            errs() << "\n";
          });
        interval.removeRange(Start, End);        
        assert(interval.ranges.size() == 1 &&
               "Newly discovered PHI interval has >1 ranges.");
        MachineBasicBlock *killMBB = getMBBFromIndex(interval.endIndex());
        VNI->addKill(terminatorGaps[killMBB]);
        VNI->setHasPHIKill(true);
        DEBUG({
            errs() << " RESULT: ";
            interval.print(errs(), tri_);
          });

        // Replace the interval with one of a NEW value number.  Note that this
        // value number isn't actually defined by an instruction, weird huh? :)
        LiveRange LR(Start, End,
          interval.getNextValue(LiveIndex(mbb->getNumber()),
                                0, false, VNInfoAllocator));
        LR.valno->setIsPHIDef(true);
        DEBUG(errs() << " replace range with " << LR);
        interval.addRange(LR);
        LR.valno->addKill(End);
        DEBUG({
            errs() << " RESULT: ";
            interval.print(errs(), tri_);
          });
      }

      // In the case of PHI elimination, each variable definition is only
      // live until the end of the block.  We've already taken care of the
      // rest of the live range.
      LiveIndex defIndex = getDefIndex(MIIdx);
      if (MO.isEarlyClobber())
        defIndex = getUseIndex(MIIdx);

      VNInfo *ValNo;
      MachineInstr *CopyMI = NULL;
      unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
      if (mi->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG ||
          mi->getOpcode() == TargetInstrInfo::INSERT_SUBREG ||
          mi->getOpcode() == TargetInstrInfo::SUBREG_TO_REG ||
          tii_->isMoveInstr(*mi, SrcReg, DstReg, SrcSubReg, DstSubReg))
        CopyMI = mi;
      ValNo = interval.getNextValue(defIndex, CopyMI, true, VNInfoAllocator);
      
      LiveIndex killIndex = getNextSlot(getMBBEndIdx(mbb));
      LiveRange LR(defIndex, killIndex, ValNo);
      interval.addRange(LR);
      ValNo->addKill(terminatorGaps[mbb]);
      ValNo->setHasPHIKill(true);
      DEBUG(errs() << " +" << LR);
    }
  }

  DEBUG(errs() << '\n');
}

void LiveIntervals::handlePhysicalRegisterDef(MachineBasicBlock *MBB,
                                              MachineBasicBlock::iterator mi,
                                              LiveIndex MIIdx,
                                              MachineOperand& MO,
                                              LiveInterval &interval,
                                              MachineInstr *CopyMI) {
  // A physical register cannot be live across basic block, so its
  // lifetime must end somewhere in its defining basic block.
  DEBUG({
      errs() << "\t\tregister: ";
      printRegName(interval.reg, tri_);
    });

  LiveIndex baseIndex = MIIdx;
  LiveIndex start = getDefIndex(baseIndex);
  // Earlyclobbers move back one.
  if (MO.isEarlyClobber())
    start = getUseIndex(MIIdx);
  LiveIndex end = start;

  // If it is not used after definition, it is considered dead at
  // the instruction defining it. Hence its interval is:
  // [defSlot(def), defSlot(def)+1)
  // For earlyclobbers, the defSlot was pushed back one; the extra
  // advance below compensates.
  if (MO.isDead()) {
    DEBUG(errs() << " dead");
    if (MO.isEarlyClobber())
      end = getNextSlot(getNextSlot(start));
    else
      end = getNextSlot(start);
    goto exit;
  }

  // If it is not dead on definition, it must be killed by a
  // subsequent instruction. Hence its interval is:
  // [defSlot(def), useSlot(kill)+1)
  baseIndex = getNextIndex(baseIndex);
  while (++mi != MBB->end()) {
    while (baseIndex.getVecIndex() < i2miMap_.size() &&
           getInstructionFromIndex(baseIndex) == 0)
      baseIndex = getNextIndex(baseIndex);
    if (mi->killsRegister(interval.reg, tri_)) {
      DEBUG(errs() << " killed");
      end = getNextSlot(getUseIndex(baseIndex));
      goto exit;
    } else {
      int DefIdx = mi->findRegisterDefOperandIdx(interval.reg, false, tri_);
      if (DefIdx != -1) {
        if (mi->isRegTiedToUseOperand(DefIdx)) {
          // Two-address instruction.
          end = getDefIndex(baseIndex);
          if (mi->getOperand(DefIdx).isEarlyClobber())
            end = getUseIndex(baseIndex);
        } else {
          // Another instruction redefines the register before it is ever read.
          // Then the register is essentially dead at the instruction that defines
          // it. Hence its interval is:
          // [defSlot(def), defSlot(def)+1)
          DEBUG(errs() << " dead");
          end = getNextSlot(start);
        }
        goto exit;
      }
    }
    
    baseIndex = getNextIndex(baseIndex);
  }
  
  // The only case we should have a dead physreg here without a killing or
  // instruction where we know it's dead is if it is live-in to the function
  // and never used. Another possible case is the implicit use of the
  // physical register has been deleted by two-address pass.
  end = getNextSlot(start);

exit:
  assert(start < end && "did not find end of interval?");

  // Already exists? Extend old live interval.
  LiveInterval::iterator OldLR = interval.FindLiveRangeContaining(start);
  bool Extend = OldLR != interval.end();
  VNInfo *ValNo = Extend
    ? OldLR->valno : interval.getNextValue(start, CopyMI, true, VNInfoAllocator);
  if (MO.isEarlyClobber() && Extend)
    ValNo->setHasRedefByEC(true);
  LiveRange LR(start, end, ValNo);
  interval.addRange(LR);
  LR.valno->addKill(end);
  DEBUG(errs() << " +" << LR << '\n');
}

void LiveIntervals::handleRegisterDef(MachineBasicBlock *MBB,
                                      MachineBasicBlock::iterator MI,
                                      LiveIndex MIIdx,
                                      MachineOperand& MO,
                                      unsigned MOIdx) {
  if (TargetRegisterInfo::isVirtualRegister(MO.getReg()))
    handleVirtualRegisterDef(MBB, MI, MIIdx, MO, MOIdx,
                             getOrCreateInterval(MO.getReg()));
  else if (allocatableRegs_[MO.getReg()]) {
    MachineInstr *CopyMI = NULL;
    unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
    if (MI->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG ||
        MI->getOpcode() == TargetInstrInfo::INSERT_SUBREG ||
        MI->getOpcode() == TargetInstrInfo::SUBREG_TO_REG ||
        tii_->isMoveInstr(*MI, SrcReg, DstReg, SrcSubReg, DstSubReg))
      CopyMI = MI;
    handlePhysicalRegisterDef(MBB, MI, MIIdx, MO,
                              getOrCreateInterval(MO.getReg()), CopyMI);
    // Def of a register also defines its sub-registers.
    for (const unsigned* AS = tri_->getSubRegisters(MO.getReg()); *AS; ++AS)
      // If MI also modifies the sub-register explicitly, avoid processing it
      // more than once. Do not pass in TRI here so it checks for exact match.
      if (!MI->modifiesRegister(*AS))
        handlePhysicalRegisterDef(MBB, MI, MIIdx, MO,
                                  getOrCreateInterval(*AS), 0);
  }
}

void LiveIntervals::handleLiveInRegister(MachineBasicBlock *MBB,
                                         LiveIndex MIIdx,
                                         LiveInterval &interval, bool isAlias) {
  DEBUG({
      errs() << "\t\tlivein register: ";
      printRegName(interval.reg, tri_);
    });

  // Look for kills, if it reaches a def before it's killed, then it shouldn't
  // be considered a livein.
  MachineBasicBlock::iterator mi = MBB->begin();
  LiveIndex baseIndex = MIIdx;
  LiveIndex start = baseIndex;
  while (baseIndex.getVecIndex() < i2miMap_.size() && 
         getInstructionFromIndex(baseIndex) == 0)
    baseIndex = getNextIndex(baseIndex);
  LiveIndex end = baseIndex;
  bool SeenDefUse = false;
  
  while (mi != MBB->end()) {
    if (mi->killsRegister(interval.reg, tri_)) {
      DEBUG(errs() << " killed");
      end = getNextSlot(getUseIndex(baseIndex));
      SeenDefUse = true;
      break;
    } else if (mi->modifiesRegister(interval.reg, tri_)) {
      // Another instruction redefines the register before it is ever read.
      // Then the register is essentially dead at the instruction that defines
      // it. Hence its interval is:
      // [defSlot(def), defSlot(def)+1)
      DEBUG(errs() << " dead");
      end = getNextSlot(getDefIndex(start));
      SeenDefUse = true;
      break;
    }

    baseIndex = getNextIndex(baseIndex);
    ++mi;
    if (mi != MBB->end()) {
      while (baseIndex.getVecIndex() < i2miMap_.size() && 
             getInstructionFromIndex(baseIndex) == 0)
        baseIndex = getNextIndex(baseIndex);
    }
  }

  // Live-in register might not be used at all.
  if (!SeenDefUse) {
    if (isAlias) {
      DEBUG(errs() << " dead");
      end = getNextSlot(getDefIndex(MIIdx));
    } else {
      DEBUG(errs() << " live through");
      end = baseIndex;
    }
  }

  VNInfo *vni =
    interval.getNextValue(LiveIndex(MBB->getNumber()),
                          0, false, VNInfoAllocator);
  vni->setIsPHIDef(true);
  LiveRange LR(start, end, vni);
  
  interval.addRange(LR);
  LR.valno->addKill(end);
  DEBUG(errs() << " +" << LR << '\n');
}

bool
LiveIntervals::isProfitableToCoalesce(LiveInterval &DstInt, LiveInterval &SrcInt,
                                   SmallVector<MachineInstr*,16> &IdentCopies,
                                   SmallVector<MachineInstr*,16> &OtherCopies) {
  bool HaveConflict = false;
  unsigned NumIdent = 0;
  for (MachineRegisterInfo::def_iterator ri = mri_->def_begin(SrcInt.reg),
         re = mri_->def_end(); ri != re; ++ri) {
    MachineInstr *MI = &*ri;
    unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
    if (!tii_->isMoveInstr(*MI, SrcReg, DstReg, SrcSubReg, DstSubReg))
      return false;
    if (SrcReg != DstInt.reg) {
      OtherCopies.push_back(MI);
      HaveConflict |= DstInt.liveAt(getInstructionIndex(MI));
    } else {
      IdentCopies.push_back(MI);
      ++NumIdent;
    }
  }

  if (!HaveConflict)
    return false; // Let coalescer handle it
  return IdentCopies.size() > OtherCopies.size();
}

void LiveIntervals::performEarlyCoalescing() {
  if (!EarlyCoalescing)
    return;

  /// Perform early coalescing: eliminate copies which feed into phi joins
  /// and whose sources are defined by the phi joins.
  for (unsigned i = 0, e = phiJoinCopies.size(); i != e; ++i) {
    MachineInstr *Join = phiJoinCopies[i];
    if (CoalescingLimit != -1 && (int)numCoalescing == CoalescingLimit)
      break;

    unsigned PHISrc, PHIDst, SrcSubReg, DstSubReg;
    bool isMove= tii_->isMoveInstr(*Join, PHISrc, PHIDst, SrcSubReg, DstSubReg);
#ifndef NDEBUG
    assert(isMove && "PHI join instruction must be a move!");
#else
    isMove = isMove;
#endif

    LiveInterval &DstInt = getInterval(PHIDst);
    LiveInterval &SrcInt = getInterval(PHISrc);
    SmallVector<MachineInstr*, 16> IdentCopies;
    SmallVector<MachineInstr*, 16> OtherCopies;
    if (!isProfitableToCoalesce(DstInt, SrcInt, IdentCopies, OtherCopies))
      continue;

    DEBUG(errs() << "PHI Join: " << *Join);
    assert(DstInt.containsOneValue() && "PHI join should have just one val#!");
    VNInfo *VNI = DstInt.getValNumInfo(0);

    // Change the non-identity copies to directly target the phi destination.
    for (unsigned i = 0, e = OtherCopies.size(); i != e; ++i) {
      MachineInstr *PHICopy = OtherCopies[i];
      DEBUG(errs() << "Moving: " << *PHICopy);

      LiveIndex MIIndex = getInstructionIndex(PHICopy);
      LiveIndex DefIndex = getDefIndex(MIIndex);
      LiveRange *SLR = SrcInt.getLiveRangeContaining(DefIndex);
      LiveIndex StartIndex = SLR->start;
      LiveIndex EndIndex = SLR->end;

      // Delete val# defined by the now identity copy and add the range from
      // beginning of the mbb to the end of the range.
      SrcInt.removeValNo(SLR->valno);
      DEBUG(errs() << "  added range [" << StartIndex << ','
            << EndIndex << "] to reg" << DstInt.reg << '\n');
      if (DstInt.liveAt(StartIndex))
        DstInt.removeRange(StartIndex, EndIndex);
      VNInfo *NewVNI = DstInt.getNextValue(DefIndex, PHICopy, true,
                                           VNInfoAllocator);
      NewVNI->setHasPHIKill(true);
      DstInt.addRange(LiveRange(StartIndex, EndIndex, NewVNI));
      for (unsigned j = 0, ee = PHICopy->getNumOperands(); j != ee; ++j) {
        MachineOperand &MO = PHICopy->getOperand(j);
        if (!MO.isReg() || MO.getReg() != PHISrc)
          continue;
        MO.setReg(PHIDst);
      }
    }

    // Now let's eliminate all the would-be identity copies.
    for (unsigned i = 0, e = IdentCopies.size(); i != e; ++i) {
      MachineInstr *PHICopy = IdentCopies[i];
      DEBUG(errs() << "Coalescing: " << *PHICopy);

      LiveIndex MIIndex = getInstructionIndex(PHICopy);
      LiveIndex DefIndex = getDefIndex(MIIndex);
      LiveRange *SLR = SrcInt.getLiveRangeContaining(DefIndex);
      LiveIndex StartIndex = SLR->start;
      LiveIndex EndIndex = SLR->end;

      // Delete val# defined by the now identity copy and add the range from
      // beginning of the mbb to the end of the range.
      SrcInt.removeValNo(SLR->valno);
      RemoveMachineInstrFromMaps(PHICopy);
      PHICopy->eraseFromParent();
      DEBUG(errs() << "  added range [" << StartIndex << ','
            << EndIndex << "] to reg" << DstInt.reg << '\n');
      DstInt.addRange(LiveRange(StartIndex, EndIndex, VNI));
    }

    // Remove the phi join and update the phi block liveness.
    LiveIndex MIIndex = getInstructionIndex(Join);
    LiveIndex UseIndex = getUseIndex(MIIndex);
    LiveIndex DefIndex = getDefIndex(MIIndex);
    LiveRange *SLR = SrcInt.getLiveRangeContaining(UseIndex);
    LiveRange *DLR = DstInt.getLiveRangeContaining(DefIndex);
    DLR->valno->setCopy(0);
    DLR->valno->setIsDefAccurate(false);
    DstInt.addRange(LiveRange(SLR->start, SLR->end, DLR->valno));
    SrcInt.removeRange(SLR->start, SLR->end);
    assert(SrcInt.empty());
    removeInterval(PHISrc);
    RemoveMachineInstrFromMaps(Join);
    Join->eraseFromParent();

    ++numCoalescing;
  }
}

/// computeIntervals - computes the live intervals for virtual
/// registers. for some ordering of the machine instructions [1,N] a
/// live interval is an interval [i, j) where 1 <= i <= j < N for
/// which a variable is live
void LiveIntervals::computeIntervals() { 
  DEBUG(errs() << "********** COMPUTING LIVE INTERVALS **********\n"
               << "********** Function: "
               << ((Value*)mf_->getFunction())->getName() << '\n');

  SmallVector<unsigned, 8> UndefUses;
  for (MachineFunction::iterator MBBI = mf_->begin(), E = mf_->end();
       MBBI != E; ++MBBI) {
    MachineBasicBlock *MBB = MBBI;
    // Track the index of the current machine instr.
    LiveIndex MIIndex = getMBBStartIdx(MBB);
    DEBUG(errs() << ((Value*)MBB->getBasicBlock())->getName() << ":\n");

    MachineBasicBlock::iterator MI = MBB->begin(), miEnd = MBB->end();

    // Create intervals for live-ins to this BB first.
    for (MachineBasicBlock::const_livein_iterator LI = MBB->livein_begin(),
           LE = MBB->livein_end(); LI != LE; ++LI) {
      handleLiveInRegister(MBB, MIIndex, getOrCreateInterval(*LI));
      // Multiple live-ins can alias the same register.
      for (const unsigned* AS = tri_->getSubRegisters(*LI); *AS; ++AS)
        if (!hasInterval(*AS))
          handleLiveInRegister(MBB, MIIndex, getOrCreateInterval(*AS),
                               true);
    }
    
    // Skip over empty initial indices.
    while (MIIndex.getVecIndex() < i2miMap_.size() &&
           getInstructionFromIndex(MIIndex) == 0)
      MIIndex = getNextIndex(MIIndex);
    
    for (; MI != miEnd; ++MI) {
      DEBUG(errs() << MIIndex << "\t" << *MI);

      // Handle defs.
      for (int i = MI->getNumOperands() - 1; i >= 0; --i) {
        MachineOperand &MO = MI->getOperand(i);
        if (!MO.isReg() || !MO.getReg())
          continue;

        // handle register defs - build intervals
        if (MO.isDef())
          handleRegisterDef(MBB, MI, MIIndex, MO, i);
        else if (MO.isUndef())
          UndefUses.push_back(MO.getReg());
      }

      // Skip over the empty slots after each instruction.
      unsigned Slots = MI->getDesc().getNumDefs();
      if (Slots == 0)
        Slots = 1;

      while (Slots--)
        MIIndex = getNextIndex(MIIndex);
      
      // Skip over empty indices.
      while (MIIndex.getVecIndex() < i2miMap_.size() &&
             getInstructionFromIndex(MIIndex) == 0)
        MIIndex = getNextIndex(MIIndex);
    }
  }

  // Create empty intervals for registers defined by implicit_def's (except
  // for those implicit_def that define values which are liveout of their
  // blocks.
  for (unsigned i = 0, e = UndefUses.size(); i != e; ++i) {
    unsigned UndefReg = UndefUses[i];
    (void)getOrCreateInterval(UndefReg);
  }
}

bool LiveIntervals::findLiveInMBBs(
                              LiveIndex Start, LiveIndex End,
                              SmallVectorImpl<MachineBasicBlock*> &MBBs) const {
  std::vector<IdxMBBPair>::const_iterator I =
    std::lower_bound(Idx2MBBMap.begin(), Idx2MBBMap.end(), Start);

  bool ResVal = false;
  while (I != Idx2MBBMap.end()) {
    if (I->first >= End)
      break;
    MBBs.push_back(I->second);
    ResVal = true;
    ++I;
  }
  return ResVal;
}

bool LiveIntervals::findReachableMBBs(
                              LiveIndex Start, LiveIndex End,
                              SmallVectorImpl<MachineBasicBlock*> &MBBs) const {
  std::vector<IdxMBBPair>::const_iterator I =
    std::lower_bound(Idx2MBBMap.begin(), Idx2MBBMap.end(), Start);

  bool ResVal = false;
  while (I != Idx2MBBMap.end()) {
    if (I->first > End)
      break;
    MachineBasicBlock *MBB = I->second;
    if (getMBBEndIdx(MBB) > End)
      break;
    for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
           SE = MBB->succ_end(); SI != SE; ++SI)
      MBBs.push_back(*SI);
    ResVal = true;
    ++I;
  }
  return ResVal;
}

LiveInterval* LiveIntervals::createInterval(unsigned reg) {
  float Weight = TargetRegisterInfo::isPhysicalRegister(reg) ? HUGE_VALF : 0.0F;
  return new LiveInterval(reg, Weight);
}

/// dupInterval - Duplicate a live interval. The caller is responsible for
/// managing the allocated memory.
LiveInterval* LiveIntervals::dupInterval(LiveInterval *li) {
  LiveInterval *NewLI = createInterval(li->reg);
  NewLI->Copy(*li, mri_, getVNInfoAllocator());
  return NewLI;
}

/// getVNInfoSourceReg - Helper function that parses the specified VNInfo
/// copy field and returns the source register that defines it.
unsigned LiveIntervals::getVNInfoSourceReg(const VNInfo *VNI) const {
  if (!VNI->getCopy())
    return 0;

  if (VNI->getCopy()->getOpcode() == TargetInstrInfo::EXTRACT_SUBREG) {
    // If it's extracting out of a physical register, return the sub-register.
    unsigned Reg = VNI->getCopy()->getOperand(1).getReg();
    if (TargetRegisterInfo::isPhysicalRegister(Reg))
      Reg = tri_->getSubReg(Reg, VNI->getCopy()->getOperand(2).getImm());
    return Reg;
  } else if (VNI->getCopy()->getOpcode() == TargetInstrInfo::INSERT_SUBREG ||
             VNI->getCopy()->getOpcode() == TargetInstrInfo::SUBREG_TO_REG)
    return VNI->getCopy()->getOperand(2).getReg();

  unsigned SrcReg, DstReg, SrcSubReg, DstSubReg;
  if (tii_->isMoveInstr(*VNI->getCopy(), SrcReg, DstReg, SrcSubReg, DstSubReg))
    return SrcReg;
  llvm_unreachable("Unrecognized copy instruction!");
  return 0;
}

//===----------------------------------------------------------------------===//
// Register allocator hooks.
//

/// getReMatImplicitUse - If the remat definition MI has one (for now, we only
/// allow one) virtual register operand, then its uses are implicitly using
/// the register. Returns the virtual register.
unsigned LiveIntervals::getReMatImplicitUse(const LiveInterval &li,
                                            MachineInstr *MI) const {
  unsigned RegOp = 0;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0 || Reg == li.reg)
      continue;
    
    if (TargetRegisterInfo::isPhysicalRegister(Reg) &&
        !allocatableRegs_[Reg])
      continue;
    // FIXME: For now, only remat MI with at most one register operand.
    assert(!RegOp &&
           "Can't rematerialize instruction with multiple register operand!");
    RegOp = MO.getReg();
#ifndef NDEBUG
    break;
#endif
  }
  return RegOp;
}

/// isValNoAvailableAt - Return true if the val# of the specified interval
/// which reaches the given instruction also reaches the specified use index.
bool LiveIntervals::isValNoAvailableAt(const LiveInterval &li, MachineInstr *MI,
                                       LiveIndex UseIdx) const {
  LiveIndex Index = getInstructionIndex(MI);  
  VNInfo *ValNo = li.FindLiveRangeContaining(Index)->valno;
  LiveInterval::const_iterator UI = li.FindLiveRangeContaining(UseIdx);
  return UI != li.end() && UI->valno == ValNo;
}

/// isReMaterializable - Returns true if the definition MI of the specified
/// val# of the specified interval is re-materializable.
bool LiveIntervals::isReMaterializable(const LiveInterval &li,
                                       const VNInfo *ValNo, MachineInstr *MI,
                                       SmallVectorImpl<LiveInterval*> &SpillIs,
                                       bool &isLoad) {
  if (DisableReMat)
    return false;

  if (MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF)
    return true;

  int FrameIdx = 0;
  if (tii_->isLoadFromStackSlot(MI, FrameIdx) &&
      mf_->getFrameInfo()->isImmutableObjectIndex(FrameIdx))
    // FIXME: Let target specific isReallyTriviallyReMaterializable determines
    // this but remember this is not safe to fold into a two-address
    // instruction.
    // This is a load from fixed stack slot. It can be rematerialized.
    return true;

  // If the target-specific rules don't identify an instruction as
  // being trivially rematerializable, use some target-independent
  // rules.
  if (!tii_->isTriviallyReMaterializable(MI)) {
    if (!EnableAggressiveRemat)
      return false;

    const TargetInstrDesc &TID = MI->getDesc();

    // Avoid instructions obviously unsafe for remat.
    if (TID.hasUnmodeledSideEffects() || TID.isNotDuplicable() ||
        TID.mayStore())
      return false;

    // Avoid instructions which load from potentially varying memory.
    if (TID.mayLoad() && !MI->isInvariantLoad(aa_))
      return false;

    // If any of the registers accessed are non-constant, conservatively assume
    // the instruction is not rematerializable.
    unsigned ImpUse = 0;
    for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = MI->getOperand(i);
      if (MO.isReg()) {
        unsigned Reg = MO.getReg();
        if (Reg == 0)
          continue;
        if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
          if (MO.isUse()) {
            // If the physreg has no defs anywhere, it's just an ambient register
            // and we can freely move its uses. Alternatively, if it's allocatable,
            // it could get allocated to something with a def during allocation.
            if (!mri_->def_empty(Reg))
              return false;
            if (allocatableRegs_.test(Reg))
              return false;
            // Check for a def among the register's aliases too.
            for (const unsigned *Alias = tri_->getAliasSet(Reg); *Alias; ++Alias) {
              unsigned AliasReg = *Alias;
              if (!mri_->def_empty(AliasReg))
                return false;
              if (allocatableRegs_.test(AliasReg))
                return false;
            }
          } else {
            // A physreg def. We can't remat it.
            return false;
          }
          continue;
        }

        // Only allow one def, and that in the first operand.
        if (MO.isDef() != (i == 0))
          return false;

        // Only allow constant-valued registers.
        bool IsLiveIn = mri_->isLiveIn(Reg);
        MachineRegisterInfo::def_iterator I = mri_->def_begin(Reg),
                                          E = mri_->def_end();

        // For the def, it should be the only def of that register.
        if (MO.isDef() && (next(I) != E || IsLiveIn))
          return false;

        if (MO.isUse()) {
          // Only allow one use other register use, as that's all the
          // remat mechanisms support currently.
          if (Reg != li.reg) {
            if (ImpUse == 0)
              ImpUse = Reg;
            else if (Reg != ImpUse)
              return false;
          }
          // For the use, there should be only one associated def.
          if (I != E && (next(I) != E || IsLiveIn))
            return false;
        }
      }
    }
  }

  unsigned ImpUse = getReMatImplicitUse(li, MI);
  if (ImpUse) {
    const LiveInterval &ImpLi = getInterval(ImpUse);
    for (MachineRegisterInfo::use_iterator ri = mri_->use_begin(li.reg),
           re = mri_->use_end(); ri != re; ++ri) {
      MachineInstr *UseMI = &*ri;
      LiveIndex UseIdx = getInstructionIndex(UseMI);
      if (li.FindLiveRangeContaining(UseIdx)->valno != ValNo)
        continue;
      if (!isValNoAvailableAt(ImpLi, MI, UseIdx))
        return false;
    }

    // If a register operand of the re-materialized instruction is going to
    // be spilled next, then it's not legal to re-materialize this instruction.
    for (unsigned i = 0, e = SpillIs.size(); i != e; ++i)
      if (ImpUse == SpillIs[i]->reg)
        return false;
  }
  return true;
}

/// isReMaterializable - Returns true if the definition MI of the specified
/// val# of the specified interval is re-materializable.
bool LiveIntervals::isReMaterializable(const LiveInterval &li,
                                       const VNInfo *ValNo, MachineInstr *MI) {
  SmallVector<LiveInterval*, 4> Dummy1;
  bool Dummy2;
  return isReMaterializable(li, ValNo, MI, Dummy1, Dummy2);
}

/// isReMaterializable - Returns true if every definition of MI of every
/// val# of the specified interval is re-materializable.
bool LiveIntervals::isReMaterializable(const LiveInterval &li,
                                       SmallVectorImpl<LiveInterval*> &SpillIs,
                                       bool &isLoad) {
  isLoad = false;
  for (LiveInterval::const_vni_iterator i = li.vni_begin(), e = li.vni_end();
       i != e; ++i) {
    const VNInfo *VNI = *i;
    if (VNI->isUnused())
      continue; // Dead val#.
    // Is the def for the val# rematerializable?
    if (!VNI->isDefAccurate())
      return false;
    MachineInstr *ReMatDefMI = getInstructionFromIndex(VNI->def);
    bool DefIsLoad = false;
    if (!ReMatDefMI ||
        !isReMaterializable(li, VNI, ReMatDefMI, SpillIs, DefIsLoad))
      return false;
    isLoad |= DefIsLoad;
  }
  return true;
}

/// FilterFoldedOps - Filter out two-address use operands. Return
/// true if it finds any issue with the operands that ought to prevent
/// folding.
static bool FilterFoldedOps(MachineInstr *MI,
                            SmallVector<unsigned, 2> &Ops,
                            unsigned &MRInfo,
                            SmallVector<unsigned, 2> &FoldOps) {
  MRInfo = 0;
  for (unsigned i = 0, e = Ops.size(); i != e; ++i) {
    unsigned OpIdx = Ops[i];
    MachineOperand &MO = MI->getOperand(OpIdx);
    // FIXME: fold subreg use.
    if (MO.getSubReg())
      return true;
    if (MO.isDef())
      MRInfo |= (unsigned)VirtRegMap::isMod;
    else {
      // Filter out two-address use operand(s).
      if (MI->isRegTiedToDefOperand(OpIdx)) {
        MRInfo = VirtRegMap::isModRef;
        continue;
      }
      MRInfo |= (unsigned)VirtRegMap::isRef;
    }
    FoldOps.push_back(OpIdx);
  }
  return false;
}
                           

/// tryFoldMemoryOperand - Attempts to fold either a spill / restore from
/// slot / to reg or any rematerialized load into ith operand of specified
/// MI. If it is successul, MI is updated with the newly created MI and
/// returns true.
bool LiveIntervals::tryFoldMemoryOperand(MachineInstr* &MI,
                                         VirtRegMap &vrm, MachineInstr *DefMI,
                                         LiveIndex InstrIdx,
                                         SmallVector<unsigned, 2> &Ops,
                                         bool isSS, int Slot, unsigned Reg) {
  // If it is an implicit def instruction, just delete it.
  if (MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF) {
    RemoveMachineInstrFromMaps(MI);
    vrm.RemoveMachineInstrFromMaps(MI);
    MI->eraseFromParent();
    ++numFolds;
    return true;
  }

  // Filter the list of operand indexes that are to be folded. Abort if
  // any operand will prevent folding.
  unsigned MRInfo = 0;
  SmallVector<unsigned, 2> FoldOps;
  if (FilterFoldedOps(MI, Ops, MRInfo, FoldOps))
    return false;

  // The only time it's safe to fold into a two address instruction is when
  // it's folding reload and spill from / into a spill stack slot.
  if (DefMI && (MRInfo & VirtRegMap::isMod))
    return false;

  MachineInstr *fmi = isSS ? tii_->foldMemoryOperand(*mf_, MI, FoldOps, Slot)
                           : tii_->foldMemoryOperand(*mf_, MI, FoldOps, DefMI);
  if (fmi) {
    // Remember this instruction uses the spill slot.
    if (isSS) vrm.addSpillSlotUse(Slot, fmi);

    // Attempt to fold the memory reference into the instruction. If
    // we can do this, we don't need to insert spill code.
    MachineBasicBlock &MBB = *MI->getParent();
    if (isSS && !mf_->getFrameInfo()->isImmutableObjectIndex(Slot))
      vrm.virtFolded(Reg, MI, fmi, (VirtRegMap::ModRef)MRInfo);
    vrm.transferSpillPts(MI, fmi);
    vrm.transferRestorePts(MI, fmi);
    vrm.transferEmergencySpills(MI, fmi);
    mi2iMap_.erase(MI);
    i2miMap_[InstrIdx.getVecIndex()] = fmi;
    mi2iMap_[fmi] = InstrIdx;
    MI = MBB.insert(MBB.erase(MI), fmi);
    ++numFolds;
    return true;
  }
  return false;
}

/// canFoldMemoryOperand - Returns true if the specified load / store
/// folding is possible.
bool LiveIntervals::canFoldMemoryOperand(MachineInstr *MI,
                                         SmallVector<unsigned, 2> &Ops,
                                         bool ReMat) const {
  // Filter the list of operand indexes that are to be folded. Abort if
  // any operand will prevent folding.
  unsigned MRInfo = 0;
  SmallVector<unsigned, 2> FoldOps;
  if (FilterFoldedOps(MI, Ops, MRInfo, FoldOps))
    return false;

  // It's only legal to remat for a use, not a def.
  if (ReMat && (MRInfo & VirtRegMap::isMod))
    return false;

  return tii_->canFoldMemoryOperand(MI, FoldOps);
}

bool LiveIntervals::intervalIsInOneMBB(const LiveInterval &li) const {
  SmallPtrSet<MachineBasicBlock*, 4> MBBs;
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    std::vector<IdxMBBPair>::const_iterator II =
      std::lower_bound(Idx2MBBMap.begin(), Idx2MBBMap.end(), I->start);
    if (II == Idx2MBBMap.end())
      continue;
    if (I->end > II->first)  // crossing a MBB.
      return false;
    MBBs.insert(II->second);
    if (MBBs.size() > 1)
      return false;
  }
  return true;
}

/// rewriteImplicitOps - Rewrite implicit use operands of MI (i.e. uses of
/// interval on to-be re-materialized operands of MI) with new register.
void LiveIntervals::rewriteImplicitOps(const LiveInterval &li,
                                       MachineInstr *MI, unsigned NewVReg,
                                       VirtRegMap &vrm) {
  // There is an implicit use. That means one of the other operand is
  // being remat'ed and the remat'ed instruction has li.reg as an
  // use operand. Make sure we rewrite that as well.
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (Reg == 0 || TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (!vrm.isReMaterialized(Reg))
      continue;
    MachineInstr *ReMatMI = vrm.getReMaterializedMI(Reg);
    MachineOperand *UseMO = ReMatMI->findRegisterUseOperand(li.reg);
    if (UseMO)
      UseMO->setReg(NewVReg);
  }
}

/// rewriteInstructionForSpills, rewriteInstructionsForSpills - Helper functions
/// for addIntervalsForSpills to rewrite uses / defs for the given live range.
bool LiveIntervals::
rewriteInstructionForSpills(const LiveInterval &li, const VNInfo *VNI,
                 bool TrySplit, LiveIndex index, LiveIndex end, 
                 MachineInstr *MI,
                 MachineInstr *ReMatOrigDefMI, MachineInstr *ReMatDefMI,
                 unsigned Slot, int LdSlot,
                 bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
                 VirtRegMap &vrm,
                 const TargetRegisterClass* rc,
                 SmallVector<int, 4> &ReMatIds,
                 const MachineLoopInfo *loopInfo,
                 unsigned &NewVReg, unsigned ImpUse, bool &HasDef, bool &HasUse,
                 DenseMap<unsigned,unsigned> &MBBVRegsMap,
                 std::vector<LiveInterval*> &NewLIs) {
  bool CanFold = false;
 RestartInstruction:
  for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
    MachineOperand& mop = MI->getOperand(i);
    if (!mop.isReg())
      continue;
    unsigned Reg = mop.getReg();
    unsigned RegI = Reg;
    if (Reg == 0 || TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (Reg != li.reg)
      continue;

    bool TryFold = !DefIsReMat;
    bool FoldSS = true; // Default behavior unless it's a remat.
    int FoldSlot = Slot;
    if (DefIsReMat) {
      // If this is the rematerializable definition MI itself and
      // all of its uses are rematerialized, simply delete it.
      if (MI == ReMatOrigDefMI && CanDelete) {
        DEBUG(errs() << "\t\t\t\tErasing re-materlizable def: "
                     << MI << '\n');
        RemoveMachineInstrFromMaps(MI);
        vrm.RemoveMachineInstrFromMaps(MI);
        MI->eraseFromParent();
        break;
      }

      // If def for this use can't be rematerialized, then try folding.
      // If def is rematerializable and it's a load, also try folding.
      TryFold = !ReMatDefMI || (ReMatDefMI && (MI == ReMatOrigDefMI || isLoad));
      if (isLoad) {
        // Try fold loads (from stack slot, constant pool, etc.) into uses.
        FoldSS = isLoadSS;
        FoldSlot = LdSlot;
      }
    }

    // Scan all of the operands of this instruction rewriting operands
    // to use NewVReg instead of li.reg as appropriate.  We do this for
    // two reasons:
    //
    //   1. If the instr reads the same spilled vreg multiple times, we
    //      want to reuse the NewVReg.
    //   2. If the instr is a two-addr instruction, we are required to
    //      keep the src/dst regs pinned.
    //
    // Keep track of whether we replace a use and/or def so that we can
    // create the spill interval with the appropriate range. 

    HasUse = mop.isUse();
    HasDef = mop.isDef();
    SmallVector<unsigned, 2> Ops;
    Ops.push_back(i);
    for (unsigned j = i+1, e = MI->getNumOperands(); j != e; ++j) {
      const MachineOperand &MOj = MI->getOperand(j);
      if (!MOj.isReg())
        continue;
      unsigned RegJ = MOj.getReg();
      if (RegJ == 0 || TargetRegisterInfo::isPhysicalRegister(RegJ))
        continue;
      if (RegJ == RegI) {
        Ops.push_back(j);
        if (!MOj.isUndef()) {
          HasUse |= MOj.isUse();
          HasDef |= MOj.isDef();
        }
      }
    }

    // Create a new virtual register for the spill interval.
    // Create the new register now so we can map the fold instruction
    // to the new register so when it is unfolded we get the correct
    // answer.
    bool CreatedNewVReg = false;
    if (NewVReg == 0) {
      NewVReg = mri_->createVirtualRegister(rc);
      vrm.grow();
      CreatedNewVReg = true;
    }

    if (!TryFold)
      CanFold = false;
    else {
      // Do not fold load / store here if we are splitting. We'll find an
      // optimal point to insert a load / store later.
      if (!TrySplit) {
        if (tryFoldMemoryOperand(MI, vrm, ReMatDefMI, index,
                                 Ops, FoldSS, FoldSlot, NewVReg)) {
          // Folding the load/store can completely change the instruction in
          // unpredictable ways, rescan it from the beginning.

          if (FoldSS) {
            // We need to give the new vreg the same stack slot as the
            // spilled interval.
            vrm.assignVirt2StackSlot(NewVReg, FoldSlot);
          }

          HasUse = false;
          HasDef = false;
          CanFold = false;
          if (isNotInMIMap(MI))
            break;
          goto RestartInstruction;
        }
      } else {
        // We'll try to fold it later if it's profitable.
        CanFold = canFoldMemoryOperand(MI, Ops, DefIsReMat);
      }
    }

    mop.setReg(NewVReg);
    if (mop.isImplicit())
      rewriteImplicitOps(li, MI, NewVReg, vrm);

    // Reuse NewVReg for other reads.
    for (unsigned j = 0, e = Ops.size(); j != e; ++j) {
      MachineOperand &mopj = MI->getOperand(Ops[j]);
      mopj.setReg(NewVReg);
      if (mopj.isImplicit())
        rewriteImplicitOps(li, MI, NewVReg, vrm);
    }
            
    if (CreatedNewVReg) {
      if (DefIsReMat) {
        vrm.setVirtIsReMaterialized(NewVReg, ReMatDefMI);
        if (ReMatIds[VNI->id] == VirtRegMap::MAX_STACK_SLOT) {
          // Each valnum may have its own remat id.
          ReMatIds[VNI->id] = vrm.assignVirtReMatId(NewVReg);
        } else {
          vrm.assignVirtReMatId(NewVReg, ReMatIds[VNI->id]);
        }
        if (!CanDelete || (HasUse && HasDef)) {
          // If this is a two-addr instruction then its use operands are
          // rematerializable but its def is not. It should be assigned a
          // stack slot.
          vrm.assignVirt2StackSlot(NewVReg, Slot);
        }
      } else {
        vrm.assignVirt2StackSlot(NewVReg, Slot);
      }
    } else if (HasUse && HasDef &&
               vrm.getStackSlot(NewVReg) == VirtRegMap::NO_STACK_SLOT) {
      // If this interval hasn't been assigned a stack slot (because earlier
      // def is a deleted remat def), do it now.
      assert(Slot != VirtRegMap::NO_STACK_SLOT);
      vrm.assignVirt2StackSlot(NewVReg, Slot);
    }

    // Re-matting an instruction with virtual register use. Add the
    // register as an implicit use on the use MI.
    if (DefIsReMat && ImpUse)
      MI->addOperand(MachineOperand::CreateReg(ImpUse, false, true));

    // Create a new register interval for this spill / remat.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    if (CreatedNewVReg) {
      NewLIs.push_back(&nI);
      MBBVRegsMap.insert(std::make_pair(MI->getParent()->getNumber(), NewVReg));
      if (TrySplit)
        vrm.setIsSplitFromReg(NewVReg, li.reg);
    }

    if (HasUse) {
      if (CreatedNewVReg) {
        LiveRange LR(getLoadIndex(index), getNextSlot(getUseIndex(index)),
                     nI.getNextValue(LiveIndex(), 0, false,
                                     VNInfoAllocator));
        DEBUG(errs() << " +" << LR);
        nI.addRange(LR);
      } else {
        // Extend the split live interval to this def / use.
        LiveIndex End = getNextSlot(getUseIndex(index));
        LiveRange LR(nI.ranges[nI.ranges.size()-1].end, End,
                     nI.getValNumInfo(nI.getNumValNums()-1));
        DEBUG(errs() << " +" << LR);
        nI.addRange(LR);
      }
    }
    if (HasDef) {
      LiveRange LR(getDefIndex(index), getStoreIndex(index),
                   nI.getNextValue(LiveIndex(), 0, false,
                                   VNInfoAllocator));
      DEBUG(errs() << " +" << LR);
      nI.addRange(LR);
    }

    DEBUG({
        errs() << "\t\t\t\tAdded new interval: ";
        nI.print(errs(), tri_);
        errs() << '\n';
      });
  }
  return CanFold;
}
bool LiveIntervals::anyKillInMBBAfterIdx(const LiveInterval &li,
                                   const VNInfo *VNI,
                                   MachineBasicBlock *MBB,
                                   LiveIndex Idx) const {
  LiveIndex End = getMBBEndIdx(MBB);
  for (unsigned j = 0, ee = VNI->kills.size(); j != ee; ++j) {
    if (VNI->kills[j].isPHIIndex())
      continue;

    LiveIndex KillIdx = VNI->kills[j];
    if (KillIdx > Idx && KillIdx < End)
      return true;
  }
  return false;
}

/// RewriteInfo - Keep track of machine instrs that will be rewritten
/// during spilling.
namespace {
  struct RewriteInfo {
    LiveIndex Index;
    MachineInstr *MI;
    bool HasUse;
    bool HasDef;
    RewriteInfo(LiveIndex i, MachineInstr *mi, bool u, bool d)
      : Index(i), MI(mi), HasUse(u), HasDef(d) {}
  };

  struct RewriteInfoCompare {
    bool operator()(const RewriteInfo &LHS, const RewriteInfo &RHS) const {
      return LHS.Index < RHS.Index;
    }
  };
}

void LiveIntervals::
rewriteInstructionsForSpills(const LiveInterval &li, bool TrySplit,
                    LiveInterval::Ranges::const_iterator &I,
                    MachineInstr *ReMatOrigDefMI, MachineInstr *ReMatDefMI,
                    unsigned Slot, int LdSlot,
                    bool isLoad, bool isLoadSS, bool DefIsReMat, bool CanDelete,
                    VirtRegMap &vrm,
                    const TargetRegisterClass* rc,
                    SmallVector<int, 4> &ReMatIds,
                    const MachineLoopInfo *loopInfo,
                    BitVector &SpillMBBs,
                    DenseMap<unsigned, std::vector<SRInfo> > &SpillIdxes,
                    BitVector &RestoreMBBs,
                    DenseMap<unsigned, std::vector<SRInfo> > &RestoreIdxes,
                    DenseMap<unsigned,unsigned> &MBBVRegsMap,
                    std::vector<LiveInterval*> &NewLIs) {
  bool AllCanFold = true;
  unsigned NewVReg = 0;
  LiveIndex start = getBaseIndex(I->start);
  LiveIndex end = getNextIndex(getBaseIndex(getPrevSlot(I->end)));

  // First collect all the def / use in this live range that will be rewritten.
  // Make sure they are sorted according to instruction index.
  std::vector<RewriteInfo> RewriteMIs;
  for (MachineRegisterInfo::reg_iterator ri = mri_->reg_begin(li.reg),
         re = mri_->reg_end(); ri != re; ) {
    MachineInstr *MI = &*ri;
    MachineOperand &O = ri.getOperand();
    ++ri;
    assert(!O.isImplicit() && "Spilling register that's used as implicit use?");
    LiveIndex index = getInstructionIndex(MI);
    if (index < start || index >= end)
      continue;

    if (O.isUndef())
      // Must be defined by an implicit def. It should not be spilled. Note,
      // this is for correctness reason. e.g.
      // 8   %reg1024<def> = IMPLICIT_DEF
      // 12  %reg1024<def> = INSERT_SUBREG %reg1024<kill>, %reg1025, 2
      // The live range [12, 14) are not part of the r1024 live interval since
      // it's defined by an implicit def. It will not conflicts with live
      // interval of r1025. Now suppose both registers are spilled, you can
      // easily see a situation where both registers are reloaded before
      // the INSERT_SUBREG and both target registers that would overlap.
      continue;
    RewriteMIs.push_back(RewriteInfo(index, MI, O.isUse(), O.isDef()));
  }
  std::sort(RewriteMIs.begin(), RewriteMIs.end(), RewriteInfoCompare());

  unsigned ImpUse = DefIsReMat ? getReMatImplicitUse(li, ReMatDefMI) : 0;
  // Now rewrite the defs and uses.
  for (unsigned i = 0, e = RewriteMIs.size(); i != e; ) {
    RewriteInfo &rwi = RewriteMIs[i];
    ++i;
    LiveIndex index = rwi.Index;
    bool MIHasUse = rwi.HasUse;
    bool MIHasDef = rwi.HasDef;
    MachineInstr *MI = rwi.MI;
    // If MI def and/or use the same register multiple times, then there
    // are multiple entries.
    unsigned NumUses = MIHasUse;
    while (i != e && RewriteMIs[i].MI == MI) {
      assert(RewriteMIs[i].Index == index);
      bool isUse = RewriteMIs[i].HasUse;
      if (isUse) ++NumUses;
      MIHasUse |= isUse;
      MIHasDef |= RewriteMIs[i].HasDef;
      ++i;
    }
    MachineBasicBlock *MBB = MI->getParent();

    if (ImpUse && MI != ReMatDefMI) {
      // Re-matting an instruction with virtual register use. Update the
      // register interval's spill weight to HUGE_VALF to prevent it from
      // being spilled.
      LiveInterval &ImpLi = getInterval(ImpUse);
      ImpLi.weight = HUGE_VALF;
    }

    unsigned MBBId = MBB->getNumber();
    unsigned ThisVReg = 0;
    if (TrySplit) {
      DenseMap<unsigned,unsigned>::iterator NVI = MBBVRegsMap.find(MBBId);
      if (NVI != MBBVRegsMap.end()) {
        ThisVReg = NVI->second;
        // One common case:
        // x = use
        // ...
        // ...
        // def = ...
        //     = use
        // It's better to start a new interval to avoid artifically
        // extend the new interval.
        if (MIHasDef && !MIHasUse) {
          MBBVRegsMap.erase(MBB->getNumber());
          ThisVReg = 0;
        }
      }
    }

    bool IsNew = ThisVReg == 0;
    if (IsNew) {
      // This ends the previous live interval. If all of its def / use
      // can be folded, give it a low spill weight.
      if (NewVReg && TrySplit && AllCanFold) {
        LiveInterval &nI = getOrCreateInterval(NewVReg);
        nI.weight /= 10.0F;
      }
      AllCanFold = true;
    }
    NewVReg = ThisVReg;

    bool HasDef = false;
    bool HasUse = false;
    bool CanFold = rewriteInstructionForSpills(li, I->valno, TrySplit,
                         index, end, MI, ReMatOrigDefMI, ReMatDefMI,
                         Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                         CanDelete, vrm, rc, ReMatIds, loopInfo, NewVReg,
                         ImpUse, HasDef, HasUse, MBBVRegsMap, NewLIs);
    if (!HasDef && !HasUse)
      continue;

    AllCanFold &= CanFold;

    // Update weight of spill interval.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    if (!TrySplit) {
      // The spill weight is now infinity as it cannot be spilled again.
      nI.weight = HUGE_VALF;
      continue;
    }

    // Keep track of the last def and first use in each MBB.
    if (HasDef) {
      if (MI != ReMatOrigDefMI || !CanDelete) {
        bool HasKill = false;
        if (!HasUse)
          HasKill = anyKillInMBBAfterIdx(li, I->valno, MBB, getDefIndex(index));
        else {
          // If this is a two-address code, then this index starts a new VNInfo.
          const VNInfo *VNI = li.findDefinedVNInfoForRegInt(getDefIndex(index));
          if (VNI)
            HasKill = anyKillInMBBAfterIdx(li, VNI, MBB, getDefIndex(index));
        }
        DenseMap<unsigned, std::vector<SRInfo> >::iterator SII =
          SpillIdxes.find(MBBId);
        if (!HasKill) {
          if (SII == SpillIdxes.end()) {
            std::vector<SRInfo> S;
            S.push_back(SRInfo(index, NewVReg, true));
            SpillIdxes.insert(std::make_pair(MBBId, S));
          } else if (SII->second.back().vreg != NewVReg) {
            SII->second.push_back(SRInfo(index, NewVReg, true));
          } else if (index > SII->second.back().index) {
            // If there is an earlier def and this is a two-address
            // instruction, then it's not possible to fold the store (which
            // would also fold the load).
            SRInfo &Info = SII->second.back();
            Info.index = index;
            Info.canFold = !HasUse;
          }
          SpillMBBs.set(MBBId);
        } else if (SII != SpillIdxes.end() &&
                   SII->second.back().vreg == NewVReg &&
                   index > SII->second.back().index) {
          // There is an earlier def that's not killed (must be two-address).
          // The spill is no longer needed.
          SII->second.pop_back();
          if (SII->second.empty()) {
            SpillIdxes.erase(MBBId);
            SpillMBBs.reset(MBBId);
          }
        }
      }
    }

    if (HasUse) {
      DenseMap<unsigned, std::vector<SRInfo> >::iterator SII =
        SpillIdxes.find(MBBId);
      if (SII != SpillIdxes.end() &&
          SII->second.back().vreg == NewVReg &&
          index > SII->second.back().index)
        // Use(s) following the last def, it's not safe to fold the spill.
        SII->second.back().canFold = false;
      DenseMap<unsigned, std::vector<SRInfo> >::iterator RII =
        RestoreIdxes.find(MBBId);
      if (RII != RestoreIdxes.end() && RII->second.back().vreg == NewVReg)
        // If we are splitting live intervals, only fold if it's the first
        // use and there isn't another use later in the MBB.
        RII->second.back().canFold = false;
      else if (IsNew) {
        // Only need a reload if there isn't an earlier def / use.
        if (RII == RestoreIdxes.end()) {
          std::vector<SRInfo> Infos;
          Infos.push_back(SRInfo(index, NewVReg, true));
          RestoreIdxes.insert(std::make_pair(MBBId, Infos));
        } else {
          RII->second.push_back(SRInfo(index, NewVReg, true));
        }
        RestoreMBBs.set(MBBId);
      }
    }

    // Update spill weight.
    unsigned loopDepth = loopInfo->getLoopDepth(MBB);
    nI.weight += getSpillWeight(HasDef, HasUse, loopDepth);
  }

  if (NewVReg && TrySplit && AllCanFold) {
    // If all of its def / use can be folded, give it a low spill weight.
    LiveInterval &nI = getOrCreateInterval(NewVReg);
    nI.weight /= 10.0F;
  }
}

bool LiveIntervals::alsoFoldARestore(int Id, LiveIndex index,
                        unsigned vr, BitVector &RestoreMBBs,
                        DenseMap<unsigned,std::vector<SRInfo> > &RestoreIdxes) {
  if (!RestoreMBBs[Id])
    return false;
  std::vector<SRInfo> &Restores = RestoreIdxes[Id];
  for (unsigned i = 0, e = Restores.size(); i != e; ++i)
    if (Restores[i].index == index &&
        Restores[i].vreg == vr &&
        Restores[i].canFold)
      return true;
  return false;
}

void LiveIntervals::eraseRestoreInfo(int Id, LiveIndex index,
                        unsigned vr, BitVector &RestoreMBBs,
                        DenseMap<unsigned,std::vector<SRInfo> > &RestoreIdxes) {
  if (!RestoreMBBs[Id])
    return;
  std::vector<SRInfo> &Restores = RestoreIdxes[Id];
  for (unsigned i = 0, e = Restores.size(); i != e; ++i)
    if (Restores[i].index == index && Restores[i].vreg)
      Restores[i].index = LiveIndex();
}

/// handleSpilledImpDefs - Remove IMPLICIT_DEF instructions which are being
/// spilled and create empty intervals for their uses.
void
LiveIntervals::handleSpilledImpDefs(const LiveInterval &li, VirtRegMap &vrm,
                                    const TargetRegisterClass* rc,
                                    std::vector<LiveInterval*> &NewLIs) {
  for (MachineRegisterInfo::reg_iterator ri = mri_->reg_begin(li.reg),
         re = mri_->reg_end(); ri != re; ) {
    MachineOperand &O = ri.getOperand();
    MachineInstr *MI = &*ri;
    ++ri;
    if (O.isDef()) {
      assert(MI->getOpcode() == TargetInstrInfo::IMPLICIT_DEF &&
             "Register def was not rewritten?");
      RemoveMachineInstrFromMaps(MI);
      vrm.RemoveMachineInstrFromMaps(MI);
      MI->eraseFromParent();
    } else {
      // This must be an use of an implicit_def so it's not part of the live
      // interval. Create a new empty live interval for it.
      // FIXME: Can we simply erase some of the instructions? e.g. Stores?
      unsigned NewVReg = mri_->createVirtualRegister(rc);
      vrm.grow();
      vrm.setIsImplicitlyDefined(NewVReg);
      NewLIs.push_back(&getOrCreateInterval(NewVReg));
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        MachineOperand &MO = MI->getOperand(i);
        if (MO.isReg() && MO.getReg() == li.reg) {
          MO.setReg(NewVReg);
          MO.setIsUndef();
        }
      }
    }
  }
}

std::vector<LiveInterval*> LiveIntervals::
addIntervalsForSpillsFast(const LiveInterval &li,
                          const MachineLoopInfo *loopInfo,
                          VirtRegMap &vrm) {
  unsigned slot = vrm.assignVirt2StackSlot(li.reg);

  std::vector<LiveInterval*> added;

  assert(li.weight != HUGE_VALF &&
         "attempt to spill already spilled interval!");

  DEBUG({
      errs() << "\t\t\t\tadding intervals for spills for interval: ";
      li.dump();
      errs() << '\n';
    });

  const TargetRegisterClass* rc = mri_->getRegClass(li.reg);

  MachineRegisterInfo::reg_iterator RI = mri_->reg_begin(li.reg);
  while (RI != mri_->reg_end()) {
    MachineInstr* MI = &*RI;
    
    SmallVector<unsigned, 2> Indices;
    bool HasUse = false;
    bool HasDef = false;
    
    for (unsigned i = 0; i != MI->getNumOperands(); ++i) {
      MachineOperand& mop = MI->getOperand(i);
      if (!mop.isReg() || mop.getReg() != li.reg) continue;
      
      HasUse |= MI->getOperand(i).isUse();
      HasDef |= MI->getOperand(i).isDef();
      
      Indices.push_back(i);
    }
    
    if (!tryFoldMemoryOperand(MI, vrm, NULL, getInstructionIndex(MI),
                              Indices, true, slot, li.reg)) {
      unsigned NewVReg = mri_->createVirtualRegister(rc);
      vrm.grow();
      vrm.assignVirt2StackSlot(NewVReg, slot);
      
      // create a new register for this spill
      LiveInterval &nI = getOrCreateInterval(NewVReg);

      // the spill weight is now infinity as it
      // cannot be spilled again
      nI.weight = HUGE_VALF;
      
      // Rewrite register operands to use the new vreg.
      for (SmallVectorImpl<unsigned>::iterator I = Indices.begin(),
           E = Indices.end(); I != E; ++I) {
        MI->getOperand(*I).setReg(NewVReg);
        
        if (MI->getOperand(*I).isUse())
          MI->getOperand(*I).setIsKill(true);
      }
      
      // Fill in  the new live interval.
      LiveIndex index = getInstructionIndex(MI);
      if (HasUse) {
        LiveRange LR(getLoadIndex(index), getUseIndex(index),
                     nI.getNextValue(LiveIndex(), 0, false,
                                     getVNInfoAllocator()));
        DEBUG(errs() << " +" << LR);
        nI.addRange(LR);
        vrm.addRestorePoint(NewVReg, MI);
      }
      if (HasDef) {
        LiveRange LR(getDefIndex(index), getStoreIndex(index),
                     nI.getNextValue(LiveIndex(), 0, false,
                                     getVNInfoAllocator()));
        DEBUG(errs() << " +" << LR);
        nI.addRange(LR);
        vrm.addSpillPoint(NewVReg, true, MI);
      }
      
      added.push_back(&nI);
        
      DEBUG({
          errs() << "\t\t\t\tadded new interval: ";
          nI.dump();
          errs() << '\n';
        });
    }
    
    
    RI = mri_->reg_begin(li.reg);
  }

  return added;
}

std::vector<LiveInterval*> LiveIntervals::
addIntervalsForSpills(const LiveInterval &li,
                      SmallVectorImpl<LiveInterval*> &SpillIs,
                      const MachineLoopInfo *loopInfo, VirtRegMap &vrm) {
  
  if (EnableFastSpilling)
    return addIntervalsForSpillsFast(li, loopInfo, vrm);
  
  assert(li.weight != HUGE_VALF &&
         "attempt to spill already spilled interval!");

  DEBUG({
      errs() << "\t\t\t\tadding intervals for spills for interval: ";
      li.print(errs(), tri_);
      errs() << '\n';
    });

  // Each bit specify whether a spill is required in the MBB.
  BitVector SpillMBBs(mf_->getNumBlockIDs());
  DenseMap<unsigned, std::vector<SRInfo> > SpillIdxes;
  BitVector RestoreMBBs(mf_->getNumBlockIDs());
  DenseMap<unsigned, std::vector<SRInfo> > RestoreIdxes;
  DenseMap<unsigned,unsigned> MBBVRegsMap;
  std::vector<LiveInterval*> NewLIs;
  const TargetRegisterClass* rc = mri_->getRegClass(li.reg);

  unsigned NumValNums = li.getNumValNums();
  SmallVector<MachineInstr*, 4> ReMatDefs;
  ReMatDefs.resize(NumValNums, NULL);
  SmallVector<MachineInstr*, 4> ReMatOrigDefs;
  ReMatOrigDefs.resize(NumValNums, NULL);
  SmallVector<int, 4> ReMatIds;
  ReMatIds.resize(NumValNums, VirtRegMap::MAX_STACK_SLOT);
  BitVector ReMatDelete(NumValNums);
  unsigned Slot = VirtRegMap::MAX_STACK_SLOT;

  // Spilling a split live interval. It cannot be split any further. Also,
  // it's also guaranteed to be a single val# / range interval.
  if (vrm.getPreSplitReg(li.reg)) {
    vrm.setIsSplitFromReg(li.reg, 0);
    // Unset the split kill marker on the last use.
    LiveIndex KillIdx = vrm.getKillPoint(li.reg);
    if (KillIdx != LiveIndex()) {
      MachineInstr *KillMI = getInstructionFromIndex(KillIdx);
      assert(KillMI && "Last use disappeared?");
      int KillOp = KillMI->findRegisterUseOperandIdx(li.reg, true);
      assert(KillOp != -1 && "Last use disappeared?");
      KillMI->getOperand(KillOp).setIsKill(false);
    }
    vrm.removeKillPoint(li.reg);
    bool DefIsReMat = vrm.isReMaterialized(li.reg);
    Slot = vrm.getStackSlot(li.reg);
    assert(Slot != VirtRegMap::MAX_STACK_SLOT);
    MachineInstr *ReMatDefMI = DefIsReMat ?
      vrm.getReMaterializedMI(li.reg) : NULL;
    int LdSlot = 0;
    bool isLoadSS = DefIsReMat && tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
    bool isLoad = isLoadSS ||
      (DefIsReMat && (ReMatDefMI->getDesc().canFoldAsLoad()));
    bool IsFirstRange = true;
    for (LiveInterval::Ranges::const_iterator
           I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
      // If this is a split live interval with multiple ranges, it means there
      // are two-address instructions that re-defined the value. Only the
      // first def can be rematerialized!
      if (IsFirstRange) {
        // Note ReMatOrigDefMI has already been deleted.
        rewriteInstructionsForSpills(li, false, I, NULL, ReMatDefMI,
                             Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                             false, vrm, rc, ReMatIds, loopInfo,
                             SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                             MBBVRegsMap, NewLIs);
      } else {
        rewriteInstructionsForSpills(li, false, I, NULL, 0,
                             Slot, 0, false, false, false,
                             false, vrm, rc, ReMatIds, loopInfo,
                             SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                             MBBVRegsMap, NewLIs);
      }
      IsFirstRange = false;
    }

    handleSpilledImpDefs(li, vrm, rc, NewLIs);
    return NewLIs;
  }

  bool TrySplit = !intervalIsInOneMBB(li);
  if (TrySplit)
    ++numSplits;
  bool NeedStackSlot = false;
  for (LiveInterval::const_vni_iterator i = li.vni_begin(), e = li.vni_end();
       i != e; ++i) {
    const VNInfo *VNI = *i;
    unsigned VN = VNI->id;
    if (VNI->isUnused())
      continue; // Dead val#.
    // Is the def for the val# rematerializable?
    MachineInstr *ReMatDefMI = VNI->isDefAccurate()
      ? getInstructionFromIndex(VNI->def) : 0;
    bool dummy;
    if (ReMatDefMI && isReMaterializable(li, VNI, ReMatDefMI, SpillIs, dummy)) {
      // Remember how to remat the def of this val#.
      ReMatOrigDefs[VN] = ReMatDefMI;
      // Original def may be modified so we have to make a copy here.
      MachineInstr *Clone = mf_->CloneMachineInstr(ReMatDefMI);
      CloneMIs.push_back(Clone);
      ReMatDefs[VN] = Clone;

      bool CanDelete = true;
      if (VNI->hasPHIKill()) {
        // A kill is a phi node, not all of its uses can be rematerialized.
        // It must not be deleted.
        CanDelete = false;
        // Need a stack slot if there is any live range where uses cannot be
        // rematerialized.
        NeedStackSlot = true;
      }
      if (CanDelete)
        ReMatDelete.set(VN);
    } else {
      // Need a stack slot if there is any live range where uses cannot be
      // rematerialized.
      NeedStackSlot = true;
    }
  }

  // One stack slot per live interval.
  if (NeedStackSlot && vrm.getPreSplitReg(li.reg) == 0) {
    if (vrm.getStackSlot(li.reg) == VirtRegMap::NO_STACK_SLOT)
      Slot = vrm.assignVirt2StackSlot(li.reg);
    
    // This case only occurs when the prealloc splitter has already assigned
    // a stack slot to this vreg.
    else
      Slot = vrm.getStackSlot(li.reg);
  }

  // Create new intervals and rewrite defs and uses.
  for (LiveInterval::Ranges::const_iterator
         I = li.ranges.begin(), E = li.ranges.end(); I != E; ++I) {
    MachineInstr *ReMatDefMI = ReMatDefs[I->valno->id];
    MachineInstr *ReMatOrigDefMI = ReMatOrigDefs[I->valno->id];
    bool DefIsReMat = ReMatDefMI != NULL;
    bool CanDelete = ReMatDelete[I->valno->id];
    int LdSlot = 0;
    bool isLoadSS = DefIsReMat && tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
    bool isLoad = isLoadSS ||
      (DefIsReMat && ReMatDefMI->getDesc().canFoldAsLoad());
    rewriteInstructionsForSpills(li, TrySplit, I, ReMatOrigDefMI, ReMatDefMI,
                               Slot, LdSlot, isLoad, isLoadSS, DefIsReMat,
                               CanDelete, vrm, rc, ReMatIds, loopInfo,
                               SpillMBBs, SpillIdxes, RestoreMBBs, RestoreIdxes,
                               MBBVRegsMap, NewLIs);
  }

  // Insert spills / restores if we are splitting.
  if (!TrySplit) {
    handleSpilledImpDefs(li, vrm, rc, NewLIs);
    return NewLIs;
  }

  SmallPtrSet<LiveInterval*, 4> AddedKill;
  SmallVector<unsigned, 2> Ops;
  if (NeedStackSlot) {
    int Id = SpillMBBs.find_first();
    while (Id != -1) {
      std::vector<SRInfo> &spills = SpillIdxes[Id];
      for (unsigned i = 0, e = spills.size(); i != e; ++i) {
        LiveIndex index = spills[i].index;
        unsigned VReg = spills[i].vreg;
        LiveInterval &nI = getOrCreateInterval(VReg);
        bool isReMat = vrm.isReMaterialized(VReg);
        MachineInstr *MI = getInstructionFromIndex(index);
        bool CanFold = false;
        bool FoundUse = false;
        Ops.clear();
        if (spills[i].canFold) {
          CanFold = true;
          for (unsigned j = 0, ee = MI->getNumOperands(); j != ee; ++j) {
            MachineOperand &MO = MI->getOperand(j);
            if (!MO.isReg() || MO.getReg() != VReg)
              continue;

            Ops.push_back(j);
            if (MO.isDef())
              continue;
            if (isReMat || 
                (!FoundUse && !alsoFoldARestore(Id, index, VReg,
                                                RestoreMBBs, RestoreIdxes))) {
              // MI has two-address uses of the same register. If the use
              // isn't the first and only use in the BB, then we can't fold
              // it. FIXME: Move this to rewriteInstructionsForSpills.
              CanFold = false;
              break;
            }
            FoundUse = true;
          }
        }
        // Fold the store into the def if possible.
        bool Folded = false;
        if (CanFold && !Ops.empty()) {
          if (tryFoldMemoryOperand(MI, vrm, NULL, index, Ops, true, Slot,VReg)){
            Folded = true;
            if (FoundUse) {
              // Also folded uses, do not issue a load.
              eraseRestoreInfo(Id, index, VReg, RestoreMBBs, RestoreIdxes);
              nI.removeRange(getLoadIndex(index), getNextSlot(getUseIndex(index)));
            }
            nI.removeRange(getDefIndex(index), getStoreIndex(index));
          }
        }

        // Otherwise tell the spiller to issue a spill.
        if (!Folded) {
          LiveRange *LR = &nI.ranges[nI.ranges.size()-1];
          bool isKill = LR->end == getStoreIndex(index);
          if (!MI->registerDefIsDead(nI.reg))
            // No need to spill a dead def.
            vrm.addSpillPoint(VReg, isKill, MI);
          if (isKill)
            AddedKill.insert(&nI);
        }
      }
      Id = SpillMBBs.find_next(Id);
    }
  }

  int Id = RestoreMBBs.find_first();
  while (Id != -1) {
    std::vector<SRInfo> &restores = RestoreIdxes[Id];
    for (unsigned i = 0, e = restores.size(); i != e; ++i) {
      LiveIndex index = restores[i].index;
      if (index == LiveIndex())
        continue;
      unsigned VReg = restores[i].vreg;
      LiveInterval &nI = getOrCreateInterval(VReg);
      bool isReMat = vrm.isReMaterialized(VReg);
      MachineInstr *MI = getInstructionFromIndex(index);
      bool CanFold = false;
      Ops.clear();
      if (restores[i].canFold) {
        CanFold = true;
        for (unsigned j = 0, ee = MI->getNumOperands(); j != ee; ++j) {
          MachineOperand &MO = MI->getOperand(j);
          if (!MO.isReg() || MO.getReg() != VReg)
            continue;

          if (MO.isDef()) {
            // If this restore were to be folded, it would have been folded
            // already.
            CanFold = false;
            break;
          }
          Ops.push_back(j);
        }
      }

      // Fold the load into the use if possible.
      bool Folded = false;
      if (CanFold && !Ops.empty()) {
        if (!isReMat)
          Folded = tryFoldMemoryOperand(MI, vrm, NULL,index,Ops,true,Slot,VReg);
        else {
          MachineInstr *ReMatDefMI = vrm.getReMaterializedMI(VReg);
          int LdSlot = 0;
          bool isLoadSS = tii_->isLoadFromStackSlot(ReMatDefMI, LdSlot);
          // If the rematerializable def is a load, also try to fold it.
          if (isLoadSS || ReMatDefMI->getDesc().canFoldAsLoad())
            Folded = tryFoldMemoryOperand(MI, vrm, ReMatDefMI, index,
                                          Ops, isLoadSS, LdSlot, VReg);
          if (!Folded) {
            unsigned ImpUse = getReMatImplicitUse(li, ReMatDefMI);
            if (ImpUse) {
              // Re-matting an instruction with virtual register use. Add the
              // register as an implicit use on the use MI and update the register
              // interval's spill weight to HUGE_VALF to prevent it from being
              // spilled.
              LiveInterval &ImpLi = getInterval(ImpUse);
              ImpLi.weight = HUGE_VALF;
              MI->addOperand(MachineOperand::CreateReg(ImpUse, false, true));
            }
          }
        }
      }
      // If folding is not possible / failed, then tell the spiller to issue a
      // load / rematerialization for us.
      if (Folded)
        nI.removeRange(getLoadIndex(index), getNextSlot(getUseIndex(index)));
      else
        vrm.addRestorePoint(VReg, MI);
    }
    Id = RestoreMBBs.find_next(Id);
  }

  // Finalize intervals: add kills, finalize spill weights, and filter out
  // dead intervals.
  std::vector<LiveInterval*> RetNewLIs;
  for (unsigned i = 0, e = NewLIs.size(); i != e; ++i) {
    LiveInterval *LI = NewLIs[i];
    if (!LI->empty()) {
      LI->weight /= InstrSlots::NUM * getApproximateInstructionCount(*LI);
      if (!AddedKill.count(LI)) {
        LiveRange *LR = &LI->ranges[LI->ranges.size()-1];
        LiveIndex LastUseIdx = getBaseIndex(LR->end);
        MachineInstr *LastUse = getInstructionFromIndex(LastUseIdx);
        int UseIdx = LastUse->findRegisterUseOperandIdx(LI->reg, false);
        assert(UseIdx != -1);
        if (!LastUse->isRegTiedToDefOperand(UseIdx)) {
          LastUse->getOperand(UseIdx).setIsKill();
          vrm.addKillPoint(LI->reg, LastUseIdx);
        }
      }
      RetNewLIs.push_back(LI);
    }
  }

  handleSpilledImpDefs(li, vrm, rc, RetNewLIs);
  return RetNewLIs;
}

/// hasAllocatableSuperReg - Return true if the specified physical register has
/// any super register that's allocatable.
bool LiveIntervals::hasAllocatableSuperReg(unsigned Reg) const {
  for (const unsigned* AS = tri_->getSuperRegisters(Reg); *AS; ++AS)
    if (allocatableRegs_[*AS] && hasInterval(*AS))
      return true;
  return false;
}

/// getRepresentativeReg - Find the largest super register of the specified
/// physical register.
unsigned LiveIntervals::getRepresentativeReg(unsigned Reg) const {
  // Find the largest super-register that is allocatable. 
  unsigned BestReg = Reg;
  for (const unsigned* AS = tri_->getSuperRegisters(Reg); *AS; ++AS) {
    unsigned SuperReg = *AS;
    if (!hasAllocatableSuperReg(SuperReg) && hasInterval(SuperReg)) {
      BestReg = SuperReg;
      break;
    }
  }
  return BestReg;
}

/// getNumConflictsWithPhysReg - Return the number of uses and defs of the
/// specified interval that conflicts with the specified physical register.
unsigned LiveIntervals::getNumConflictsWithPhysReg(const LiveInterval &li,
                                                   unsigned PhysReg) const {
  unsigned NumConflicts = 0;
  const LiveInterval &pli = getInterval(getRepresentativeReg(PhysReg));
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(li.reg),
         E = mri_->reg_end(); I != E; ++I) {
    MachineOperand &O = I.getOperand();
    MachineInstr *MI = O.getParent();
    LiveIndex Index = getInstructionIndex(MI);
    if (pli.liveAt(Index))
      ++NumConflicts;
  }
  return NumConflicts;
}

/// spillPhysRegAroundRegDefsUses - Spill the specified physical register
/// around all defs and uses of the specified interval. Return true if it
/// was able to cut its interval.
bool LiveIntervals::spillPhysRegAroundRegDefsUses(const LiveInterval &li,
                                            unsigned PhysReg, VirtRegMap &vrm) {
  unsigned SpillReg = getRepresentativeReg(PhysReg);

  for (const unsigned *AS = tri_->getAliasSet(PhysReg); *AS; ++AS)
    // If there are registers which alias PhysReg, but which are not a
    // sub-register of the chosen representative super register. Assert
    // since we can't handle it yet.
    assert(*AS == SpillReg || !allocatableRegs_[*AS] || !hasInterval(*AS) ||
           tri_->isSuperRegister(*AS, SpillReg));

  bool Cut = false;
  LiveInterval &pli = getInterval(SpillReg);
  SmallPtrSet<MachineInstr*, 8> SeenMIs;
  for (MachineRegisterInfo::reg_iterator I = mri_->reg_begin(li.reg),
         E = mri_->reg_end(); I != E; ++I) {
    MachineOperand &O = I.getOperand();
    MachineInstr *MI = O.getParent();
    if (SeenMIs.count(MI))
      continue;
    SeenMIs.insert(MI);
    LiveIndex Index = getInstructionIndex(MI);
    if (pli.liveAt(Index)) {
      vrm.addEmergencySpill(SpillReg, MI);
      LiveIndex StartIdx = getLoadIndex(Index);
      LiveIndex EndIdx = getNextSlot(getStoreIndex(Index));
      if (pli.isInOneLiveRange(StartIdx, EndIdx)) {
        pli.removeRange(StartIdx, EndIdx);
        Cut = true;
      } else {
        std::string msg;
        raw_string_ostream Msg(msg);
        Msg << "Ran out of registers during register allocation!";
        if (MI->getOpcode() == TargetInstrInfo::INLINEASM) {
          Msg << "\nPlease check your inline asm statement for invalid "
               << "constraints:\n";
          MI->print(Msg, tm_);
        }
        llvm_report_error(Msg.str());
      }
      for (const unsigned* AS = tri_->getSubRegisters(SpillReg); *AS; ++AS) {
        if (!hasInterval(*AS))
          continue;
        LiveInterval &spli = getInterval(*AS);
        if (spli.liveAt(Index))
          spli.removeRange(getLoadIndex(Index), getNextSlot(getStoreIndex(Index)));
      }
    }
  }
  return Cut;
}

LiveRange LiveIntervals::addLiveRangeToEndOfBlock(unsigned reg,
                                                  MachineInstr* startInst) {
  LiveInterval& Interval = getOrCreateInterval(reg);
  VNInfo* VN = Interval.getNextValue(
    LiveIndex(getInstructionIndex(startInst), LiveIndex::DEF),
    startInst, true, getVNInfoAllocator());
  VN->setHasPHIKill(true);
  VN->kills.push_back(terminatorGaps[startInst->getParent()]);
  LiveRange LR(
    LiveIndex(getInstructionIndex(startInst), LiveIndex::DEF),
    getNextSlot(getMBBEndIdx(startInst->getParent())), VN);
  Interval.addRange(LR);
  
  return LR;
}

