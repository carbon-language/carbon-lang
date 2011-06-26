//===-- llvm/CodeGen/Splitter.cpp -  Splitter -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "loopsplitter"

#include "Splitter.h"

#include "RegisterCoalescer.h"
#include "llvm/Module.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/SlotIndexes.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"

using namespace llvm;

char LoopSplitter::ID = 0;
INITIALIZE_PASS_BEGIN(LoopSplitter, "loop-splitting",
                "Split virtual regists across loop boundaries.", false, false)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(SlotIndexes)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_END(LoopSplitter, "loop-splitting",
                "Split virtual regists across loop boundaries.", false, false)

namespace llvm {

  class StartSlotComparator {
  public:
    StartSlotComparator(LiveIntervals &lis) : lis(lis) {}
    bool operator()(const MachineBasicBlock *mbb1,
                    const MachineBasicBlock *mbb2) const {
      return lis.getMBBStartIdx(mbb1) < lis.getMBBStartIdx(mbb2);
    }
  private:
    LiveIntervals &lis;
  };

  class LoopSplit {
  public:
    LoopSplit(LoopSplitter &ls, LiveInterval &li, MachineLoop &loop)
      : ls(ls), li(li), loop(loop), valid(true), inSplit(false), newLI(0) {
      assert(TargetRegisterInfo::isVirtualRegister(li.reg) &&
             "Cannot split physical registers.");
    }

    LiveInterval& getLI() const { return li; }

    MachineLoop& getLoop() const { return loop; }

    bool isValid() const { return valid; }

    bool isWorthwhile() const { return valid && (inSplit || !outSplits.empty()); }

    void invalidate() { valid = false; }

    void splitIncoming() { inSplit = true; }

    void splitOutgoing(MachineLoop::Edge &edge) { outSplits.insert(edge); }

    void addLoopInstr(MachineInstr *i) { loopInstrs.push_back(i); }

    void apply() {
      assert(valid && "Attempt to apply invalid split.");
      applyIncoming();
      applyOutgoing();
      copyRanges();
      renameInside();
    }

  private:
    LoopSplitter &ls;
    LiveInterval &li;
    MachineLoop &loop;
    bool valid, inSplit;
    std::set<MachineLoop::Edge> outSplits;
    std::vector<MachineInstr*> loopInstrs;

    LiveInterval *newLI;
    std::map<VNInfo*, VNInfo*> vniMap;

    LiveInterval* getNewLI() {
      if (newLI == 0) {
        const TargetRegisterClass *trc = ls.mri->getRegClass(li.reg);
        unsigned vreg = ls.mri->createVirtualRegister(trc);
        newLI = &ls.lis->getOrCreateInterval(vreg);
      }
      return newLI;
    }

    VNInfo* getNewVNI(VNInfo *oldVNI) {
      VNInfo *newVNI = vniMap[oldVNI];

      if (newVNI == 0) {
        newVNI = getNewLI()->createValueCopy(oldVNI,
                                             ls.lis->getVNInfoAllocator());
        vniMap[oldVNI] = newVNI;
      }

      return newVNI;
    }

    void applyIncoming() {
      if (!inSplit) {
        return;
      }

      MachineBasicBlock *preHeader = loop.getLoopPreheader();
      if (preHeader == 0) {
        assert(ls.canInsertPreHeader(loop) &&
               "Can't insert required preheader.");
        preHeader = &ls.insertPreHeader(loop);
      }

      LiveRange *preHeaderRange =
        ls.lis->findExitingRange(li, preHeader);
      assert(preHeaderRange != 0 && "Range not live into preheader.");

      // Insert the new copy.
      MachineInstr *copy = BuildMI(*preHeader,
                                   preHeader->getFirstTerminator(),
                                   DebugLoc(),
                                   ls.tii->get(TargetOpcode::COPY))
        .addReg(getNewLI()->reg, RegState::Define)
        .addReg(li.reg, RegState::Kill);

      ls.lis->InsertMachineInstrInMaps(copy);

      SlotIndex copyDefIdx = ls.lis->getInstructionIndex(copy).getDefIndex();

      VNInfo *newVal = getNewVNI(preHeaderRange->valno);
      newVal->def = copyDefIdx;
      newVal->setCopy(copy);
      li.removeRange(copyDefIdx, ls.lis->getMBBEndIdx(preHeader), true);

      getNewLI()->addRange(LiveRange(copyDefIdx,
                                     ls.lis->getMBBEndIdx(preHeader),
                                     newVal));
    }

    void applyOutgoing() {

      for (std::set<MachineLoop::Edge>::iterator osItr = outSplits.begin(),
                                                 osEnd = outSplits.end();
           osItr != osEnd; ++osItr) {
        MachineLoop::Edge edge = *osItr;
        MachineBasicBlock *outBlock = edge.second;
        if (ls.isCriticalEdge(edge)) {
          assert(ls.canSplitEdge(edge) && "Unsplitable critical edge.");
          outBlock = &ls.splitEdge(edge, loop);
        }
        LiveRange *outRange = ls.lis->findEnteringRange(li, outBlock);
        assert(outRange != 0 && "No exiting range?");

        MachineInstr *copy = BuildMI(*outBlock, outBlock->begin(),
                                     DebugLoc(),
                                     ls.tii->get(TargetOpcode::COPY))
          .addReg(li.reg, RegState::Define)
          .addReg(getNewLI()->reg, RegState::Kill);

        ls.lis->InsertMachineInstrInMaps(copy);

        SlotIndex copyDefIdx = ls.lis->getInstructionIndex(copy).getDefIndex();
        
        // Blow away output range definition.
        outRange->valno->def = ls.lis->getInvalidIndex();
        li.removeRange(ls.lis->getMBBStartIdx(outBlock), copyDefIdx);

        SlotIndex newDefIdx = ls.lis->getMBBStartIdx(outBlock);
        assert(ls.lis->getInstructionFromIndex(newDefIdx) == 0 &&
               "PHI def index points at actual instruction.");
        VNInfo *newVal =
          getNewLI()->getNextValue(newDefIdx, 0, ls.lis->getVNInfoAllocator());

        getNewLI()->addRange(LiveRange(ls.lis->getMBBStartIdx(outBlock),
                                       copyDefIdx, newVal));
                                       
      }
    }

    void copyRange(LiveRange &lr) {
      std::pair<bool, LoopSplitter::SlotPair> lsr =
        ls.getLoopSubRange(lr, loop);
      
      if (!lsr.first)
        return;

      LiveRange loopRange(lsr.second.first, lsr.second.second,
                          getNewVNI(lr.valno));

      li.removeRange(loopRange.start, loopRange.end, true);

      getNewLI()->addRange(loopRange);
    }

    void copyRanges() {
      for (std::vector<MachineInstr*>::iterator iItr = loopInstrs.begin(),
                                                iEnd = loopInstrs.end();
           iItr != iEnd; ++iItr) {
        MachineInstr &instr = **iItr;
        SlotIndex instrIdx = ls.lis->getInstructionIndex(&instr);
        if (instr.modifiesRegister(li.reg, 0)) {
          LiveRange *defRange =
            li.getLiveRangeContaining(instrIdx.getDefIndex());
          if (defRange != 0) // May have caught this already.
            copyRange(*defRange);
        }
        if (instr.readsRegister(li.reg, 0)) {
          LiveRange *useRange =
            li.getLiveRangeContaining(instrIdx.getUseIndex());
          if (useRange != 0) { // May have caught this already.
            copyRange(*useRange);
          }
        }
      }

      for (MachineLoop::block_iterator bbItr = loop.block_begin(),
                                       bbEnd = loop.block_end();
           bbItr != bbEnd; ++bbItr) {
        MachineBasicBlock &loopBlock = **bbItr;
        LiveRange *enteringRange =
          ls.lis->findEnteringRange(li, &loopBlock);
        if (enteringRange != 0) {
          copyRange(*enteringRange);
        }
      }
    } 

    void renameInside() {
      for (std::vector<MachineInstr*>::iterator iItr = loopInstrs.begin(),
                                                iEnd = loopInstrs.end();
           iItr != iEnd; ++iItr) {
        MachineInstr &instr = **iItr;
        for (unsigned i = 0; i < instr.getNumOperands(); ++i) {
          MachineOperand &mop = instr.getOperand(i);
          if (mop.isReg() && mop.getReg() == li.reg) {
            mop.setReg(getNewLI()->reg);
          }
        }
      }
    }

  };

  void LoopSplitter::getAnalysisUsage(AnalysisUsage &au) const {
    au.addRequired<MachineDominatorTree>();
    au.addPreserved<MachineDominatorTree>();
    au.addRequired<MachineLoopInfo>();
    au.addPreserved<MachineLoopInfo>();
    au.addPreserved<RegisterCoalescer>();
    au.addPreserved<CalculateSpillWeights>();
    au.addPreserved<LiveStacks>();
    au.addRequired<SlotIndexes>();
    au.addPreserved<SlotIndexes>();
    au.addRequired<LiveIntervals>();
    au.addPreserved<LiveIntervals>();
    MachineFunctionPass::getAnalysisUsage(au);
  }

  bool LoopSplitter::runOnMachineFunction(MachineFunction &fn) {

    mf = &fn;
    mri = &mf->getRegInfo();
    tii = mf->getTarget().getInstrInfo();
    tri = mf->getTarget().getRegisterInfo();
    sis = &getAnalysis<SlotIndexes>();
    lis = &getAnalysis<LiveIntervals>();
    mli = &getAnalysis<MachineLoopInfo>();
    mdt = &getAnalysis<MachineDominatorTree>();

    fqn = mf->getFunction()->getParent()->getModuleIdentifier() + "." +
      mf->getFunction()->getName().str();

    dbgs() << "Splitting " << mf->getFunction()->getName() << ".";

    dumpOddTerminators();

//      dbgs() << "----------------------------------------\n";
//      lis->dump();
//      dbgs() << "----------------------------------------\n";
       
//     std::deque<MachineLoop*> loops;
//     std::copy(mli->begin(), mli->end(), std::back_inserter(loops));
//     dbgs() << "Loops:\n";
//     while (!loops.empty()) {
//       MachineLoop &loop = *loops.front();
//       loops.pop_front();
//       std::copy(loop.begin(), loop.end(), std::back_inserter(loops));

//       dumpLoopInfo(loop);
//     }
    
    //lis->dump();
    //exit(0);

    // Setup initial intervals.
    for (LiveIntervals::iterator liItr = lis->begin(), liEnd = lis->end();
         liItr != liEnd; ++liItr) {
      LiveInterval *li = liItr->second;

      if (TargetRegisterInfo::isVirtualRegister(li->reg) &&
          !lis->intervalIsInOneMBB(*li)) {
        intervals.push_back(li);
      }
    }

    processIntervals();

    intervals.clear();

//     dbgs() << "----------------------------------------\n";
//     lis->dump();
//     dbgs() << "----------------------------------------\n";

    dumpOddTerminators();

    //exit(1);

    return false;
  }

  void LoopSplitter::releaseMemory() {
    fqn.clear();
    intervals.clear();
    loopRangeMap.clear();
  }

  void LoopSplitter::dumpOddTerminators() {
    for (MachineFunction::iterator bbItr = mf->begin(), bbEnd = mf->end();
         bbItr != bbEnd; ++bbItr) {
      MachineBasicBlock *mbb = &*bbItr;
      MachineBasicBlock *a = 0, *b = 0;
      SmallVector<MachineOperand, 4> c;
      if (tii->AnalyzeBranch(*mbb, a, b, c)) {
        dbgs() << "MBB#" << mbb->getNumber() << " has multiway terminator.\n";
        dbgs() << "  Terminators:\n";
        for (MachineBasicBlock::iterator iItr = mbb->begin(), iEnd = mbb->end();
             iItr != iEnd; ++iItr) {
          MachineInstr *instr= &*iItr;
          dbgs() << "    " << *instr << "";
        }
        dbgs() << "\n  Listed successors: [ ";
        for (MachineBasicBlock::succ_iterator sItr = mbb->succ_begin(), sEnd = mbb->succ_end();
             sItr != sEnd; ++sItr) {
          MachineBasicBlock *succMBB = *sItr;
          dbgs() << succMBB->getNumber() << " ";
        }
        dbgs() << "]\n\n";
      }
    }
  }

  void LoopSplitter::dumpLoopInfo(MachineLoop &loop) {
    MachineBasicBlock &headerBlock = *loop.getHeader();
    typedef SmallVector<MachineLoop::Edge, 8> ExitEdgesList;
    ExitEdgesList exitEdges;
    loop.getExitEdges(exitEdges);

    dbgs() << "  Header: BB#" << headerBlock.getNumber() << ", Contains: [ ";
    for (std::vector<MachineBasicBlock*>::const_iterator
           subBlockItr = loop.getBlocks().begin(),
           subBlockEnd = loop.getBlocks().end();
         subBlockItr != subBlockEnd; ++subBlockItr) {
      MachineBasicBlock &subBlock = **subBlockItr;
      dbgs() << "BB#" << subBlock.getNumber() << " ";
    }
    dbgs() << "], Exit edges: [ ";
    for (ExitEdgesList::iterator exitEdgeItr = exitEdges.begin(),
                                 exitEdgeEnd = exitEdges.end();
         exitEdgeItr != exitEdgeEnd; ++exitEdgeItr) {
      MachineLoop::Edge &exitEdge = *exitEdgeItr;
      dbgs() << "(MBB#" << exitEdge.first->getNumber()
             << ", MBB#" << exitEdge.second->getNumber() << ") ";
    }
    dbgs() << "], Sub-Loop Headers: [ ";
    for (MachineLoop::iterator subLoopItr = loop.begin(),
                               subLoopEnd = loop.end();
         subLoopItr != subLoopEnd; ++subLoopItr) {
      MachineLoop &subLoop = **subLoopItr;
      MachineBasicBlock &subLoopBlock = *subLoop.getHeader();
      dbgs() << "BB#" << subLoopBlock.getNumber() << " ";
    }
    dbgs() << "]\n";
  }

  void LoopSplitter::updateTerminators(MachineBasicBlock &mbb) {
    mbb.updateTerminator();

    for (MachineBasicBlock::iterator miItr = mbb.begin(), miEnd = mbb.end();
         miItr != miEnd; ++miItr) {
      if (lis->isNotInMIMap(miItr)) {
        lis->InsertMachineInstrInMaps(miItr);
      }
    }
  }

  bool LoopSplitter::canInsertPreHeader(MachineLoop &loop) {
    MachineBasicBlock *header = loop.getHeader();
    MachineBasicBlock *a = 0, *b = 0;
    SmallVector<MachineOperand, 4> c;

    for (MachineBasicBlock::pred_iterator pbItr = header->pred_begin(),
                                          pbEnd = header->pred_end();
         pbItr != pbEnd; ++pbItr) {
      MachineBasicBlock *predBlock = *pbItr;
      if (!!tii->AnalyzeBranch(*predBlock, a, b, c)) {
        return false;
      }
    }

    MachineFunction::iterator headerItr(header);
    if (headerItr == mf->begin())
      return true;
    MachineBasicBlock *headerLayoutPred = llvm::prior(headerItr);
    assert(headerLayoutPred != 0 && "Header should have layout pred.");

    return (!tii->AnalyzeBranch(*headerLayoutPred, a, b, c));
  }

  MachineBasicBlock& LoopSplitter::insertPreHeader(MachineLoop &loop) {
    assert(loop.getLoopPreheader() == 0 && "Loop already has preheader.");

    MachineBasicBlock &header = *loop.getHeader();

    // Save the preds - we'll need to update them once we insert the preheader.
    typedef std::set<MachineBasicBlock*> HeaderPreds;
    HeaderPreds headerPreds;

    for (MachineBasicBlock::pred_iterator predItr = header.pred_begin(),
                                          predEnd = header.pred_end();
         predItr != predEnd; ++predItr) {
      if (!loop.contains(*predItr))
        headerPreds.insert(*predItr);
    }

    assert(!headerPreds.empty() && "No predecessors for header?");

    //dbgs() << fqn << " MBB#" << header.getNumber() << " inserting preheader...";

    MachineBasicBlock *preHeader =
      mf->CreateMachineBasicBlock(header.getBasicBlock());

    assert(preHeader != 0 && "Failed to create pre-header.");

    mf->insert(header, preHeader);

    for (HeaderPreds::iterator hpItr = headerPreds.begin(),
                               hpEnd = headerPreds.end(); 
         hpItr != hpEnd; ++hpItr) {
      assert(*hpItr != 0 && "How'd a null predecessor get into this set?");
      MachineBasicBlock &hp = **hpItr;
      hp.ReplaceUsesOfBlockWith(&header, preHeader);
    }
    preHeader->addSuccessor(&header);

    MachineBasicBlock *oldLayoutPred =
      llvm::prior(MachineFunction::iterator(preHeader));
    if (oldLayoutPred != 0) {
      updateTerminators(*oldLayoutPred);
    }

    lis->InsertMBBInMaps(preHeader);

    if (MachineLoop *parentLoop = loop.getParentLoop()) {
      assert(parentLoop->getHeader() != loop.getHeader() &&
             "Parent loop has same header?");
      parentLoop->addBasicBlockToLoop(preHeader, mli->getBase());

      // Invalidate all parent loop ranges.
      while (parentLoop != 0) {
        loopRangeMap.erase(parentLoop);
        parentLoop = parentLoop->getParentLoop();
      }
    }

    for (LiveIntervals::iterator liItr = lis->begin(),
                                 liEnd = lis->end();
         liItr != liEnd; ++liItr) {
      LiveInterval &li = *liItr->second;

      // Is this safe for physregs?
      // TargetRegisterInfo::isPhysicalRegister(li.reg) ||
      if (!lis->isLiveInToMBB(li, &header))
        continue;

      if (lis->isLiveInToMBB(li, preHeader)) {
        assert(lis->isLiveOutOfMBB(li, preHeader) &&
               "Range terminates in newly added preheader?");
        continue;
      }

      bool insertRange = false;

      for (MachineBasicBlock::pred_iterator predItr = preHeader->pred_begin(),
                                            predEnd = preHeader->pred_end();
           predItr != predEnd; ++predItr) {
        MachineBasicBlock *predMBB = *predItr;
        if (lis->isLiveOutOfMBB(li, predMBB)) {
          insertRange = true;
          break;
        }
      }

      if (!insertRange)
        continue;

      SlotIndex newDefIdx = lis->getMBBStartIdx(preHeader);
      assert(lis->getInstructionFromIndex(newDefIdx) == 0 &&
             "PHI def index points at actual instruction.");
      VNInfo *newVal = li.getNextValue(newDefIdx, 0, lis->getVNInfoAllocator());
      li.addRange(LiveRange(lis->getMBBStartIdx(preHeader),
                            lis->getMBBEndIdx(preHeader),
                            newVal));
    }


    //dbgs() << "Dumping SlotIndexes:\n";
    //sis->dump();

    //dbgs() << "done. (Added MBB#" << preHeader->getNumber() << ")\n";

    return *preHeader;
  }

  bool LoopSplitter::isCriticalEdge(MachineLoop::Edge &edge) {
    assert(edge.first->succ_size() > 1 && "Non-sensical edge.");
    if (edge.second->pred_size() > 1)
      return true;
    return false;
  }

  bool LoopSplitter::canSplitEdge(MachineLoop::Edge &edge) {
    MachineFunction::iterator outBlockItr(edge.second);
    if (outBlockItr == mf->begin())
      return true;
    MachineBasicBlock *outBlockLayoutPred = llvm::prior(outBlockItr);
    assert(outBlockLayoutPred != 0 && "Should have a layout pred if out!=begin.");
    MachineBasicBlock *a = 0, *b = 0;
    SmallVector<MachineOperand, 4> c;
    return (!tii->AnalyzeBranch(*outBlockLayoutPred, a, b, c) &&
            !tii->AnalyzeBranch(*edge.first, a, b, c));
  }

  MachineBasicBlock& LoopSplitter::splitEdge(MachineLoop::Edge &edge,
                                             MachineLoop &loop) {

    MachineBasicBlock &inBlock = *edge.first;
    MachineBasicBlock &outBlock = *edge.second;

    assert((inBlock.succ_size() > 1) && (outBlock.pred_size() > 1) &&
           "Splitting non-critical edge?");

    //dbgs() << fqn << " Splitting edge (MBB#" << inBlock.getNumber()
    //       << " -> MBB#" << outBlock.getNumber() << ")...";

    MachineBasicBlock *splitBlock =
      mf->CreateMachineBasicBlock();

    assert(splitBlock != 0 && "Failed to create split block.");

    mf->insert(&outBlock, splitBlock);

    inBlock.ReplaceUsesOfBlockWith(&outBlock, splitBlock);
    splitBlock->addSuccessor(&outBlock);

    MachineBasicBlock *oldLayoutPred =
      llvm::prior(MachineFunction::iterator(splitBlock));
    if (oldLayoutPred != 0) {
      updateTerminators(*oldLayoutPred);
    }

    lis->InsertMBBInMaps(splitBlock);

    loopRangeMap.erase(&loop);

    MachineLoop *splitParentLoop = loop.getParentLoop();
    while (splitParentLoop != 0 &&
           !splitParentLoop->contains(&outBlock)) {
      splitParentLoop = splitParentLoop->getParentLoop();
    }

    if (splitParentLoop != 0) {
      assert(splitParentLoop->contains(&loop) &&
             "Split-block parent doesn't contain original loop?");
      splitParentLoop->addBasicBlockToLoop(splitBlock, mli->getBase());
      
      // Invalidate all parent loop ranges.
      while (splitParentLoop != 0) {
        loopRangeMap.erase(splitParentLoop);
        splitParentLoop = splitParentLoop->getParentLoop();
      }
    }


    for (LiveIntervals::iterator liItr = lis->begin(),
                                 liEnd = lis->end();
         liItr != liEnd; ++liItr) {
      LiveInterval &li = *liItr->second;
      bool intersects = lis->isLiveOutOfMBB(li, &inBlock) &&
                       lis->isLiveInToMBB(li, &outBlock);
      if (lis->isLiveInToMBB(li, splitBlock)) {
        if (!intersects) {
          li.removeRange(lis->getMBBStartIdx(splitBlock),
                         lis->getMBBEndIdx(splitBlock), true);
        }
      } else if (intersects) {
        SlotIndex newDefIdx = lis->getMBBStartIdx(splitBlock);
        assert(lis->getInstructionFromIndex(newDefIdx) == 0 &&
               "PHI def index points at actual instruction.");
        VNInfo *newVal = li.getNextValue(newDefIdx, 0,
                                         lis->getVNInfoAllocator());
        li.addRange(LiveRange(lis->getMBBStartIdx(splitBlock),
                              lis->getMBBEndIdx(splitBlock),
                              newVal));
      }
    }

    //dbgs() << "done. (Added MBB#" << splitBlock->getNumber() << ")\n";

    return *splitBlock;
  }

  LoopSplitter::LoopRanges& LoopSplitter::getLoopRanges(MachineLoop &loop) {
    typedef std::set<MachineBasicBlock*, StartSlotComparator> LoopMBBSet;
    LoopRangeMap::iterator lrItr = loopRangeMap.find(&loop);
    if (lrItr == loopRangeMap.end()) {
      LoopMBBSet loopMBBs((StartSlotComparator(*lis))); 
      std::copy(loop.block_begin(), loop.block_end(),
                std::inserter(loopMBBs, loopMBBs.begin()));

      assert(!loopMBBs.empty() && "No blocks in loop?");

      LoopRanges &loopRanges = loopRangeMap[&loop];
      assert(loopRanges.empty() && "Loop encountered but not processed?");
      SlotIndex oldEnd = lis->getMBBEndIdx(*loopMBBs.begin());
      loopRanges.push_back(
        std::make_pair(lis->getMBBStartIdx(*loopMBBs.begin()),
                       lis->getInvalidIndex()));
      for (LoopMBBSet::iterator curBlockItr = llvm::next(loopMBBs.begin()),
                                curBlockEnd = loopMBBs.end();
           curBlockItr != curBlockEnd; ++curBlockItr) {
        SlotIndex newStart = lis->getMBBStartIdx(*curBlockItr);
        if (newStart != oldEnd) {
          loopRanges.back().second = oldEnd;
          loopRanges.push_back(std::make_pair(newStart,
                                              lis->getInvalidIndex()));
        }
        oldEnd = lis->getMBBEndIdx(*curBlockItr);
      }

      loopRanges.back().second =
        lis->getMBBEndIdx(*llvm::prior(loopMBBs.end()));

      return loopRanges;
    }
    return lrItr->second;
  }

  std::pair<bool, LoopSplitter::SlotPair> LoopSplitter::getLoopSubRange(
                                                           const LiveRange &lr,
                                                           MachineLoop &loop) {
    LoopRanges &loopRanges = getLoopRanges(loop);
    LoopRanges::iterator lrItr = loopRanges.begin(),
                         lrEnd = loopRanges.end();
    while (lrItr != lrEnd && lr.start >= lrItr->second) {
      ++lrItr;
    }

    if (lrItr == lrEnd) {
      SlotIndex invalid = lis->getInvalidIndex();
      return std::make_pair(false, SlotPair(invalid, invalid));
    }

    SlotIndex srStart(lr.start < lrItr->first ? lrItr->first : lr.start);
    SlotIndex srEnd(lr.end > lrItr->second ? lrItr->second : lr.end);

    return std::make_pair(true, SlotPair(srStart, srEnd));      
  }

  void LoopSplitter::dumpLoopRanges(MachineLoop &loop) {
    LoopRanges &loopRanges = getLoopRanges(loop);
    dbgs() << "For loop MBB#" << loop.getHeader()->getNumber() << ", subranges are: [ ";
    for (LoopRanges::iterator lrItr = loopRanges.begin(), lrEnd = loopRanges.end();
         lrItr != lrEnd; ++lrItr) {
      dbgs() << "[" << lrItr->first << ", " << lrItr->second << ") ";
    }
    dbgs() << "]\n";
  }

  void LoopSplitter::processHeader(LoopSplit &split) {
    MachineBasicBlock &header = *split.getLoop().getHeader();
    //dbgs() << "  Processing loop header BB#" << header.getNumber() << "\n";

    if (!lis->isLiveInToMBB(split.getLI(), &header))
      return; // Not live in, but nothing wrong so far.

    MachineBasicBlock *preHeader = split.getLoop().getLoopPreheader();
    if (!preHeader) {

      if (!canInsertPreHeader(split.getLoop())) {
        split.invalidate();
        return; // Couldn't insert a pre-header. Bail on this interval.
      }

      for (MachineBasicBlock::pred_iterator predItr = header.pred_begin(),
                                            predEnd = header.pred_end();
           predItr != predEnd; ++predItr) {
        if (lis->isLiveOutOfMBB(split.getLI(), *predItr)) {
          split.splitIncoming();
          break;
        }
      }
    } else if (lis->isLiveOutOfMBB(split.getLI(), preHeader)) {
      split.splitIncoming();
    }
  }

  void LoopSplitter::processLoopExits(LoopSplit &split) {
    typedef SmallVector<MachineLoop::Edge, 8> ExitEdgesList;
    ExitEdgesList exitEdges;
    split.getLoop().getExitEdges(exitEdges);

    //dbgs() << "  Processing loop exits:\n";

    for (ExitEdgesList::iterator exitEdgeItr = exitEdges.begin(),
                                 exitEdgeEnd = exitEdges.end();
         exitEdgeItr != exitEdgeEnd; ++exitEdgeItr) {
      MachineLoop::Edge exitEdge = *exitEdgeItr;

      LiveRange *outRange =
        split.getLI().getLiveRangeContaining(lis->getMBBStartIdx(exitEdge.second));

      if (outRange != 0) {
        if (isCriticalEdge(exitEdge) && !canSplitEdge(exitEdge)) {
          split.invalidate();
          return;
        }

        split.splitOutgoing(exitEdge);
      }
    }
  }

  void LoopSplitter::processLoopUses(LoopSplit &split) {
    std::set<MachineInstr*> processed;

    for (MachineRegisterInfo::reg_iterator
           rItr = mri->reg_begin(split.getLI().reg),
           rEnd = mri->reg_end();
      rItr != rEnd; ++rItr) {
      MachineInstr &instr = *rItr;
      if (split.getLoop().contains(&instr) && processed.count(&instr) == 0) {
        split.addLoopInstr(&instr);
        processed.insert(&instr);
      }
    }

    //dbgs() << "  Rewriting reg" << li.reg << " to reg" << newLI->reg
    //       << " in blocks [ ";
    //dbgs() << "]\n";
  }

  bool LoopSplitter::splitOverLoop(LiveInterval &li, MachineLoop &loop) {
    assert(TargetRegisterInfo::isVirtualRegister(li.reg) &&
           "Attempt to split physical register.");

    LoopSplit split(*this, li, loop);
    processHeader(split);
    if (split.isValid())
      processLoopExits(split);
    if (split.isValid())
      processLoopUses(split);
    if (split.isValid() /* && split.isWorthwhile() */) {
      split.apply();
      DEBUG(dbgs() << "Success.\n");
      return true;
    }
    DEBUG(dbgs() << "Failed.\n");
    return false;
  }

  void LoopSplitter::processInterval(LiveInterval &li) {
    std::deque<MachineLoop*> loops;
    std::copy(mli->begin(), mli->end(), std::back_inserter(loops));

    while (!loops.empty()) {
      MachineLoop &loop = *loops.front();
      loops.pop_front();
      DEBUG(
        dbgs() << fqn << " reg" << li.reg << " " << li.weight << " BB#"
               << loop.getHeader()->getNumber() << " ";
      );
      if (!splitOverLoop(li, loop)) {
        // Couldn't split over outer loop, schedule sub-loops to be checked.
	std::copy(loop.begin(), loop.end(), std::back_inserter(loops));
      }
    }
  }

  void LoopSplitter::processIntervals() {
    while (!intervals.empty()) {
      LiveInterval &li = *intervals.front();
      intervals.pop_front();

      assert(!lis->intervalIsInOneMBB(li) &&
             "Single interval in process worklist.");

      processInterval(li);      
    }
  }

}
