//===------ RegAllocPBQP.cpp ---- PBQP Register Allocator -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a Partitioned Boolean Quadratic Programming (PBQP) based
// register allocator for LLVM. This allocator works by constructing a PBQP
// problem representing the register allocation problem under consideration,
// solving this using a PBQP solver, and mapping the solution back to a
// register assignment. If any variables are selected for spilling then spill
// code is inserted and the process repeated.
//
// The PBQP solver (pbqp.c) provided for this allocator uses a heuristic tuned
// for register allocation. For more information on PBQP for register
// allocation, see the following papers:
//
//   (1) Hames, L. and Scholz, B. 2006. Nearly optimal register allocation with
//   PBQP. In Proceedings of the 7th Joint Modular Languages Conference
//   (JMLC'06). LNCS, vol. 4228. Springer, New York, NY, USA. 346-361.
//
//   (2) Scholz, B., Eckstein, E. 2002. Register allocation for irregular
//   architectures. In Proceedings of the Joint Conference on Languages,
//   Compilers and Tools for Embedded Systems (LCTES'02), ACM Press, New York,
//   NY, USA, 139-148.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"

#include "RenderMachineFunction.h"
#include "Splitter.h"
#include "VirtRegMap.h"
#include "VirtRegRewriter.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/RegAllocPBQP.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PBQP/HeuristicSolver.h"
#include "llvm/CodeGen/PBQP/Graph.h"
#include "llvm/CodeGen/PBQP/Heuristics/Briggs.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include <limits>
#include <memory>
#include <set>
#include <vector>

namespace llvm {

using namespace PBQP;
  using namespace PBQP::Heuristics;

static RegisterRegAlloc
registerPBQPRepAlloc("pbqp", "PBQP register allocator",
                       llvm::createPBQPRegisterAllocator);

static cl::opt<bool>
pbqpCoalescing("pbqp-coalescing",
                cl::desc("Attempt coalescing during PBQP register allocation."),
                cl::init(false), cl::Hidden);

static cl::opt<bool>
pbqpBuilder("pbqp-builder",
                cl::desc("Use new builder system."),
                cl::init(false), cl::Hidden);


static cl::opt<bool>
pbqpPreSplitting("pbqp-pre-splitting",
                 cl::desc("Pre-splite before PBQP register allocation."),
                 cl::init(false), cl::Hidden);

char RegAllocPBQP::ID = 0;

unsigned PBQPRAProblem::getVRegForNode(PBQP::Graph::ConstNodeItr node) const {
  Node2VReg::const_iterator vregItr = node2VReg.find(node);
  assert(vregItr != node2VReg.end() && "No vreg for node.");
  return vregItr->second;
}

PBQP::Graph::NodeItr PBQPRAProblem::getNodeForVReg(unsigned vreg) const {
  VReg2Node::const_iterator nodeItr = vreg2Node.find(vreg);
  assert(nodeItr != vreg2Node.end() && "No node for vreg.");
  return nodeItr->second;
  
}

const PBQPRAProblem::AllowedSet&
  PBQPRAProblem::getAllowedSet(unsigned vreg) const {
  AllowedSetMap::const_iterator allowedSetItr = allowedSets.find(vreg);
  assert(allowedSetItr != allowedSets.end() && "No pregs for vreg.");
  const AllowedSet &allowedSet = allowedSetItr->second;
  return allowedSet;
}

unsigned PBQPRAProblem::getPRegForOption(unsigned vreg, unsigned option) const {
  assert(isPRegOption(vreg, option) && "Not a preg option.");

  const AllowedSet& allowedSet = getAllowedSet(vreg);
  assert(option <= allowedSet.size() && "Option outside allowed set.");
  return allowedSet[option - 1];
}

std::auto_ptr<PBQPRAProblem> PBQPBuilder::build(
                                             MachineFunction *mf,
                                             const LiveIntervals *lis,
                                             const RegSet &vregs) {

  typedef std::vector<const LiveInterval*> LIVector;

  MachineRegisterInfo *mri = &mf->getRegInfo();
  const TargetRegisterInfo *tri = mf->getTarget().getRegisterInfo();  

  std::auto_ptr<PBQPRAProblem> p(new PBQPRAProblem());
  PBQP::Graph &g = p->getGraph();
  RegSet pregs;

  // Collect the set of preg intervals, record that they're used in the MF.
  for (LiveIntervals::const_iterator itr = lis->begin(), end = lis->end();
       itr != end; ++itr) {
    if (TargetRegisterInfo::isPhysicalRegister(itr->first)) {
      pregs.insert(itr->first);
      mri->setPhysRegUsed(itr->first);
    }
  }

  BitVector reservedRegs = tri->getReservedRegs(*mf);

  // Iterate over vregs. 
  for (RegSet::const_iterator vregItr = vregs.begin(), vregEnd = vregs.end();
       vregItr != vregEnd; ++vregItr) {
    unsigned vreg = *vregItr;
    const TargetRegisterClass *trc = mri->getRegClass(vreg);
    const LiveInterval *vregLI = &lis->getInterval(vreg);

    // Compute an initial allowed set for the current vreg.
    typedef std::vector<unsigned> VRAllowed;
    VRAllowed vrAllowed;
    for (TargetRegisterClass::iterator aoItr = trc->allocation_order_begin(*mf),
                                       aoEnd = trc->allocation_order_end(*mf);
         aoItr != aoEnd; ++aoItr) {
      unsigned preg = *aoItr;
      if (!reservedRegs.test(preg)) {
        vrAllowed.push_back(preg);
      }
    }

    // Remove any physical registers which overlap.
    for (RegSet::const_iterator pregItr = pregs.begin(),
                                pregEnd = pregs.end();
         pregItr != pregEnd; ++pregItr) {
      unsigned preg = *pregItr;
      const LiveInterval *pregLI = &lis->getInterval(preg);

      if (pregLI->empty())
        continue;

      if (!vregLI->overlaps(*pregLI))
        continue;

      // Remove the register from the allowed set.
      VRAllowed::iterator eraseItr =
        std::find(vrAllowed.begin(), vrAllowed.end(), preg);

      if (eraseItr != vrAllowed.end()) {
        vrAllowed.erase(eraseItr);
      }

      // Also remove any aliases.
      const unsigned *aliasItr = tri->getAliasSet(preg);
      if (aliasItr != 0) {
        for (; *aliasItr != 0; ++aliasItr) {
          VRAllowed::iterator eraseItr =
            std::find(vrAllowed.begin(), vrAllowed.end(), *aliasItr);

          if (eraseItr != vrAllowed.end()) {
            vrAllowed.erase(eraseItr);
          }
        }
      }
    }

    // Construct the node.
    PBQP::Graph::NodeItr node = 
      g.addNode(PBQP::Vector(vrAllowed.size() + 1, 0));

    // Record the mapping and allowed set in the problem.
    p->recordVReg(vreg, node, vrAllowed.begin(), vrAllowed.end());

    PBQP::PBQPNum spillCost = (vregLI->weight != 0.0) ?
        vregLI->weight : std::numeric_limits<PBQP::PBQPNum>::min();

    addSpillCosts(g.getNodeCosts(node), spillCost);
  }

  for (RegSet::iterator vr1Itr = vregs.begin(), vrEnd = vregs.end();
         vr1Itr != vrEnd; ++vr1Itr) {
    unsigned vr1 = *vr1Itr;
    const LiveInterval &l1 = lis->getInterval(vr1);
    const PBQPRAProblem::AllowedSet &vr1Allowed = p->getAllowedSet(vr1);

    for (RegSet::iterator vr2Itr = llvm::next(vr1Itr);
         vr2Itr != vrEnd; ++vr2Itr) {
      unsigned vr2 = *vr2Itr;
      const LiveInterval &l2 = lis->getInterval(vr2);
      const PBQPRAProblem::AllowedSet &vr2Allowed = p->getAllowedSet(vr2);

      assert(!l2.empty() && "Empty interval in vreg set?");
      if (l1.overlaps(l2)) {
        PBQP::Graph::EdgeItr edge =
          g.addEdge(p->getNodeForVReg(vr1), p->getNodeForVReg(vr2),
                    PBQP::Matrix(vr1Allowed.size()+1, vr2Allowed.size()+1, 0));

        addInterferenceCosts(g.getEdgeCosts(edge), vr1Allowed, vr2Allowed, tri);
      }
    }
  }

  return p;
}

void PBQPBuilder::addSpillCosts(PBQP::Vector &costVec,
                                PBQP::PBQPNum spillCost) {
  costVec[0] = spillCost;
}

void PBQPBuilder::addInterferenceCosts(PBQP::Matrix &costMat,
                                       const PBQPRAProblem::AllowedSet &vr1Allowed,
                                       const PBQPRAProblem::AllowedSet &vr2Allowed,
                                       const TargetRegisterInfo *tri) {
  assert(costMat.getRows() == vr1Allowed.size() + 1 && "Matrix height mismatch.");
  assert(costMat.getCols() == vr2Allowed.size() + 1 && "Matrix width mismatch.");

  for (unsigned i = 0; i < vr1Allowed.size(); ++i) {
    unsigned preg1 = vr1Allowed[i];

    for (unsigned j = 0; j < vr2Allowed.size(); ++j) {
      unsigned preg2 = vr2Allowed[j];

      if (tri->regsOverlap(preg1, preg2)) {
        costMat[i + 1][j + 1] = std::numeric_limits<PBQP::PBQPNum>::infinity();
      }
    }
  }
}



void RegAllocPBQP::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<SlotIndexes>();
  au.addPreserved<SlotIndexes>();
  au.addRequired<LiveIntervals>();
  //au.addRequiredID(SplitCriticalEdgesID);
  au.addRequired<RegisterCoalescer>();
  au.addRequired<CalculateSpillWeights>();
  au.addRequired<LiveStacks>();
  au.addPreserved<LiveStacks>();
  au.addRequired<MachineLoopInfo>();
  au.addPreserved<MachineLoopInfo>();
  if (pbqpPreSplitting)
    au.addRequired<LoopSplitter>();
  au.addRequired<VirtRegMap>();
  au.addRequired<RenderMachineFunction>();
  MachineFunctionPass::getAnalysisUsage(au);
}

template <typename RegContainer>
PBQP::Vector RegAllocPBQP::buildCostVector(unsigned vReg,
                                           const RegContainer &allowed,
                                           const CoalesceMap &coalesces,
                                           PBQP::PBQPNum spillCost) const {

  typedef typename RegContainer::const_iterator AllowedItr;

  // Allocate vector. Additional element (0th) used for spill option
  PBQP::Vector v(allowed.size() + 1, 0);

  v[0] = spillCost;

  // Iterate over the allowed registers inserting coalesce benefits if there
  // are any.
  unsigned ai = 0;
  for (AllowedItr itr = allowed.begin(), end = allowed.end();
       itr != end; ++itr, ++ai) {

    unsigned pReg = *itr;

    CoalesceMap::const_iterator cmItr =
      coalesces.find(RegPair(vReg, pReg));

    // No coalesce - on to the next preg.
    if (cmItr == coalesces.end())
      continue;

    // We have a coalesce - insert the benefit.
    v[ai + 1] = -cmItr->second;
  }

  return v;
}

template <typename RegContainer>
PBQP::Matrix* RegAllocPBQP::buildInterferenceMatrix(
      const RegContainer &allowed1, const RegContainer &allowed2) const {

  typedef typename RegContainer::const_iterator RegContainerIterator;

  // Construct a PBQP matrix representing the cost of allocation options. The
  // rows and columns correspond to the allocation options for the two live
  // intervals.  Elements will be infinite where corresponding registers alias,
  // since we cannot allocate aliasing registers to interfering live intervals.
  // All other elements (non-aliasing combinations) will have zero cost. Note
  // that the spill option (element 0,0) has zero cost, since we can allocate
  // both intervals to memory safely (the cost for each individual allocation
  // to memory is accounted for by the cost vectors for each live interval).
  PBQP::Matrix *m =
    new PBQP::Matrix(allowed1.size() + 1, allowed2.size() + 1, 0);

  // Assume this is a zero matrix until proven otherwise.  Zero matrices occur
  // between interfering live ranges with non-overlapping register sets (e.g.
  // non-overlapping reg classes, or disjoint sets of allowed regs within the
  // same class). The term "overlapping" is used advisedly: sets which do not
  // intersect, but contain registers which alias, will have non-zero matrices.
  // We optimize zero matrices away to improve solver speed.
  bool isZeroMatrix = true;


  // Row index. Starts at 1, since the 0th row is for the spill option, which
  // is always zero.
  unsigned ri = 1;

  // Iterate over allowed sets, insert infinities where required.
  for (RegContainerIterator a1Itr = allowed1.begin(), a1End = allowed1.end();
       a1Itr != a1End; ++a1Itr) {

    // Column index, starts at 1 as for row index.
    unsigned ci = 1;
    unsigned reg1 = *a1Itr;

    for (RegContainerIterator a2Itr = allowed2.begin(), a2End = allowed2.end();
         a2Itr != a2End; ++a2Itr) {

      unsigned reg2 = *a2Itr;

      // If the row/column regs are identical or alias insert an infinity.
      if (tri->regsOverlap(reg1, reg2)) {
        (*m)[ri][ci] = std::numeric_limits<PBQP::PBQPNum>::infinity();
        isZeroMatrix = false;
      }

      ++ci;
    }

    ++ri;
  }

  // If this turns out to be a zero matrix...
  if (isZeroMatrix) {
    // free it and return null.
    delete m;
    return 0;
  }

  // ...otherwise return the cost matrix.
  return m;
}

template <typename RegContainer>
PBQP::Matrix* RegAllocPBQP::buildCoalescingMatrix(
      const RegContainer &allowed1, const RegContainer &allowed2,
      PBQP::PBQPNum cBenefit) const {

  typedef typename RegContainer::const_iterator RegContainerIterator;

  // Construct a PBQP Matrix representing the benefits of coalescing. As with
  // interference matrices the rows and columns represent allowed registers
  // for the LiveIntervals which are (potentially) to be coalesced. The amount
  // -cBenefit will be placed in any element representing the same register
  // for both intervals.
  PBQP::Matrix *m =
    new PBQP::Matrix(allowed1.size() + 1, allowed2.size() + 1, 0);

  // Reset costs to zero.
  m->reset(0);

  // Assume the matrix is zero till proven otherwise. Zero matrices will be
  // optimized away as in the interference case.
  bool isZeroMatrix = true;

  // Row index. Starts at 1, since the 0th row is for the spill option, which
  // is always zero.
  unsigned ri = 1;

  // Iterate over the allowed sets, insert coalescing benefits where
  // appropriate.
  for (RegContainerIterator a1Itr = allowed1.begin(), a1End = allowed1.end();
       a1Itr != a1End; ++a1Itr) {

    // Column index, starts at 1 as for row index.
    unsigned ci = 1;
    unsigned reg1 = *a1Itr;

    for (RegContainerIterator a2Itr = allowed2.begin(), a2End = allowed2.end();
         a2Itr != a2End; ++a2Itr) {

      // If the row and column represent the same register insert a beneficial
      // cost to preference this allocation - it would allow us to eliminate a
      // move instruction.
      if (reg1 == *a2Itr) {
        (*m)[ri][ci] = -cBenefit;
        isZeroMatrix = false;
      }

      ++ci;
    }

    ++ri;
  }

  // If this turns out to be a zero matrix...
  if (isZeroMatrix) {
    // ...free it and return null.
    delete m;
    return 0;
  }

  return m;
}

RegAllocPBQP::CoalesceMap RegAllocPBQP::findCoalesces() {

  typedef MachineFunction::const_iterator MFIterator;
  typedef MachineBasicBlock::const_iterator MBBIterator;
  typedef LiveInterval::const_vni_iterator VNIIterator;

  CoalesceMap coalescesFound;

  // To find coalesces we need to iterate over the function looking for
  // copy instructions.
  for (MFIterator bbItr = mf->begin(), bbEnd = mf->end();
       bbItr != bbEnd; ++bbItr) {

    const MachineBasicBlock *mbb = &*bbItr;

    for (MBBIterator iItr = mbb->begin(), iEnd = mbb->end();
         iItr != iEnd; ++iItr) {

      const MachineInstr *instr = &*iItr;

      // If this isn't a copy then continue to the next instruction.
      if (!instr->isCopy())
        continue;

      unsigned srcReg = instr->getOperand(1).getReg();
      unsigned dstReg = instr->getOperand(0).getReg();

      // If the registers are already the same our job is nice and easy.
      if (dstReg == srcReg)
        continue;

      bool srcRegIsPhysical = TargetRegisterInfo::isPhysicalRegister(srcReg),
           dstRegIsPhysical = TargetRegisterInfo::isPhysicalRegister(dstReg);

      // If both registers are physical then we can't coalesce.
      if (srcRegIsPhysical && dstRegIsPhysical)
        continue;

      // If it's a copy that includes two virtual register but the source and
      // destination classes differ then we can't coalesce.
      if (!srcRegIsPhysical && !dstRegIsPhysical &&
          mri->getRegClass(srcReg) != mri->getRegClass(dstReg))
        continue;

      // If one is physical and one is virtual, check that the physical is
      // allocatable in the class of the virtual.
      if (srcRegIsPhysical && !dstRegIsPhysical) {
        const TargetRegisterClass *dstRegClass = mri->getRegClass(dstReg);
        if (std::find(dstRegClass->allocation_order_begin(*mf),
                      dstRegClass->allocation_order_end(*mf), srcReg) ==
            dstRegClass->allocation_order_end(*mf))
          continue;
      }
      if (!srcRegIsPhysical && dstRegIsPhysical) {
        const TargetRegisterClass *srcRegClass = mri->getRegClass(srcReg);
        if (std::find(srcRegClass->allocation_order_begin(*mf),
                      srcRegClass->allocation_order_end(*mf), dstReg) ==
            srcRegClass->allocation_order_end(*mf))
          continue;
      }

      // If we've made it here we have a copy with compatible register classes.
      // We can probably coalesce, but we need to consider overlap.
      const LiveInterval *srcLI = &lis->getInterval(srcReg),
                         *dstLI = &lis->getInterval(dstReg);

      if (srcLI->overlaps(*dstLI)) {
        // Even in the case of an overlap we might still be able to coalesce,
        // but we need to make sure that no definition of either range occurs
        // while the other range is live.

        // Otherwise start by assuming we're ok.
        bool badDef = false;

        // Test all defs of the source range.
        for (VNIIterator
               vniItr = srcLI->vni_begin(), vniEnd = srcLI->vni_end();
               vniItr != vniEnd; ++vniItr) {

          // If we find a poorly defined def we err on the side of caution.
          if (!(*vniItr)->def.isValid()) {
            badDef = true;
            break;
          }

          // If we find a def that kills the coalescing opportunity then
          // record it and break from the loop.
          if (dstLI->liveAt((*vniItr)->def)) {
            badDef = true;
            break;
          }
        }

        // If we have a bad def give up, continue to the next instruction.
        if (badDef)
          continue;

        // Otherwise test definitions of the destination range.
        for (VNIIterator
               vniItr = dstLI->vni_begin(), vniEnd = dstLI->vni_end();
               vniItr != vniEnd; ++vniItr) {

          // We want to make sure we skip the copy instruction itself.
          if ((*vniItr)->getCopy() == instr)
            continue;

          if (!(*vniItr)->def.isValid()) {
            badDef = true;
            break;
          }

          if (srcLI->liveAt((*vniItr)->def)) {
            badDef = true;
            break;
          }
        }

        // As before a bad def we give up and continue to the next instr.
        if (badDef)
          continue;
      }

      // If we make it to here then either the ranges didn't overlap, or they
      // did, but none of their definitions would prevent us from coalescing.
      // We're good to go with the coalesce.

      float cBenefit = std::pow(10.0f, (float)loopInfo->getLoopDepth(mbb)) / 5.0;

      coalescesFound[RegPair(srcReg, dstReg)] = cBenefit;
      coalescesFound[RegPair(dstReg, srcReg)] = cBenefit;
    }

  }

  return coalescesFound;
}

void RegAllocPBQP::findVRegIntervalsToAlloc() {

  // Iterate over all live ranges.
  for (LiveIntervals::iterator itr = lis->begin(), end = lis->end();
       itr != end; ++itr) {

    // Ignore physical ones.
    if (TargetRegisterInfo::isPhysicalRegister(itr->first))
      continue;

    LiveInterval *li = itr->second;

    // If this live interval is non-empty we will use pbqp to allocate it.
    // Empty intervals we allocate in a simple post-processing stage in
    // finalizeAlloc.
    if (!li->empty()) {
      vregsToAlloc.insert(li->reg);
    }
    else {
      emptyIntervalVRegs.insert(li->reg);
    }
  }
}

PBQP::Graph RegAllocPBQP::constructPBQPProblem() {

  typedef std::vector<const LiveInterval*> LIVector;
  typedef std::vector<unsigned> RegVector;

  // This will store the physical intervals for easy reference.
  LIVector physIntervals;

  // Start by clearing the old node <-> live interval mappings & allowed sets
  li2Node.clear();
  node2LI.clear();
  allowedSets.clear();

  // Populate physIntervals, update preg use:
  for (LiveIntervals::iterator itr = lis->begin(), end = lis->end();
       itr != end; ++itr) {

    if (TargetRegisterInfo::isPhysicalRegister(itr->first)) {
      physIntervals.push_back(itr->second);
      mri->setPhysRegUsed(itr->second->reg);
    }
  }

  // Iterate over vreg intervals, construct live interval <-> node number
  //  mappings.
  for (RegSet::const_iterator itr = vregsToAlloc.begin(),
                              end = vregsToAlloc.end();
       itr != end; ++itr) {
    const LiveInterval *li = &lis->getInterval(*itr);

    li2Node[li] = node2LI.size();
    node2LI.push_back(li);
  }

  // Get the set of potential coalesces.
  CoalesceMap coalesces;

  if (pbqpCoalescing) {
    coalesces = findCoalesces();
  }

  // Construct a PBQP solver for this problem
  PBQP::Graph problem;
  problemNodes.resize(vregsToAlloc.size());

  // Resize allowedSets container appropriately.
  allowedSets.resize(vregsToAlloc.size());

  BitVector ReservedRegs = tri->getReservedRegs(*mf);

  // Iterate over virtual register intervals to compute allowed sets...
  for (unsigned node = 0; node < node2LI.size(); ++node) {

    // Grab pointers to the interval and its register class.
    const LiveInterval *li = node2LI[node];
    const TargetRegisterClass *liRC = mri->getRegClass(li->reg);

    // Start by assuming all allocable registers in the class are allowed...
    RegVector liAllowed;
    TargetRegisterClass::iterator aob = liRC->allocation_order_begin(*mf);
    TargetRegisterClass::iterator aoe = liRC->allocation_order_end(*mf);
    for (TargetRegisterClass::iterator it = aob; it != aoe; ++it)
      if (!ReservedRegs.test(*it))
        liAllowed.push_back(*it);

    // Eliminate the physical registers which overlap with this range, along
    // with all their aliases.
    for (LIVector::iterator pItr = physIntervals.begin(),
       pEnd = physIntervals.end(); pItr != pEnd; ++pItr) {

      if (!li->overlaps(**pItr))
        continue;

      unsigned pReg = (*pItr)->reg;

      // If we get here then the live intervals overlap, but we're still ok
      // if they're coalescable.
      if (coalesces.find(RegPair(li->reg, pReg)) != coalesces.end()) {
        DEBUG(dbgs() << "CoalescingOverride: (" << li->reg << ", " << pReg << ")\n");
        continue;
      }

      // If we get here then we have a genuine exclusion.

      // Remove the overlapping reg...
      RegVector::iterator eraseItr =
        std::find(liAllowed.begin(), liAllowed.end(), pReg);

      if (eraseItr != liAllowed.end())
        liAllowed.erase(eraseItr);

      const unsigned *aliasItr = tri->getAliasSet(pReg);

      if (aliasItr != 0) {
        // ...and its aliases.
        for (; *aliasItr != 0; ++aliasItr) {
          RegVector::iterator eraseItr =
            std::find(liAllowed.begin(), liAllowed.end(), *aliasItr);

          if (eraseItr != liAllowed.end()) {
            liAllowed.erase(eraseItr);
          }
        }
      }
    }

    // Copy the allowed set into a member vector for use when constructing cost
    // vectors & matrices, and mapping PBQP solutions back to assignments.
    allowedSets[node] = AllowedSet(liAllowed.begin(), liAllowed.end());

    // Set the spill cost to the interval weight, or epsilon if the
    // interval weight is zero
    PBQP::PBQPNum spillCost = (li->weight != 0.0) ?
        li->weight : std::numeric_limits<PBQP::PBQPNum>::min();

    // Build a cost vector for this interval.
    problemNodes[node] =
      problem.addNode(
        buildCostVector(li->reg, allowedSets[node], coalesces, spillCost));

  }


  // Now add the cost matrices...
  for (unsigned node1 = 0; node1 < node2LI.size(); ++node1) {
    const LiveInterval *li = node2LI[node1];

    // Test for live range overlaps and insert interference matrices.
    for (unsigned node2 = node1 + 1; node2 < node2LI.size(); ++node2) {
      const LiveInterval *li2 = node2LI[node2];

      CoalesceMap::const_iterator cmItr =
        coalesces.find(RegPair(li->reg, li2->reg));

      PBQP::Matrix *m = 0;

      if (cmItr != coalesces.end()) {
        m = buildCoalescingMatrix(allowedSets[node1], allowedSets[node2],
                                  cmItr->second);
      }
      else if (li->overlaps(*li2)) {
        m = buildInterferenceMatrix(allowedSets[node1], allowedSets[node2]);
      }

      if (m != 0) {
        problem.addEdge(problemNodes[node1],
                        problemNodes[node2],
                        *m);

        delete m;
      }
    }
  }

  assert(problem.getNumNodes() == allowedSets.size());
/*
  std::cerr << "Allocating for " << problem.getNumNodes() << " nodes, "
            << problem.getNumEdges() << " edges.\n";

  problem.printDot(std::cerr);
*/
  // We're done, PBQP problem constructed - return it.
  return problem;
}

void RegAllocPBQP::addStackInterval(const LiveInterval *spilled,
                                    MachineRegisterInfo* mri) {
  int stackSlot = vrm->getStackSlot(spilled->reg);

  if (stackSlot == VirtRegMap::NO_STACK_SLOT)
    return;

  const TargetRegisterClass *RC = mri->getRegClass(spilled->reg);
  LiveInterval &stackInterval = lss->getOrCreateInterval(stackSlot, RC);

  VNInfo *vni;
  if (stackInterval.getNumValNums() != 0)
    vni = stackInterval.getValNumInfo(0);
  else
    vni = stackInterval.getNextValue(
      SlotIndex(), 0, false, lss->getVNInfoAllocator());

  LiveInterval &rhsInterval = lis->getInterval(spilled->reg);
  stackInterval.MergeRangesInAsValue(rhsInterval, vni);
}

bool RegAllocPBQP::mapPBQPToRegAlloc(const PBQP::Solution &solution) {

  // Set to true if we have any spills
  bool anotherRoundNeeded = false;

  // Clear the existing allocation.
  vrm->clearAllVirt();

  // Iterate over the nodes mapping the PBQP solution to a register assignment.
  for (unsigned node = 0; node < node2LI.size(); ++node) {
    unsigned virtReg = node2LI[node]->reg,
             allocSelection = solution.getSelection(problemNodes[node]);


    // If the PBQP solution is non-zero it's a physical register...
    if (allocSelection != 0) {
      // Get the physical reg, subtracting 1 to account for the spill option.
      unsigned physReg = allowedSets[node][allocSelection - 1];

      DEBUG(dbgs() << "VREG " << virtReg << " -> "
            << tri->getName(physReg) << " (Option: " << allocSelection << ")\n");

      assert(physReg != 0);

      // Add to the virt reg map and update the used phys regs.
      vrm->assignVirt2Phys(virtReg, physReg);
    }
    // ...Otherwise it's a spill.
    else {

      // Make sure we ignore this virtual reg on the next round
      // of allocation
      vregsToAlloc.erase(virtReg);

      // Insert spill ranges for this live range
      const LiveInterval *spillInterval = node2LI[node];
      double oldSpillWeight = spillInterval->weight;
      SmallVector<LiveInterval*, 8> spillIs;
      rmf->rememberUseDefs(spillInterval);
      std::vector<LiveInterval*> newSpills =
        lis->addIntervalsForSpills(*spillInterval, spillIs, loopInfo, *vrm);
      addStackInterval(spillInterval, mri);
      rmf->rememberSpills(spillInterval, newSpills);

      (void) oldSpillWeight;
      DEBUG(dbgs() << "VREG " << virtReg << " -> SPILLED (Option: 0, Cost: "
                   << oldSpillWeight << ", New vregs: ");

      // Copy any newly inserted live intervals into the list of regs to
      // allocate.
      for (std::vector<LiveInterval*>::const_iterator
           itr = newSpills.begin(), end = newSpills.end();
           itr != end; ++itr) {

        assert(!(*itr)->empty() && "Empty spill range.");

        DEBUG(dbgs() << (*itr)->reg << " ");

        vregsToAlloc.insert((*itr)->reg);
      }

      DEBUG(dbgs() << ")\n");

      // We need another round if spill intervals were added.
      anotherRoundNeeded |= !newSpills.empty();
    }
  }

  return !anotherRoundNeeded;
}

bool RegAllocPBQP::mapPBQPToRegAlloc2(const PBQPRAProblem &problem,
                                      const PBQP::Solution &solution) {
  // Set to true if we have any spills
  bool anotherRoundNeeded = false;

  // Clear the existing allocation.
  vrm->clearAllVirt();

  const PBQP::Graph &g = problem.getGraph();
  // Iterate over the nodes mapping the PBQP solution to a register
  // assignment.
  for (PBQP::Graph::ConstNodeItr node = g.nodesBegin(),
                                 nodeEnd = g.nodesEnd();
       node != nodeEnd; ++node) {
    unsigned vreg = problem.getVRegForNode(node);
    unsigned alloc = solution.getSelection(node);

    if (problem.isPRegOption(vreg, alloc)) {
      unsigned preg = problem.getPRegForOption(vreg, alloc);    
      DEBUG(dbgs() << "VREG " << vreg << " -> " << tri->getName(preg) << "\n");
      assert(preg != 0 && "Invalid preg selected.");
      vrm->assignVirt2Phys(vreg, preg);      
    } else if (problem.isSpillOption(vreg, alloc)) {
      vregsToAlloc.erase(vreg);
      const LiveInterval* spillInterval = &lis->getInterval(vreg);
      double oldWeight = spillInterval->weight;
      SmallVector<LiveInterval*, 8> spillIs;
      rmf->rememberUseDefs(spillInterval);
      std::vector<LiveInterval*> newSpills =
        lis->addIntervalsForSpills(*spillInterval, spillIs, loopInfo, *vrm);
      addStackInterval(spillInterval, mri);
      rmf->rememberSpills(spillInterval, newSpills);

      (void) oldWeight;
      DEBUG(dbgs() << "VREG " << vreg << " -> SPILLED (Cost: "
                   << oldWeight << ", New vregs: ");

      // Copy any newly inserted live intervals into the list of regs to
      // allocate.
      for (std::vector<LiveInterval*>::const_iterator
           itr = newSpills.begin(), end = newSpills.end();
           itr != end; ++itr) {
        assert(!(*itr)->empty() && "Empty spill range.");
        DEBUG(dbgs() << (*itr)->reg << " ");
        vregsToAlloc.insert((*itr)->reg);
      }

      DEBUG(dbgs() << ")\n");

      // We need another round if spill intervals were added.
      anotherRoundNeeded |= !newSpills.empty();
    } else {
      assert(false && "Unknown allocation option.");
    }
  }

  return !anotherRoundNeeded;
}


void RegAllocPBQP::finalizeAlloc() const {
  typedef LiveIntervals::iterator LIIterator;
  typedef LiveInterval::Ranges::const_iterator LRIterator;

  // First allocate registers for the empty intervals.
  for (RegSet::const_iterator
         itr = emptyIntervalVRegs.begin(), end = emptyIntervalVRegs.end();
         itr != end; ++itr) {
    LiveInterval *li = &lis->getInterval(*itr);

    unsigned physReg = vrm->getRegAllocPref(li->reg);

    if (physReg == 0) {
      const TargetRegisterClass *liRC = mri->getRegClass(li->reg);
      physReg = *liRC->allocation_order_begin(*mf);
    }

    vrm->assignVirt2Phys(li->reg, physReg);
  }

  // Finally iterate over the basic blocks to compute and set the live-in sets.
  SmallVector<MachineBasicBlock*, 8> liveInMBBs;
  MachineBasicBlock *entryMBB = &*mf->begin();

  for (LIIterator liItr = lis->begin(), liEnd = lis->end();
       liItr != liEnd; ++liItr) {

    const LiveInterval *li = liItr->second;
    unsigned reg = 0;

    // Get the physical register for this interval
    if (TargetRegisterInfo::isPhysicalRegister(li->reg)) {
      reg = li->reg;
    }
    else if (vrm->isAssignedReg(li->reg)) {
      reg = vrm->getPhys(li->reg);
    }
    else {
      // Ranges which are assigned a stack slot only are ignored.
      continue;
    }

    if (reg == 0) {
      // Filter out zero regs - they're for intervals that were spilled.
      continue;
    }

    // Iterate over the ranges of the current interval...
    for (LRIterator lrItr = li->begin(), lrEnd = li->end();
         lrItr != lrEnd; ++lrItr) {

      // Find the set of basic blocks which this range is live into...
      if (lis->findLiveInMBBs(lrItr->start, lrItr->end,  liveInMBBs)) {
        // And add the physreg for this interval to their live-in sets.
        for (unsigned i = 0; i < liveInMBBs.size(); ++i) {
          if (liveInMBBs[i] != entryMBB) {
            if (!liveInMBBs[i]->isLiveIn(reg)) {
              liveInMBBs[i]->addLiveIn(reg);
            }
          }
        }
        liveInMBBs.clear();
      }
    }
  }

}

bool RegAllocPBQP::runOnMachineFunction(MachineFunction &MF) {

  mf = &MF;
  tm = &mf->getTarget();
  tri = tm->getRegisterInfo();
  tii = tm->getInstrInfo();
  mri = &mf->getRegInfo(); 

  lis = &getAnalysis<LiveIntervals>();
  lss = &getAnalysis<LiveStacks>();
  loopInfo = &getAnalysis<MachineLoopInfo>();
  rmf = &getAnalysis<RenderMachineFunction>();

  vrm = &getAnalysis<VirtRegMap>();


  DEBUG(dbgs() << "PBQP Register Allocating for " << mf->getFunction()->getName() << "\n");

  // Allocator main loop:
  //
  // * Map current regalloc problem to a PBQP problem
  // * Solve the PBQP problem
  // * Map the solution back to a register allocation
  // * Spill if necessary
  //
  // This process is continued till no more spills are generated.

  // Find the vreg intervals in need of allocation.
  findVRegIntervalsToAlloc();

  // If there are non-empty intervals allocate them using pbqp.
  if (!vregsToAlloc.empty()) {

    bool pbqpAllocComplete = false;
    unsigned round = 0;

    if (!pbqpBuilder) {
      while (!pbqpAllocComplete) {
        DEBUG(dbgs() << "  PBQP Regalloc round " << round << ":\n");

        PBQP::Graph problem = constructPBQPProblem();
        PBQP::Solution solution =
          PBQP::HeuristicSolver<PBQP::Heuristics::Briggs>::solve(problem);

        pbqpAllocComplete = mapPBQPToRegAlloc(solution);

        ++round;
      }
    } else {
      while (!pbqpAllocComplete) {
        DEBUG(dbgs() << "  PBQP Regalloc round " << round << ":\n");

        std::auto_ptr<PBQPRAProblem> problem =
          builder->build(mf, lis, vregsToAlloc);
        PBQP::Solution solution =
          HeuristicSolver<Briggs>::solve(problem->getGraph());

        pbqpAllocComplete = mapPBQPToRegAlloc2(*problem, solution);

        ++round;
      }
    }
  }

  // Finalise allocation, allocate empty ranges.
  finalizeAlloc();

  rmf->renderMachineFunction("After PBQP register allocation.", vrm);

  vregsToAlloc.clear();
  emptyIntervalVRegs.clear();
  li2Node.clear();
  node2LI.clear();
  allowedSets.clear();
  problemNodes.clear();

  DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *vrm << "\n");

  // Run rewriter
  std::auto_ptr<VirtRegRewriter> rewriter(createVirtRegRewriter());

  rewriter->runOnMachineFunction(*mf, *vrm, lis);

  return true;
}

FunctionPass* createPBQPRegisterAllocator() {
  return new RegAllocPBQP(std::auto_ptr<PBQPBuilder>(new PBQPBuilder()));
}

}

#undef DEBUG_TYPE
