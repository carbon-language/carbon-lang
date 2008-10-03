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
// allocation see the following papers: 
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
// Author: Lang Hames
// Email: lhames@gmail.com
//
//===----------------------------------------------------------------------===//

// TODO:
// 
// * Use of std::set in constructPBQPProblem destroys allocation order preference.
// Switch to an order preserving container.
// 
// * Coalescing support.

#define DEBUG_TYPE "regalloc"

#include "PBQP.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <map>
#include <set>
#include <vector>
#include <limits>

using namespace llvm;

static RegisterRegAlloc
registerPBQPRepAlloc("pbqp", "  PBQP register allocator",
                     createPBQPRegisterAllocator);


namespace {

  //!
  //! PBQP based allocators solve the register allocation problem by mapping
  //! register allocation problems to Partitioned Boolean Quadratic
  //! Programming problems.
  class VISIBILITY_HIDDEN PBQPRegAlloc : public MachineFunctionPass {
  public:

    static char ID;
    
    //! Construct a PBQP register allocator.
    PBQPRegAlloc() : MachineFunctionPass((intptr_t)&ID) {}

    //! Return the pass name.
    virtual const char* getPassName() const throw() {
      return "PBQP Register Allocator";
    }

    //! PBQP analysis usage.
    virtual void getAnalysisUsage(AnalysisUsage &au) const {
      au.addRequired<LiveIntervals>();
      au.addRequired<MachineLoopInfo>();
      MachineFunctionPass::getAnalysisUsage(au);
    }

    //! Perform register allocation
    virtual bool runOnMachineFunction(MachineFunction &MF);

  private:
    typedef std::map<const LiveInterval*, unsigned> LI2NodeMap;
    typedef std::vector<const LiveInterval*> Node2LIMap;
    typedef std::vector<unsigned> AllowedSet;
    typedef std::vector<AllowedSet> AllowedSetMap;
    typedef std::set<unsigned> IgnoreSet;

    MachineFunction *mf;
    const TargetMachine *tm;
    const TargetRegisterInfo *tri;
    const TargetInstrInfo *tii;
    const MachineLoopInfo *loopInfo;
    MachineRegisterInfo *mri;

    LiveIntervals *li;
    VirtRegMap *vrm;

    LI2NodeMap li2Node;
    Node2LIMap node2LI;
    AllowedSetMap allowedSets;
    IgnoreSet ignoreSet;

    //! Builds a PBQP cost vector.
    template <typename Container>
    PBQPVector* buildCostVector(const Container &allowed,
                                PBQPNum spillCost) const;

    //! \brief Builds a PBQP interfernce matrix.
    //!
    //! @return Either a pointer to a non-zero PBQP matrix representing the
    //!         allocation option costs, or a null pointer for a zero matrix.
    //!
    //! Expects allowed sets for two interfering LiveIntervals. These allowed
    //! sets should contain only allocable registers from the LiveInterval's
    //! register class, with any interfering pre-colored registers removed.
    template <typename Container>
    PBQPMatrix* buildInterferenceMatrix(const Container &allowed1,
                                        const Container &allowed2) const;

    //!
    //! Expects allowed sets for two potentially coalescable LiveIntervals,
    //! and an estimated benefit due to coalescing. The allowed sets should
    //! contain only allocable registers from the LiveInterval's register
    //! classes, with any interfering pre-colored registers removed.
    template <typename Container>
    PBQPMatrix* buildCoalescingMatrix(const Container &allowed1,
                                      const Container &allowed2,
                                      PBQPNum cBenefit) const;

    //! \brief Helper functior for constructInitialPBQPProblem().
    //!
    //! This function iterates over the Function we are about to allocate for
    //! and computes spill costs.
    void calcSpillCosts();

    //! \brief Scans the MachineFunction being allocated to find coalescing
    //  opportunities.
    void findCoalescingOpportunities();

    //! \brief Constructs a PBQP problem representation of the register
    //! allocation problem for this function.
    //!
    //! @return a PBQP solver object for the register allocation problem.
    pbqp* constructPBQPProblem();

    //! \brief Given a solved PBQP problem maps this solution back to a register
    //! assignment.
    bool mapPBQPToRegAlloc(pbqp *problem); 

  };

  char PBQPRegAlloc::ID = 0;
}


template <typename Container>
PBQPVector* PBQPRegAlloc::buildCostVector(const Container &allowed,
                                          PBQPNum spillCost) const {

  // Allocate vector. Additional element (0th) used for spill option
  PBQPVector *v = new PBQPVector(allowed.size() + 1);

  (*v)[0] = spillCost;

  return v;
}

template <typename Container>
PBQPMatrix* PBQPRegAlloc::buildInterferenceMatrix(
      const Container &allowed1, const Container &allowed2) const {

  typedef typename Container::const_iterator ContainerIterator;

  // Construct a PBQP matrix representing the cost of allocation options. The
  // rows and columns correspond to the allocation options for the two live
  // intervals.  Elements will be infinite where corresponding registers alias,
  // since we cannot allocate aliasing registers to interfering live intervals.
  // All other elements (non-aliasing combinations) will have zero cost. Note
  // that the spill option (element 0,0) has zero cost, since we can allocate
  // both intervals to memory safely (the cost for each individual allocation
  // to memory is accounted for by the cost vectors for each live interval).
  PBQPMatrix *m = new PBQPMatrix(allowed1.size() + 1, allowed2.size() + 1);
 
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
  for (ContainerIterator a1Itr = allowed1.begin(), a1End = allowed1.end();
       a1Itr != a1End; ++a1Itr) {

    // Column index, starts at 1 as for row index.
    unsigned ci = 1;
    unsigned reg1 = *a1Itr;

    for (ContainerIterator a2Itr = allowed2.begin(), a2End = allowed2.end();
         a2Itr != a2End; ++a2Itr) {

      unsigned reg2 = *a2Itr;

      // If the row/column regs are identical or alias insert an infinity.
      if ((reg1 == reg2) || tri->areAliases(reg1, reg2)) {
        (*m)[ri][ci] = std::numeric_limits<PBQPNum>::infinity();
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

void PBQPRegAlloc::calcSpillCosts() {

  // Calculate the spill cost for each live interval by iterating over the
  // function counting loads and stores, with loop depth taken into account.
  for (MachineFunction::const_iterator bbItr = mf->begin(), bbEnd = mf->end();
       bbItr != bbEnd; ++bbItr) {

    const MachineBasicBlock *mbb = &*bbItr;
    float loopDepth = loopInfo->getLoopDepth(mbb);

    for (MachineBasicBlock::const_iterator
         iItr = mbb->begin(), iEnd = mbb->end(); iItr != iEnd; ++iItr) {

      const MachineInstr *instr = &*iItr;

      for (unsigned opNo = 0; opNo < instr->getNumOperands(); ++opNo) {

        const MachineOperand &mo = instr->getOperand(opNo);

        // We're not interested in non-registers...
        if (!mo.isReg())
          continue;
 
        unsigned moReg = mo.getReg();

        // ...Or invalid registers...
        if (moReg == 0)
          continue;

        // ...Or physical registers...
        if (TargetRegisterInfo::isPhysicalRegister(moReg)) 
          continue;

        assert ((mo.isUse() || mo.isDef()) &&
                "Not a use, not a def, what is it?");

	//... Just the virtual registers. We treat loads and stores as equal.
	li->getInterval(moReg).weight += powf(10.0f, loopDepth);
      }

    }

  }

}

pbqp* PBQPRegAlloc::constructPBQPProblem() {

  typedef std::vector<const LiveInterval*> LIVector;
  typedef std::set<unsigned> RegSet;

  // These will store the physical & virtual intervals, respectively.
  LIVector physIntervals, virtIntervals;

  // Start by clearing the old node <-> live interval mappings & allowed sets
  li2Node.clear();
  node2LI.clear();
  allowedSets.clear();

  // Iterate over intervals classifying them as physical or virtual, and
  // constructing live interval <-> node number mappings.
  for (LiveIntervals::iterator itr = li->begin(), end = li->end();
       itr != end; ++itr) {

    if (itr->second->getNumValNums() != 0) {
      DOUT << "Live range has " << itr->second->getNumValNums() << ": " << itr->second << "\n";
    }

    if (TargetRegisterInfo::isPhysicalRegister(itr->first)) {
      physIntervals.push_back(itr->second);
      mri->setPhysRegUsed(itr->second->reg);
    }
    else {

      // If we've allocated this virtual register interval a stack slot on a
      // previous round then it's not an allocation candidate
      if (ignoreSet.find(itr->first) != ignoreSet.end())
        continue;

      li2Node[itr->second] = node2LI.size();
      node2LI.push_back(itr->second);
      virtIntervals.push_back(itr->second);
    }
  }

  // Early out if there's no regs to allocate for.
  if (virtIntervals.empty())
    return 0;

  // Construct a PBQP solver for this problem
  pbqp *solver = alloc_pbqp(virtIntervals.size());

  // Resize allowedSets container appropriately.
  allowedSets.resize(virtIntervals.size());

  // Iterate over virtual register intervals to compute allowed sets...
  for (unsigned node = 0; node < node2LI.size(); ++node) {

    // Grab pointers to the interval and its register class.
    const LiveInterval *li = node2LI[node];
    const TargetRegisterClass *liRC = mri->getRegClass(li->reg);
    
    // Start by assuming all allocable registers in the class are allowed...
    RegSet liAllowed(liRC->allocation_order_begin(*mf),
                     liRC->allocation_order_end(*mf));

    // If this range is non-empty then eliminate the physical registers which
    // overlap with this range, along with all their aliases.
    if (!li->empty()) {
      for (LIVector::iterator pItr = physIntervals.begin(),
           pEnd = physIntervals.end(); pItr != pEnd; ++pItr) {

        if (li->overlaps(**pItr)) {

          unsigned pReg = (*pItr)->reg;

          // Remove the overlapping reg...
          liAllowed.erase(pReg);

          const unsigned *aliasItr = tri->getAliasSet(pReg);

          if (aliasItr != 0) {
            // ...and its aliases.
            for (; *aliasItr != 0; ++aliasItr) {
              liAllowed.erase(*aliasItr);
            }

          }
        
        }

      }

    }

    // Copy the allowed set into a member vector for use when constructing cost
    // vectors & matrices, and mapping PBQP solutions back to assignments.
    allowedSets[node] = AllowedSet(liAllowed.begin(), liAllowed.end());

    // Set the spill cost to the interval weight, or epsilon if the
    // interval weight is zero
    PBQPNum spillCost = (li->weight != 0.0) ? 
        li->weight : std::numeric_limits<PBQPNum>::min();

    // Build a cost vector for this interval.
    add_pbqp_nodecosts(solver, node,
                       buildCostVector(allowedSets[node], spillCost));

  }

  // Now add the cost matrices...
  for (unsigned node1 = 0; node1 < node2LI.size(); ++node1) {
      
    const LiveInterval *li = node2LI[node1];

    if (li->empty())
      continue;
 
    // Test for live range overlaps and insert interference matrices.
    for (unsigned node2 = node1 + 1; node2 < node2LI.size(); ++node2) {
      const LiveInterval *li2 = node2LI[node2];

      if (li2->empty())
        continue;

      if (li->overlaps(*li2)) {
        PBQPMatrix *m =
          buildInterferenceMatrix(allowedSets[node1], allowedSets[node2]);

        if (m != 0) {
          add_pbqp_edgecosts(solver, node1, node2, m);
          delete m;
        }
      }
    }
  }

  // We're done, PBQP problem constructed - return it.
  return solver; 
}

bool PBQPRegAlloc::mapPBQPToRegAlloc(pbqp *problem) {
  
  // Set to true if we have any spills
  bool anotherRoundNeeded = false;

  // Clear the existing allocation.
  vrm->clearAllVirt();
  
  // Iterate over the nodes mapping the PBQP solution to a register assignment.
  for (unsigned node = 0; node < node2LI.size(); ++node) {
    unsigned symReg = node2LI[node]->reg,
             allocSelection = get_pbqp_solution(problem, node);

    // If the PBQP solution is non-zero it's a physical register...
    if (allocSelection != 0) {
      // Get the physical reg, subtracting 1 to account for the spill option.
      unsigned physReg = allowedSets[node][allocSelection - 1];

      // Add to the virt reg map and update the used phys regs.
      vrm->assignVirt2Phys(symReg, physReg);
      mri->setPhysRegUsed(physReg);
    }
    // ...Otherwise it's a spill.
    else {

      // Make sure we ignore this virtual reg on the next round
      // of allocation
      ignoreSet.insert(node2LI[node]->reg);

      float SSWeight;

      // Insert spill ranges for this live range
      SmallVector<LiveInterval*, 8> spillIs;
      std::vector<LiveInterval*> newSpills =
        li->addIntervalsForSpills(*node2LI[node], spillIs, loopInfo, *vrm,
                                  SSWeight);

      // We need another round if spill intervals were added.
      anotherRoundNeeded |= !newSpills.empty();
    }
  }

  return !anotherRoundNeeded;
}

bool PBQPRegAlloc::runOnMachineFunction(MachineFunction &MF) {
  
  mf = &MF;
  tm = &mf->getTarget();
  tri = tm->getRegisterInfo();
  mri = &mf->getRegInfo();

  li = &getAnalysis<LiveIntervals>();
  loopInfo = &getAnalysis<MachineLoopInfo>();

  std::auto_ptr<VirtRegMap> vrmAutoPtr(new VirtRegMap(*mf));
  vrm = vrmAutoPtr.get();

  // Allocator main loop:
  // 
  // * Map current regalloc problem to a PBQP problem
  // * Solve the PBQP problem
  // * Map the solution back to a register allocation
  // * Spill if necessary
  // 
  // This process is continued till no more spills are generated.

  bool regallocComplete = false;
  
  // Calculate spill costs for intervals
  calcSpillCosts();

  while (!regallocComplete) {
    pbqp *problem = constructPBQPProblem();
   
    // Fast out if there's no problem to solve.
    if (problem == 0)
      return true;
 
    solve_pbqp(problem);
   
    regallocComplete = mapPBQPToRegAlloc(problem);

    free_pbqp(problem); 
  }

  ignoreSet.clear();

  std::auto_ptr<Spiller> spiller(createSpiller());

  spiller->runOnMachineFunction(*mf, *vrm);
    
  return true; 
}

FunctionPass* llvm::createPBQPRegisterAllocator() {
  return new PBQPRegAlloc();
}


#undef DEBUG_TYPE
