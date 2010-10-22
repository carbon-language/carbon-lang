//===-- RegAllocBase.h - basic regalloc interface and driver --*- C++ -*---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RegAllocBase class, which is the skeleton of a basic
// register allocation algorithm and interface for extending it. It provides the
// building blocks on which to construct other experimental allocators and test
// the validity of two principles:
// 
// - If virtual and physical register liveness is modeled using intervals, then
// on-the-fly interference checking is cheap. Furthermore, interferences can be
// lazily cached and reused.
// 
// - Register allocation complexity, and generated code performance is
// determined by the effectiveness of live range splitting rather than optimal
// coloring.
//
// Following the first principle, interfering checking revolves around the
// LiveIntervalUnion data structure.
//
// To fulfill the second principle, the basic allocator provides a driver for
// incremental splitting. It essentially punts on the problem of register
// coloring, instead driving the assignment of virtual to physical registers by
// the cost of splitting. The basic allocator allows for heuristic reassignment
// of registers, if a more sophisticated allocator chooses to do that.
//
// This framework provides a way to engineer the compile time vs. code
// quality trade-off without relying a particular theoretical solver.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_REGALLOCBASE
#define LLVM_CODEGEN_REGALLOCBASE

#include "LiveIntervalUnion.h"
#include "VirtRegMap.h"
#include "llvm/CodeGen/LiveIntervalAnalysis.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/ADT/OwningPtr.h"
#include <vector>
#include <queue>

namespace llvm {

class VirtRegMap;

/// RegAllocBase provides the register allocation driver and interface that can
/// be extended to add interesting heuristics.
///
/// More sophisticated allocators must override the selectOrSplit() method to
/// implement live range splitting and must specify a comparator to determine
/// register assignment priority. LessSpillWeightPriority is provided as a
/// standard comparator.
class RegAllocBase {
protected:
  typedef SmallVector<LiveInterval*, 4> LiveVirtRegs;
  typedef LiveVirtRegs::iterator LVRIter;
  
  // Array of LiveIntervalUnions indexed by physical register.
  class LIUArray {
    unsigned nRegs_;
    OwningArrayPtr<LiveIntervalUnion> array_;
  public:
    LIUArray(): nRegs_(0) {}

    unsigned numRegs() const { return nRegs_; }

    void init(unsigned nRegs);

    void clear();
    
    LiveIntervalUnion& operator[](unsigned physReg) {
      assert(physReg <  nRegs_ && "physReg out of bounds");
      return array_[physReg];
    }
  };
  
  const TargetRegisterInfo *tri_;
  VirtRegMap *vrm_;
  LiveIntervals *lis_;
  LIUArray physReg2liu_;

  RegAllocBase(): tri_(0), vrm_(0), lis_(0) {}

  // A RegAlloc pass should call this before allocatePhysRegs.
  void init(const TargetRegisterInfo &tri, VirtRegMap &vrm, LiveIntervals &lis);

  // The top-level driver. Specialize with the comparator that determines the
  // priority of assigning live virtual registers. The output is a VirtRegMap
  // that us updated with physical register assignments.
  template<typename LICompare>
  void allocatePhysRegs(LICompare liCompare);

  // A RegAlloc pass should override this to provide the allocation heuristics.
  // Each call must guarantee forward progess by returning an available
  // PhysReg or new set of split LiveVirtRegs. It is up to the splitter to
  // converge quickly toward fully spilled live ranges.
  virtual unsigned selectOrSplit(LiveInterval &lvr,
                                 LiveVirtRegs &splitLVRs) = 0;

  // A RegAlloc pass should call this when PassManager releases its memory.
  virtual void releaseMemory();

  // Helper for checking interference between a live virtual register and a
  // physical register, including all its register aliases.
  bool checkPhysRegInterference(LiveIntervalUnion::Query &query, unsigned preg);
  
private:
  template<typename PQ>
  void seedLiveVirtRegs(PQ &lvrQ);
};

// Heuristic that determines the priority of assigning virtual to physical
// registers. The main impact of the heuristic is expected to be compile time.
// The default is to simply compare spill weights.
struct LessSpillWeightPriority
  : public std::binary_function<LiveInterval,LiveInterval, bool> {
  bool operator()(const LiveInterval *left, const LiveInterval *right) const {
    return left->weight < right->weight;
  }
};

// Visit all the live virtual registers. If they are already assigned to a
// physical register, unify them with the corresponding LiveIntervalUnion,
// otherwise push them on the priority queue for later assignment.
template<typename PQ>
void RegAllocBase::seedLiveVirtRegs(PQ &lvrQ) {
  for (LiveIntervals::iterator liItr = lis_->begin(), liEnd = lis_->end();
       liItr != liEnd; ++liItr) {
    unsigned reg = liItr->first;
    LiveInterval &li = *liItr->second;
    if (TargetRegisterInfo::isPhysicalRegister(reg)) {
      physReg2liu_[reg].unify(li);
    }
    else {
      lvrQ.push(&li);
    }
  }
}

// Top-level driver to manage the queue of unassigned LiveVirtRegs and call the
// selectOrSplit implementation.
template<typename LICompare>
void RegAllocBase::allocatePhysRegs(LICompare liCompare) {
  typedef std::priority_queue
    <LiveInterval*, std::vector<LiveInterval*>, LICompare> LiveVirtRegQueue;

  LiveVirtRegQueue lvrQ(liCompare);
  seedLiveVirtRegs(lvrQ);
  while (!lvrQ.empty()) {
    LiveInterval *lvr = lvrQ.top();
    lvrQ.pop();
    LiveVirtRegs splitLVRs;
    unsigned availablePhysReg = selectOrSplit(*lvr, splitLVRs);
    if (availablePhysReg) {
      assert(splitLVRs.empty() && "inconsistent splitting");
      assert(!vrm_->hasPhys(lvr->reg) && "duplicate vreg in interval unions");
      vrm_->assignVirt2Phys(lvr->reg, availablePhysReg);
      physReg2liu_[availablePhysReg].unify(*lvr);
    }
    else {
      for (LVRIter lvrI = splitLVRs.begin(), lvrEnd = splitLVRs.end();
           lvrI != lvrEnd; ++lvrI ) {
        assert(TargetRegisterInfo::isVirtualRegister((*lvrI)->reg) &&
               "expect split value in virtual register");
        lvrQ.push(*lvrI);
      }
    }
  }
}

} // end namespace llvm

#endif // !defined(LLVM_CODEGEN_REGALLOCBASE)
