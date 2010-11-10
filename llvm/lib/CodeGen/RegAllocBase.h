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

#include "llvm/ADT/OwningPtr.h"

namespace llvm {

template<typename T> class SmallVectorImpl;
class TargetRegisterInfo;
class VirtRegMap;
class LiveIntervals;
class Spiller;

// Heuristic that determines the priority of assigning virtual to physical
// registers. The main impact of the heuristic is expected to be compile time.
// The default is to simply compare spill weights.
struct LessSpillWeightPriority
  : public std::binary_function<LiveInterval,LiveInterval, bool> {
  bool operator()(const LiveInterval *left, const LiveInterval *right) const {
    return left->weight < right->weight;
  }
};

// Forward declare a priority queue of live virtual registers. If an
// implementation needs to prioritize by anything other than spill weight, then
// this will become an abstract base class with virtual calls to push/get.
class LiveVirtRegQueue;

/// RegAllocBase provides the register allocation driver and interface that can
/// be extended to add interesting heuristics.
///
/// More sophisticated allocators must override the selectOrSplit() method to
/// implement live range splitting and must specify a comparator to determine
/// register assignment priority. LessSpillWeightPriority is provided as a
/// standard comparator.
class RegAllocBase {
protected:
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

  // Current queries, one per physreg. They must be reinitialized each time we
  // query on a new live virtual register.
  OwningArrayPtr<LiveIntervalUnion::Query> queries_;

  RegAllocBase(): tri_(0), vrm_(0), lis_(0) {}

  virtual ~RegAllocBase() {}

  // A RegAlloc pass should call this before allocatePhysRegs.
  void init(const TargetRegisterInfo &tri, VirtRegMap &vrm, LiveIntervals &lis);

  // The top-level driver. The output is a VirtRegMap that us updated with
  // physical register assignments.
  //
  // If an implementation wants to override the LiveInterval comparator, we
  // should modify this interface to allow passing in an instance derived from
  // LiveVirtRegQueue.
  void allocatePhysRegs();

  // Get a temporary reference to a Spiller instance.
  virtual Spiller &spiller() = 0;
  
  // A RegAlloc pass should override this to provide the allocation heuristics.
  // Each call must guarantee forward progess by returning an available PhysReg
  // or new set of split live virtual registers. It is up to the splitter to
  // converge quickly toward fully spilled live ranges.
  virtual unsigned selectOrSplit(LiveInterval &lvr,
                                 SmallVectorImpl<LiveInterval*> &splitLVRs) = 0;

  // A RegAlloc pass should call this when PassManager releases its memory.
  virtual void releaseMemory();

  // Helper for checking interference between a live virtual register and a
  // physical register, including all its register aliases. If an interference
  // exists, return the interfering register, which may be preg or an alias.
  unsigned checkPhysRegInterference(LiveInterval& lvr, unsigned preg);

  // Helper for spilling all live virtual registers currently unified under preg
  // that interfere with the most recently queried lvr.  Return true if spilling
  // was successful, and append any new spilled/split intervals to splitLVRs.
  bool spillInterferences(unsigned preg,
                          SmallVectorImpl<LiveInterval*> &splitLVRs);

#ifndef NDEBUG
  // Verify each LiveIntervalUnion.
  void verify();
#endif
  
private:
  void seedLiveVirtRegs(LiveVirtRegQueue &lvrQ);

  void spillReg(unsigned reg, SmallVectorImpl<LiveInterval*> &splitLVRs);
};

} // end namespace llvm

#endif // !defined(LLVM_CODEGEN_REGALLOCBASE)
