//===-- RegAllocBasic.cpp - basic register allocator ----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the RABasic function pass, which provides a minimal
// implementation of the basic register allocator.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "regalloc"
#include "RegAllocBase.h"
#include "RenderMachineFunction.h"
#include "Spiller.h"
#include "VirtRegRewriter.h"
#include "llvm/Function.h"
#include "llvm/PassAnalysisSupport.h"
#include "llvm/CodeGen/CalcSpillWeights.h"
#include "llvm/CodeGen/LiveStackAnalysis.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/RegisterCoalescer.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

static RegisterRegAlloc basicRegAlloc("basic", "basic register allocator",
                                      createBasicRegisterAllocator);

namespace {

/// RABasic provides a minimal implementation of the basic register allocation
/// algorithm. It prioritizes live virtual registers by spill weight and spills
/// whenever a register is unavailable. This is not practical in production but
/// provides a useful baseline both for measuring other allocators and comparing
/// the speed of the basic algorithm against other styles of allocators.
class RABasic : public MachineFunctionPass, public RegAllocBase
{
  // context
  MachineFunction *mf_;
  const TargetMachine *tm_;
  MachineRegisterInfo *mri_;

  // analyses
  LiveStacks *ls_;
  RenderMachineFunction *rmf_;

  // state
  std::auto_ptr<Spiller> spiller_;

public:
  RABasic();

  /// Return the pass name.
  virtual const char* getPassName() const {
    return "Basic Register Allocator";
  }

  /// RABasic analysis usage.
  virtual void getAnalysisUsage(AnalysisUsage &au) const;

  virtual void releaseMemory();

  virtual unsigned selectOrSplit(LiveInterval &lvr, LiveVirtRegs &splitLVRs);

  /// Perform register allocation.
  virtual bool runOnMachineFunction(MachineFunction &mf);

  static char ID;
};

char RABasic::ID = 0;

} // end anonymous namespace

// We should not need to publish the initializer as long as no other passes
// require RABasic.
#if 0 // disable INITIALIZE_PASS
INITIALIZE_PASS_BEGIN(RABasic, "basic-regalloc",
                      "Basic Register Allocator", false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(StrongPHIElimination)
INITIALIZE_AG_DEPENDENCY(RegisterCoalescer)
INITIALIZE_PASS_DEPENDENCY(CalculateSpillWeights)
INITIALIZE_PASS_DEPENDENCY(LiveStacks)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_DEPENDENCY(VirtRegMap)
#ifndef NDEBUG
INITIALIZE_PASS_DEPENDENCY(RenderMachineFunction)
#endif
INITIALIZE_PASS_END(RABasic, "basic-regalloc",
                    "Basic Register Allocator", false, false)
#endif // INITIALIZE_PASS

RABasic::RABasic(): MachineFunctionPass(ID) {
  initializeLiveIntervalsPass(*PassRegistry::getPassRegistry());
  initializeSlotIndexesPass(*PassRegistry::getPassRegistry());
  initializeStrongPHIEliminationPass(*PassRegistry::getPassRegistry());
  initializeRegisterCoalescerAnalysisGroup(*PassRegistry::getPassRegistry());
  initializeCalculateSpillWeightsPass(*PassRegistry::getPassRegistry());
  initializeLiveStacksPass(*PassRegistry::getPassRegistry());
  initializeMachineLoopInfoPass(*PassRegistry::getPassRegistry());
  initializeVirtRegMapPass(*PassRegistry::getPassRegistry());
  initializeRenderMachineFunctionPass(*PassRegistry::getPassRegistry());
}

void RABasic::getAnalysisUsage(AnalysisUsage &au) const {
  au.setPreservesCFG();
  au.addRequired<LiveIntervals>();
  au.addPreserved<SlotIndexes>();
  if (StrongPHIElim)
    au.addRequiredID(StrongPHIEliminationID);
  au.addRequiredTransitive<RegisterCoalescer>();
  au.addRequired<CalculateSpillWeights>();
  au.addRequired<LiveStacks>();
  au.addPreserved<LiveStacks>();
  au.addRequired<MachineLoopInfo>();
  au.addPreserved<MachineLoopInfo>();
  au.addRequired<VirtRegMap>();
  au.addPreserved<VirtRegMap>();
  DEBUG(au.addRequired<RenderMachineFunction>());
  MachineFunctionPass::getAnalysisUsage(au);
}

void RABasic::releaseMemory() {
  spiller_.reset(0);
  RegAllocBase::releaseMemory();
}

//===----------------------------------------------------------------------===//
//                         RegAllocBase Implementation
//===----------------------------------------------------------------------===//

// Instantiate a LiveIntervalUnion for each physical register.
void RegAllocBase::LIUArray::init(unsigned nRegs) {
  array_.reset(new LiveIntervalUnion[nRegs]);
  nRegs_ = nRegs;
  for (unsigned pr = 0; pr < nRegs; ++pr) {
    array_[pr].init(pr);
  }
}

void RegAllocBase::init(const TargetRegisterInfo &tri, VirtRegMap &vrm,
                        LiveIntervals &lis) {
  tri_ = &tri;
  vrm_ = &vrm;
  lis_ = &lis;
  physReg2liu_.init(tri_->getNumRegs());
}

void RegAllocBase::LIUArray::clear() {
  nRegs_ =  0;
  array_.reset(0);
}

void RegAllocBase::releaseMemory() {
  physReg2liu_.clear();
}

// Check if this live virtual reg interferes with a physical register. If not,
// then check for interference on each register that aliases with the physical
// register.
bool RegAllocBase::checkPhysRegInterference(LiveIntervalUnion::Query &query,
                                            unsigned preg) {
  if (query.checkInterference())
    return true;
  for (const unsigned *asI = tri_->getAliasSet(preg); *asI; ++asI) {
    // We assume it's very unlikely for a register in the alias set to also be
    // in the original register class. So we don't bother caching the
    // interference.
    LiveIntervalUnion::Query subQuery(query.lvr(), physReg2liu_[*asI] );
    if (subQuery.checkInterference())
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
//                         RABasic Implementation
//===----------------------------------------------------------------------===//

// Driver for the register assignment and splitting heuristics.
// Manages iteration over the LiveIntervalUnions.
// 
// Minimal implementation of register assignment and splitting--spills whenever
// we run out of registers.
//
// selectOrSplit can only be called once per live virtual register. We then do a
// single interference test for each register the correct class until we find an
// available register. So, the number of interference tests in the worst case is
// |vregs| * |machineregs|. And since the number of interference tests is
// minimal, there is no value in caching them.
unsigned RABasic::selectOrSplit(LiveInterval &lvr, LiveVirtRegs &splitLVRs) {
  // Check for an available reg in this class. 
  const TargetRegisterClass *trc = mri_->getRegClass(lvr.reg);
  for (TargetRegisterClass::iterator trcI = trc->allocation_order_begin(*mf_),
         trcEnd = trc->allocation_order_end(*mf_);
       trcI != trcEnd; ++trcI) {
    unsigned preg = *trcI;
    LiveIntervalUnion::Query query(lvr, physReg2liu_[preg]);
    if (!checkPhysRegInterference(query, preg)) {
      DEBUG(dbgs() << "\tallocating: " << tri_->getName(preg) << lvr << '\n');
      return preg;
    }
  }
  DEBUG(dbgs() << "\tspilling: " << lvr << '\n');
  SmallVector<LiveInterval*, 1> spillIs; // ignored
  spiller_->spill(&lvr, splitLVRs, spillIs);

  // FIXME: update LiveStacks
  return 0;
}

bool RABasic::runOnMachineFunction(MachineFunction &mf) {
  DEBUG(dbgs() << "********** BASIC REGISTER ALLOCATION **********\n"
               << "********** Function: "
               << ((Value*)mf.getFunction())->getName() << '\n');

  mf_ = &mf;
  tm_ = &mf.getTarget();
  mri_ = &mf.getRegInfo(); 

  DEBUG(rmf_ = &getAnalysis<RenderMachineFunction>());
  
  RegAllocBase::init(*tm_->getRegisterInfo(), getAnalysis<VirtRegMap>(),
                     getAnalysis<LiveIntervals>());

  spiller_.reset(createSpiller(*this, *mf_, *vrm_));
  
  allocatePhysRegs(LessSpillWeightPriority());

  // Diagnostic output before rewriting
  DEBUG(dbgs() << "Post alloc VirtRegMap:\n" << *vrm_ << "\n");

  // optional HTML output
  DEBUG(rmf_->renderMachineFunction("After basic register allocation.", vrm_));

  // Run rewriter
  std::auto_ptr<VirtRegRewriter> rewriter(createVirtRegRewriter());
  rewriter->runOnMachineFunction(*mf_, *vrm_, lis_);
  
  return true;
}

FunctionPass* llvm::createBasicRegisterAllocator() 
{
  return new RABasic();
}
