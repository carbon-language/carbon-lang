//===- lib/CodeGen/MachineTraceMetrics.cpp ----------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "early-ifcvt"
#include "MachineTraceMetrics.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/ADT/PostOrderIterator.h"

using namespace llvm;

char MachineTraceMetrics::ID = 0;
char &llvm::MachineTraceMetricsID = MachineTraceMetrics::ID;

INITIALIZE_PASS_BEGIN(MachineTraceMetrics,
                  "machine-trace-metrics", "Machine Trace Metrics", false, true)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(MachineTraceMetrics,
                  "machine-trace-metrics", "Machine Trace Metrics", false, true)

MachineTraceMetrics::MachineTraceMetrics()
  : MachineFunctionPass(ID), TII(0), TRI(0), MRI(0), Loops(0) {
  std::fill(Ensembles, array_endof(Ensembles), (Ensemble*)0);
}

void MachineTraceMetrics::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<MachineBranchProbabilityInfo>();
  AU.addRequired<MachineLoopInfo>();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineTraceMetrics::runOnMachineFunction(MachineFunction &Func) {
  MF = &Func;
  TII = MF->getTarget().getInstrInfo();
  TRI = MF->getTarget().getRegisterInfo();
  MRI = &MF->getRegInfo();
  Loops = &getAnalysis<MachineLoopInfo>();
  BlockInfo.resize(MF->getNumBlockIDs());
  return false;
}

void MachineTraceMetrics::releaseMemory() {
  BlockInfo.clear();
  for (unsigned i = 0; i != TS_NumStrategies; ++i) {
    delete Ensembles[i];
    Ensembles[i] = 0;
  }
}

//===----------------------------------------------------------------------===//
//                          Fixed block information
//===----------------------------------------------------------------------===//
//
// The number of instructions in a basic block and the CPU resources used by
// those instructions don't depend on any given trace strategy.

/// Is MI an instruction that should be considered free because it will likely
/// be eliminated by later passes?
static bool isFree(const MachineInstr *MI) {
  switch(MI->getOpcode()) {
  default: return false;
  case TargetOpcode::PHI:
  case TargetOpcode::PROLOG_LABEL:
  case TargetOpcode::EH_LABEL:
  case TargetOpcode::GC_LABEL:
  case TargetOpcode::KILL:
  case TargetOpcode::EXTRACT_SUBREG:
  case TargetOpcode::INSERT_SUBREG:
  case TargetOpcode::IMPLICIT_DEF:
  case TargetOpcode::SUBREG_TO_REG:
  case TargetOpcode::COPY_TO_REGCLASS:
  case TargetOpcode::DBG_VALUE:
  case TargetOpcode::REG_SEQUENCE:
  case TargetOpcode::COPY:
    return true;
  }
}

/// Compute the resource usage in basic block MBB.
const MachineTraceMetrics::FixedBlockInfo*
MachineTraceMetrics::getResources(const MachineBasicBlock *MBB) {
  assert(MBB && "No basic block");
  FixedBlockInfo *FBI = &BlockInfo[MBB->getNumber()];
  if (FBI->hasResources())
    return FBI;

  // Compute resource usage in the block.
  // FIXME: Compute per-functional unit counts.
  FBI->HasCalls = false;
  unsigned InstrCount = 0;
  for (MachineBasicBlock::const_iterator I = MBB->begin(), E = MBB->end();
       I != E; ++I) {
    const MachineInstr *MI = I;
    if (isFree(MI))
      continue;
    ++InstrCount;
    if (MI->isCall())
      FBI->HasCalls = true;
  }
  FBI->InstrCount = InstrCount;
  return FBI;
}

//===----------------------------------------------------------------------===//
//                         Ensemble utility functions
//===----------------------------------------------------------------------===//

MachineTraceMetrics::Ensemble::Ensemble(MachineTraceMetrics *ct)
  : CT(*ct) {
  BlockInfo.resize(CT.BlockInfo.size());
}

// Virtual destructor serves as an anchor.
MachineTraceMetrics::Ensemble::~Ensemble() {}

const MachineLoop*
MachineTraceMetrics::Ensemble::getLoopFor(const MachineBasicBlock *MBB) const {
  return CT.Loops->getLoopFor(MBB);
}

// Update resource-related information in the TraceBlockInfo for MBB.
// Only update resources related to the trace above MBB.
void MachineTraceMetrics::Ensemble::
computeDepthResources(const MachineBasicBlock *MBB) {
  TraceBlockInfo *TBI = &BlockInfo[MBB->getNumber()];

  // Compute resources from trace above. The top block is simple.
  if (!TBI->Pred) {
    TBI->InstrDepth = 0;
    TBI->Head = MBB->getNumber();
    return;
  }

  // Compute from the block above. A post-order traversal ensures the
  // predecessor is always computed first.
  TraceBlockInfo *PredTBI = &BlockInfo[TBI->Pred->getNumber()];
  assert(PredTBI->hasValidDepth() && "Trace above has not been computed yet");
  const FixedBlockInfo *PredFBI = CT.getResources(TBI->Pred);
  TBI->InstrDepth = PredTBI->InstrDepth + PredFBI->InstrCount;
  TBI->Head = PredTBI->Head;
}

// Update resource-related information in the TraceBlockInfo for MBB.
// Only update resources related to the trace below MBB.
void MachineTraceMetrics::Ensemble::
computeHeightResources(const MachineBasicBlock *MBB) {
  TraceBlockInfo *TBI = &BlockInfo[MBB->getNumber()];

  // Compute resources for the current block.
  TBI->InstrHeight = CT.getResources(MBB)->InstrCount;

  // The trace tail is done.
  if (!TBI->Succ) {
    TBI->Tail = MBB->getNumber();
    return;
  }

  // Compute from the block below. A post-order traversal ensures the
  // predecessor is always computed first.
  TraceBlockInfo *SuccTBI = &BlockInfo[TBI->Succ->getNumber()];
  assert(SuccTBI->hasValidHeight() && "Trace below has not been computed yet");
  TBI->InstrHeight += SuccTBI->InstrHeight;
  TBI->Tail = SuccTBI->Tail;
}

// Check if depth resources for MBB are valid and return the TBI.
// Return NULL if the resources have been invalidated.
const MachineTraceMetrics::TraceBlockInfo*
MachineTraceMetrics::Ensemble::
getDepthResources(const MachineBasicBlock *MBB) const {
  const TraceBlockInfo *TBI = &BlockInfo[MBB->getNumber()];
  return TBI->hasValidDepth() ? TBI : 0;
}

// Check if height resources for MBB are valid and return the TBI.
// Return NULL if the resources have been invalidated.
const MachineTraceMetrics::TraceBlockInfo*
MachineTraceMetrics::Ensemble::
getHeightResources(const MachineBasicBlock *MBB) const {
  const TraceBlockInfo *TBI = &BlockInfo[MBB->getNumber()];
  return TBI->hasValidHeight() ? TBI : 0;
}

//===----------------------------------------------------------------------===//
//                         Trace Selection Strategies
//===----------------------------------------------------------------------===//
//
// A trace selection strategy is implemented as a sub-class of Ensemble. The
// trace through a block B is computed by two DFS traversals of the CFG
// starting from B. One upwards, and one downwards. During the upwards DFS,
// pickTracePred() is called on the post-ordered blocks. During the downwards
// DFS, pickTraceSucc() is called in a post-order.
//

// MinInstrCountEnsemble - Pick the trace that executes the least number of
// instructions.
namespace {
class MinInstrCountEnsemble : public MachineTraceMetrics::Ensemble {
  const char *getName() const { return "MinInstr"; }
  const MachineBasicBlock *pickTracePred(const MachineBasicBlock*);
  const MachineBasicBlock *pickTraceSucc(const MachineBasicBlock*);

public:
  MinInstrCountEnsemble(MachineTraceMetrics *ct)
    : MachineTraceMetrics::Ensemble(ct) {}
};
}

// Select the preferred predecessor for MBB.
const MachineBasicBlock*
MinInstrCountEnsemble::pickTracePred(const MachineBasicBlock *MBB) {
  if (MBB->pred_empty())
    return 0;
  const MachineLoop *CurLoop = getLoopFor(MBB);
  // Don't leave loops, and never follow back-edges.
  if (CurLoop && MBB == CurLoop->getHeader())
    return 0;
  unsigned CurCount = CT.getResources(MBB)->InstrCount;
  const MachineBasicBlock *Best = 0;
  unsigned BestDepth = 0;
  for (MachineBasicBlock::const_pred_iterator
       I = MBB->pred_begin(), E = MBB->pred_end(); I != E; ++I) {
    const MachineBasicBlock *Pred = *I;
    const MachineTraceMetrics::TraceBlockInfo *PredTBI =
      getDepthResources(Pred);
    // Ignore invalidated predecessors. This never happens on the first scan,
    // but if we rejected this predecessor earlier, it won't be revalidated.
    if (!PredTBI)
      continue;
    // Don't consider predecessors in other loops.
    if (getLoopFor(Pred) != CurLoop)
      continue;
    // Pick the predecessor that would give this block the smallest InstrDepth.
    unsigned Depth = PredTBI->InstrDepth + CurCount;
    if (!Best || Depth < BestDepth)
      Best = Pred, BestDepth = Depth;
  }
  return Best;
}

// Select the preferred successor for MBB.
const MachineBasicBlock*
MinInstrCountEnsemble::pickTraceSucc(const MachineBasicBlock *MBB) {
  if (MBB->pred_empty())
    return 0;
  const MachineLoop *CurLoop = getLoopFor(MBB);
  const MachineBasicBlock *Best = 0;
  unsigned BestHeight = 0;
  for (MachineBasicBlock::const_succ_iterator
       I = MBB->succ_begin(), E = MBB->succ_end(); I != E; ++I) {
    const MachineBasicBlock *Succ = *I;
    const MachineTraceMetrics::TraceBlockInfo *SuccTBI =
      getHeightResources(Succ);
    // Ignore invalidated successors.
    if (!SuccTBI)
      continue;
    // Don't consider back-edges.
    if (CurLoop && Succ == CurLoop->getHeader())
      continue;
    // Don't consider successors in other loops.
    if (getLoopFor(Succ) != CurLoop)
      continue;
    // Pick the successor that would give this block the smallest InstrHeight.
    unsigned Height = SuccTBI->InstrHeight;
    if (!Best || Height < BestHeight)
      Best = Succ, BestHeight = Height;
  }
  return Best;
}

// Get an Ensemble sub-class for the requested trace strategy.
MachineTraceMetrics::Ensemble *
MachineTraceMetrics::getEnsemble(MachineTraceMetrics::Strategy strategy) {
  assert(strategy < TS_NumStrategies && "Invalid trace strategy enum");
  Ensemble *&E = Ensembles[strategy];
  if (E)
    return E;

  // Allocate new Ensemble on demand.
  switch (strategy) {
  case TS_MinInstrCount: return (E = new MinInstrCountEnsemble(this));
  default: llvm_unreachable("Invalid trace strategy enum");
  }
}

void MachineTraceMetrics::invalidate(const MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Invalidate traces through BB#" << MBB->getNumber() << '\n');
  BlockInfo[MBB->getNumber()].invalidate();
  for (unsigned i = 0; i != TS_NumStrategies; ++i)
    if (Ensembles[i])
      Ensembles[i]->invalidate(MBB);
}

void MachineTraceMetrics::verify() const {
#ifndef NDEBUG
  assert(BlockInfo.size() == MF->getNumBlockIDs() && "Outdated BlockInfo size");
  for (unsigned i = 0; i != TS_NumStrategies; ++i)
    if (Ensembles[i])
      Ensembles[i]->verify();
#endif
}

//===----------------------------------------------------------------------===//
//                               Trace building
//===----------------------------------------------------------------------===//
//
// Traces are built by two CFG traversals. To avoid recomputing too much, use a
// set abstraction that confines the search to the current loop, and doesn't
// revisit blocks.

namespace {
struct LoopBounds {
  MutableArrayRef<MachineTraceMetrics::TraceBlockInfo> Blocks;
  const MachineLoopInfo *Loops;
  const MachineLoop *CurLoop;
  bool Downward;
  LoopBounds(MutableArrayRef<MachineTraceMetrics::TraceBlockInfo> blocks,
             const MachineLoopInfo *loops, const MachineLoop *curloop)
    : Blocks(blocks), Loops(loops), CurLoop(curloop), Downward(false) {}
};
}

// Specialize po_iterator_storage in order to prune the post-order traversal so
// it is limited to the current loop and doesn't traverse the loop back edges.
namespace llvm {
template<>
class po_iterator_storage<LoopBounds, true> {
  LoopBounds &LB;
public:
  po_iterator_storage(LoopBounds &lb) : LB(lb) {}
  void finishPostorder(const MachineBasicBlock*) {}

  bool insertEdge(const MachineBasicBlock *From, const MachineBasicBlock *To) {
    // Skip already visited To blocks.
    MachineTraceMetrics::TraceBlockInfo &TBI = LB.Blocks[To->getNumber()];
    if (LB.Downward ? TBI.hasValidHeight() : TBI.hasValidDepth())
      return false;
    // Don't follow CurLoop backedges.
    if (LB.CurLoop && (LB.Downward ? To : From) == LB.CurLoop->getHeader())
      return false;
    // Don't leave CurLoop.
    if (LB.Loops->getLoopFor(To) != LB.CurLoop)
      return false;
    // This is a new block. The PO traversal will compute height/depth
    // resources, causing us to reject new edges to To. This only works because
    // we reject back-edges, so the CFG is cycle-free.
    return true;
  }
};
}

/// Compute the trace through MBB.
void MachineTraceMetrics::Ensemble::computeTrace(const MachineBasicBlock *MBB) {
  DEBUG(dbgs() << "Computing " << getName() << " trace through BB#"
               << MBB->getNumber() << '\n');
  // Set up loop bounds for the backwards post-order traversal.
  LoopBounds Bounds(BlockInfo, CT.Loops, getLoopFor(MBB));

  // Run an upwards post-order search for the trace start.
  Bounds.Downward = false;
  typedef ipo_ext_iterator<const MachineBasicBlock*, LoopBounds> UpwardPO;
  for (UpwardPO I = ipo_ext_begin(MBB, Bounds), E = ipo_ext_end(MBB, Bounds);
       I != E; ++I) {
    DEBUG(dbgs() << "  pred for BB#" << I->getNumber() << ": ");
    TraceBlockInfo &TBI = BlockInfo[I->getNumber()];
    // All the predecessors have been visited, pick the preferred one.
    TBI.Pred = pickTracePred(*I);
    DEBUG({
      if (TBI.Pred)
        dbgs() << "BB#" << TBI.Pred->getNumber() << '\n';
      else
        dbgs() << "null\n";
    });
    // The trace leading to I is now known, compute the depth resources.
    computeDepthResources(*I);
  }

  // Run a downwards post-order search for the trace end.
  Bounds.Downward = true;
  typedef po_ext_iterator<const MachineBasicBlock*, LoopBounds> DownwardPO;
  for (DownwardPO I = po_ext_begin(MBB, Bounds), E = po_ext_end(MBB, Bounds);
       I != E; ++I) {
    DEBUG(dbgs() << "  succ for BB#" << I->getNumber() << ": ");
    TraceBlockInfo &TBI = BlockInfo[I->getNumber()];
    // All the successors have been visited, pick the preferred one.
    TBI.Succ = pickTraceSucc(*I);
    DEBUG({
      if (TBI.Pred)
        dbgs() << "BB#" << TBI.Succ->getNumber() << '\n';
      else
        dbgs() << "null\n";
    });
    // The trace leaving I is now known, compute the height resources.
    computeHeightResources(*I);
  }
}

/// Invalidate traces through BadMBB.
void
MachineTraceMetrics::Ensemble::invalidate(const MachineBasicBlock *BadMBB) {
  SmallVector<const MachineBasicBlock*, 16> WorkList;
  TraceBlockInfo &BadTBI = BlockInfo[BadMBB->getNumber()];

  // Invalidate height resources of blocks above MBB.
  if (BadTBI.hasValidHeight()) {
    BadTBI.invalidateHeight();
    WorkList.push_back(BadMBB);
    do {
      const MachineBasicBlock *MBB = WorkList.pop_back_val();
      DEBUG(dbgs() << "Invalidate BB#" << MBB->getNumber() << ' ' << getName()
            << " height.\n");
      // Find any MBB predecessors that have MBB as their preferred successor.
      // They are the only ones that need to be invalidated.
      for (MachineBasicBlock::const_pred_iterator
           I = MBB->pred_begin(), E = MBB->pred_end(); I != E; ++I) {
        TraceBlockInfo &TBI = BlockInfo[(*I)->getNumber()];
        if (!TBI.hasValidHeight())
          continue;
        if (TBI.Succ == MBB) {
          TBI.invalidateHeight();
          WorkList.push_back(*I);
          continue;
        }
        // Verify that TBI.Succ is actually a *I successor.
        assert((!TBI.Succ || (*I)->isSuccessor(TBI.Succ)) && "CFG changed");
      }
    } while (!WorkList.empty());
  }

  // Invalidate depth resources of blocks below MBB.
  if (BadTBI.hasValidDepth()) {
    BadTBI.invalidateDepth();
    WorkList.push_back(BadMBB);
    do {
      const MachineBasicBlock *MBB = WorkList.pop_back_val();
      DEBUG(dbgs() << "Invalidate BB#" << MBB->getNumber() << ' ' << getName()
            << " depth.\n");
      // Find any MBB successors that have MBB as their preferred predecessor.
      // They are the only ones that need to be invalidated.
      for (MachineBasicBlock::const_succ_iterator
           I = MBB->succ_begin(), E = MBB->succ_end(); I != E; ++I) {
        TraceBlockInfo &TBI = BlockInfo[(*I)->getNumber()];
        if (!TBI.hasValidDepth())
          continue;
        if (TBI.Pred == MBB) {
          TBI.invalidateDepth();
          WorkList.push_back(*I);
          continue;
        }
        // Verify that TBI.Pred is actually a *I predecessor.
        assert((!TBI.Pred || (*I)->isPredecessor(TBI.Pred)) && "CFG changed");
      }
    } while (!WorkList.empty());
  }
}

void MachineTraceMetrics::Ensemble::verify() const {
#ifndef NDEBUG
  assert(BlockInfo.size() == CT.MF->getNumBlockIDs() &&
         "Outdated BlockInfo size");
  for (unsigned Num = 0, e = BlockInfo.size(); Num != e; ++Num) {
    const TraceBlockInfo &TBI = BlockInfo[Num];
    if (TBI.hasValidDepth() && TBI.Pred) {
      const MachineBasicBlock *MBB = CT.MF->getBlockNumbered(Num);
      assert(MBB->isPredecessor(TBI.Pred) && "CFG doesn't match trace");
      assert(BlockInfo[TBI.Pred->getNumber()].hasValidDepth() &&
             "Trace is broken, depth should have been invalidated.");
      const MachineLoop *Loop = getLoopFor(MBB);
      assert(!(Loop && MBB == Loop->getHeader()) && "Trace contains backedge");
    }
    if (TBI.hasValidHeight() && TBI.Succ) {
      const MachineBasicBlock *MBB = CT.MF->getBlockNumbered(Num);
      assert(MBB->isSuccessor(TBI.Succ) && "CFG doesn't match trace");
      assert(BlockInfo[TBI.Succ->getNumber()].hasValidHeight() &&
             "Trace is broken, height should have been invalidated.");
      const MachineLoop *Loop = getLoopFor(MBB);
      const MachineLoop *SuccLoop = getLoopFor(TBI.Succ);
      assert(!(Loop && Loop == SuccLoop && TBI.Succ == Loop->getHeader()) &&
             "Trace contains backedge");
    }
  }
#endif
}

MachineTraceMetrics::Trace
MachineTraceMetrics::Ensemble::getTrace(const MachineBasicBlock *MBB) {
  // FIXME: Check cache tags, recompute as needed.
  computeTrace(MBB);
  return Trace(*this, BlockInfo[MBB->getNumber()]);
}

void MachineTraceMetrics::Ensemble::print(raw_ostream &OS) const {
  OS << getName() << " ensemble:\n";
  for (unsigned i = 0, e = BlockInfo.size(); i != e; ++i) {
    OS << "  BB#" << i << '\t';
    BlockInfo[i].print(OS);
    OS << '\n';
  }
}

void MachineTraceMetrics::TraceBlockInfo::print(raw_ostream &OS) const {
  if (hasValidDepth()) {
    OS << "depth=" << InstrDepth;
    if (Pred)
      OS << " pred=BB#" << Pred->getNumber();
    else
      OS << " pred=null";
    OS << " head=BB#" << Head;
  } else
    OS << "depth invalid";
  OS << ", ";
  if (hasValidHeight()) {
    OS << "height=" << InstrHeight;
    if (Succ)
      OS << " succ=BB#" << Succ->getNumber();
    else
      OS << " succ=null";
    OS << " tail=BB#" << Tail;
  } else
    OS << "height invalid";
}

void MachineTraceMetrics::Trace::print(raw_ostream &OS) const {
  unsigned MBBNum = &TBI - &TE.BlockInfo[0];

  OS << TE.getName() << " trace BB#" << TBI.Head << " --> BB#" << MBBNum
     << " --> BB#" << TBI.Tail << ':';
  if (TBI.hasValidHeight() && TBI.hasValidDepth())
    OS << ' ' << getInstrCount() << " instrs.";

  const MachineTraceMetrics::TraceBlockInfo *Block = &TBI;
  OS << "\nBB#" << MBBNum;
  while (Block->hasValidDepth() && Block->Pred) {
    unsigned Num = Block->Pred->getNumber();
    OS << " <- BB#" << Num;
    Block = &TE.BlockInfo[Num];
  }

  Block = &TBI;
  OS << "\n    ";
  while (Block->hasValidHeight() && Block->Succ) {
    unsigned Num = Block->Succ->getNumber();
    OS << " -> BB#" << Num;
    Block = &TE.BlockInfo[Num];
  }
  OS << '\n';
}
