//===---- ScheduleDAG.cpp - Implement the ScheduleDAG class ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements the ScheduleDAG class, which is a base class used by
// scheduling implementation classes.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "pre-RA-sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Support/Debug.h"
using namespace llvm;

ScheduleDAG::ScheduleDAG(SelectionDAG *dag, MachineBasicBlock *bb,
                         const TargetMachine &tm)
  : DAG(dag), BB(bb), TM(tm), MRI(BB->getParent()->getRegInfo()) {
  TII = TM.getInstrInfo();
  MF  = BB->getParent();
  TRI = TM.getRegisterInfo();
  TLI = TM.getTargetLowering();
  ConstPool = MF->getConstantPool();
}

ScheduleDAG::~ScheduleDAG() {}

/// CalculateDepths - compute depths using algorithms for the longest
/// paths in the DAG
void ScheduleDAG::CalculateDepths() {
  unsigned DAGSize = SUnits.size();
  std::vector<SUnit*> WorkList;
  WorkList.reserve(DAGSize);

  // Initialize the data structures
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    unsigned Degree = SU->Preds.size();
    // Temporarily use the Depth field as scratch space for the degree count.
    SU->Depth = Degree;

    // Is it a node without dependencies?
    if (Degree == 0) {
        assert(SU->Preds.empty() && "SUnit should have no predecessors");
        // Collect leaf nodes
        WorkList.push_back(SU);
    }
  }

  // Process nodes in the topological order
  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back();
    WorkList.pop_back();
    unsigned SUDepth = 0;

    // Use dynamic programming:
    // When current node is being processed, all of its dependencies
    // are already processed.
    // So, just iterate over all predecessors and take the longest path
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      unsigned PredDepth = I->Dep->Depth;
      if (PredDepth+1 > SUDepth) {
          SUDepth = PredDepth + 1;
      }
    }

    SU->Depth = SUDepth;

    // Update degrees of all nodes depending on current SUnit
    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      SUnit *SU = I->Dep;
      if (!--SU->Depth)
        // If all dependencies of the node are processed already,
        // then the longest path for the node can be computed now
        WorkList.push_back(SU);
    }
  }
}

/// CalculateHeights - compute heights using algorithms for the longest
/// paths in the DAG
void ScheduleDAG::CalculateHeights() {
  unsigned DAGSize = SUnits.size();
  std::vector<SUnit*> WorkList;
  WorkList.reserve(DAGSize);

  // Initialize the data structures
  for (unsigned i = 0, e = DAGSize; i != e; ++i) {
    SUnit *SU = &SUnits[i];
    unsigned Degree = SU->Succs.size();
    // Temporarily use the Height field as scratch space for the degree count.
    SU->Height = Degree;

    // Is it a node without dependencies?
    if (Degree == 0) {
        assert(SU->Succs.empty() && "Something wrong");
        assert(WorkList.empty() && "Should be empty");
        // Collect leaf nodes
        WorkList.push_back(SU);
    }
  }

  // Process nodes in the topological order
  while (!WorkList.empty()) {
    SUnit *SU = WorkList.back();
    WorkList.pop_back();
    unsigned SUHeight = 0;

    // Use dynamic programming:
    // When current node is being processed, all of its dependencies
    // are already processed.
    // So, just iterate over all successors and take the longest path
    for (SUnit::const_succ_iterator I = SU->Succs.begin(), E = SU->Succs.end();
         I != E; ++I) {
      unsigned SuccHeight = I->Dep->Height;
      if (SuccHeight+1 > SUHeight) {
          SUHeight = SuccHeight + 1;
      }
    }

    SU->Height = SUHeight;

    // Update degrees of all nodes depending on current SUnit
    for (SUnit::const_pred_iterator I = SU->Preds.begin(), E = SU->Preds.end();
         I != E; ++I) {
      SUnit *SU = I->Dep;
      if (!--SU->Height)
        // If all dependencies of the node are processed already,
        // then the longest path for the node can be computed now
        WorkList.push_back(SU);
    }
  }
}

/// dump - dump the schedule.
void ScheduleDAG::dumpSchedule() const {
  for (unsigned i = 0, e = Sequence.size(); i != e; i++) {
    if (SUnit *SU = Sequence[i])
      SU->dump(this);
    else
      cerr << "**** NOOP ****\n";
  }
}


/// Run - perform scheduling.
///
void ScheduleDAG::Run() {
  Schedule();
  
  DOUT << "*** Final schedule ***\n";
  DEBUG(dumpSchedule());
  DOUT << "\n";
}

/// SUnit - Scheduling unit. It's an wrapper around either a single SDNode or
/// a group of nodes flagged together.
void SUnit::dump(const ScheduleDAG *G) const {
  cerr << "SU(" << NodeNum << "): ";
  G->dumpNode(this);
}

void SUnit::dumpAll(const ScheduleDAG *G) const {
  dump(G);

  cerr << "  # preds left       : " << NumPredsLeft << "\n";
  cerr << "  # succs left       : " << NumSuccsLeft << "\n";
  cerr << "  Latency            : " << Latency << "\n";
  cerr << "  Depth              : " << Depth << "\n";
  cerr << "  Height             : " << Height << "\n";

  if (Preds.size() != 0) {
    cerr << "  Predecessors:\n";
    for (SUnit::const_succ_iterator I = Preds.begin(), E = Preds.end();
         I != E; ++I) {
      if (I->isCtrl)
        cerr << "   ch  #";
      else
        cerr << "   val #";
      cerr << I->Dep << " - SU(" << I->Dep->NodeNum << ")";
      if (I->isSpecial)
        cerr << " *";
      cerr << "\n";
    }
  }
  if (Succs.size() != 0) {
    cerr << "  Successors:\n";
    for (SUnit::const_succ_iterator I = Succs.begin(), E = Succs.end();
         I != E; ++I) {
      if (I->isCtrl)
        cerr << "   ch  #";
      else
        cerr << "   val #";
      cerr << I->Dep << " - SU(" << I->Dep->NodeNum << ")";
      if (I->isSpecial)
        cerr << " *";
      cerr << "\n";
    }
  }
  cerr << "\n";
}
