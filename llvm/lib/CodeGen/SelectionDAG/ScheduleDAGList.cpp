//===-- ScheduleDAGSimple.cpp - Implement a list scheduler for isel DAG ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Evan Cheng and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a simple two pass scheduler.  The first pass attempts to push
// backward any lengthy instructions and critical paths.  The second pass packs
// instructions into semi-optimal time slots.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched"
#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetInstrInfo.h"
#include <algorithm>
#include <queue>
using namespace llvm;


namespace llvm {
/// Sorting functions for ready queue.
struct LSSortPred : public std::binary_function<SDOperand, SDOperand, bool> {
  bool operator()(const SDOperand* left, const SDOperand* right) const {
    return true;
  }
};

/// ScheduleDAGList - List scheduler.

class ScheduleDAGList : public ScheduleDAG {
private:
  LSSortPred &Cmp;

  // Ready queue
  std::priority_queue<SDOperand*, std::vector<SDOperand*>, LSSortPred> Ready;
                      
public:
  ScheduleDAGList(SelectionDAG &dag, MachineBasicBlock *bb,
                  const TargetMachine &tm, LSSortPred cmp)
    : ScheduleDAG(listSchedulingBURR, dag, bb, tm), Cmp(cmp), Ready(Cmp)
  {};

  void Schedule();
};
}  // end namespace llvm

void ScheduleDAGList::Schedule() {
}
  

llvm::ScheduleDAG*
llvm::createBURRListDAGScheduler(SelectionDAG &DAG,
                                 MachineBasicBlock *BB) {
  return new ScheduleDAGList(DAG, BB, DAG.getTarget(), LSSortPred());
}
