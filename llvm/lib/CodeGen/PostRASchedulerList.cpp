//===----- SchedulePostRAList.cpp - list scheduler ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Dale Johannesen and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a top-down list scheduler, using standard algorithms.
// The basic approach uses a priority queue of available nodes to schedule.
// One at a time, nodes are taken from the priority queue (thus in priority
// order), checked for legality to schedule, and emitted if legal.
//
// Nodes may not be legal to schedule either due to structural hazards (e.g.
// pipeline or resource constraints) or because an input to the instruction has
// not completed execution.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "post-RA-sched"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/Debug.h"
//#include "llvm/ADT/Statistic.h"
//#include <climits>
//#include <queue>
#include "llvm/Support/CommandLine.h"
using namespace llvm;

namespace {
  bool NoPostRAScheduling;

  // When this works it will be on by default.
  cl::opt<bool, true>
  DisablePostRAScheduler("disable-post-RA-scheduler",
               cl::desc("Disable scheduling after register allocation"),
               cl::location(NoPostRAScheduling),
               cl::init(true));

  class VISIBILITY_HIDDEN SchedulePostRATDList : public MachineFunctionPass {
  public:
    static char ID;
    SchedulePostRATDList() : MachineFunctionPass((intptr_t)&ID) {}
  private:
    MachineFunction *MF;
    const TargetMachine *TM;
  public:
    const char *getPassName() const {
      return "Post RA top-down list latency scheduler (STUB)";
    }

    bool runOnMachineFunction(MachineFunction &Fn);
  };
  char SchedulePostRATDList::ID = 0;
}

bool SchedulePostRATDList::runOnMachineFunction(MachineFunction &Fn) {
  if (NoPostRAScheduling)
    return true;

  DOUT << "SchedulePostRATDList\n";
  MF = &Fn;
  TM = &MF->getTarget();

  // Loop over all of the basic blocks
  for (MachineFunction::iterator MBB = Fn.begin(), MBBe = Fn.end();
       MBB != MBBe; ++MBB)
    ;

  return true;
}
  

//===----------------------------------------------------------------------===//
//                         Public Constructor Functions
//===----------------------------------------------------------------------===//

FunctionPass *llvm::createPostRAScheduler() {
  return new SchedulePostRATDList();
}
