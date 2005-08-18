//===-- ScheduleDAG.cpp - Implement a trivial DAG scheduler ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This implements a simple code linearizer for DAGs.  This is not a very good
// way to emit code, but gets working code quickly.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "sched"
#include "llvm/CodeGen/SelectionDAGISel.h"
using namespace llvm;

/// Pick a safe ordering and emit instructions for each target node in the
/// graph.
void SelectionDAGISel::ScheduleAndEmitDAG(SelectionDAG &SD) {
  
}
