//===- TargetSchedule.td - Target Independent Scheduling ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the target-independent scheduling interfaces which should
// be implemented by each target which is using TableGen based scheduling.
//
//===----------------------------------------------------------------------===//

include "llvm/Target/TargetItinerary.td"

// The SchedMachineModel is defined by subtargets for three categories of data:
// 1) Basic properties for coarse grained instruction cost model.
// 2) Scheduler Read/Write resources for simple per-opcode cost model.
// 3) Instruction itineraties for detailed reservation tables.
//
// Default values for basic properties are defined in MCSchedModel. "-1"
// indicates that the property is not overriden by the target description.
class SchedMachineModel {
  int IssueWidth = -1; // Max instructions that may be scheduled per cycle.
  int MinLatency = -1; // Determines which instrucions are allowed in a group.
                       // (-1) inorder (0) ooo, (1): inorder +var latencies.
  int LoadLatency = -1; // Cycles for loads to access the cache.
  int HighLatency = -1; // Approximation of cycles for "high latency" ops.
  int MispredictPenalty = -1; // Extra cycles for a mispredicted branch.

  ProcessorItineraries Itineraries = NoItineraries;

  bit NoModel = 0; // Special tag to indicate missing machine model.
}

def NoSchedModel : SchedMachineModel {
  let NoModel = 1;
}

// TODO: Define classes for processor and scheduler resources.
