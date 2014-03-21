//=- AArch64ScheduleA53.td - ARM Cortex-A53 Scheduling Definitions -*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the itinerary class data for the ARM Cortex A53 processors.
//
//===----------------------------------------------------------------------===//

// ===---------------------------------------------------------------------===//
// The following definitions describe the simpler per-operand machine model.
// This works with MachineScheduler. See MCSchedModel.h for details.

// Cortex-A53 machine model for scheduling and other instruction cost heuristics.
def CortexA53Model : SchedMachineModel {
  let IssueWidth = 2;  // 2 micro-ops are dispatched per cycle.
  let MinLatency = 1 ; // OperandCycles are interpreted as MinLatency.
  let LoadLatency = 2; // Optimistic load latency assuming bypass.
                       // This is overriden by OperandCycles if the
                       // Itineraries are queried instead.
  let MispredictPenalty = 9; // Based on "Cortex-A53 Software Optimisation
                             // Specification - Instruction Timings"
                             // v 1.0 Spreadsheet
}


//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each pipeline as a ProcResource using the default BufferSize = -1.
// Cortex-A53 is in-order and therefore should be using BufferSize = 0. The
// current configuration performs better with the basic latencies provided so
// far. Will revisit BufferSize once the latency information is more accurate.

let SchedModel = CortexA53Model in {

def A53UnitALU    : ProcResource<2>;                        // Int ALU
def A53UnitMAC    : ProcResource<1>;                        // Int MAC
def A53UnitDiv    : ProcResource<1>;                        // Int Division
def A53UnitLdSt   : ProcResource<1>;                        // Load/Store
def A53UnitB      : ProcResource<1>;                        // Branch
def A53UnitFPALU  : ProcResource<1>;                        // FP ALU
def A53UnitFPMDS  : ProcResource<1>;                        // FP Mult/Div/Sqrt


//===----------------------------------------------------------------------===//
// Subtarget-specific SchedWrite types which both map the ProcResources and
// set the latency.

// Issue - Every instruction must consume an A53WriteIssue. Optionally,
//         instructions that cannot be dual-issued will also include the
//         A53WriteIssue2nd in their SchedRW list. That second WriteRes will
//         ensure that a second issue slot is consumed.
def A53WriteIssue : SchedWriteRes<[]>;
def A53WriteIssue2nd : SchedWriteRes<[]> { let Latency = 0; }

// ALU - These are reduced to 1 despite a true latency of 4 in order to easily
//       model forwarding logic. Once forwarding is properly modelled, then
//       they'll be corrected.
def : WriteRes<WriteALU, [A53UnitALU]> { let Latency = 1; }
def : WriteRes<WriteALUs, [A53UnitALU]> { let Latency = 1; }
def : WriteRes<WriteCMP, [A53UnitALU]> { let Latency = 1; }

// MAC
def : WriteRes<WriteMAC, [A53UnitMAC]> { let Latency = 4; }

// Div
def : WriteRes<WriteDiv, [A53UnitDiv]> { let Latency = 4; }

// Load - Note: Vector loads take 1-5 cycles to issue. For the WriteVecLd below,
//        choosing the median of 3 which makes the latency 6. May model this more
//        carefully in the future.
def : WriteRes<WriteLd, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WritePreLd, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteVecLd, [A53UnitLdSt]> { let Latency = 6; }

// Store - Note: Vector stores take 1-3 cycles to issue. For the ReadVecSt below,
//         choosing the median of 2 which makes the latency 5. May model this more
//         carefully in the future.
def : WriteRes<WriteSt, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteVecSt, [A53UnitLdSt]> { let Latency = 5; }

// Branch
def : WriteRes<WriteBr, [A53UnitB]>;
def : WriteRes<WriteBrL, [A53UnitB]>;

// FP ALU
def : WriteRes<WriteFPALU, [A53UnitFPALU]> {let Latency = 6; }

// FP MAC, Mul, Div, Sqrt
//   Using Double Precision numbers for now as a worst case. Additionally, not
//   modeling the exact hazard but instead treating the whole pipe as a hazard.
//   As an example VMUL, VMLA, and others are actually pipelined. VDIV and VSQRT
//   have a total latency of 33 and 32 respectively but only a hazard of 29 and
//   28 (double-prescion example).
def : WriteRes<WriteFPMAC, [A53UnitFPMDS]> { let Latency = 10; }
def : WriteRes<WriteFPMul, [A53UnitFPMDS]> { let Latency = 6; }
def : WriteRes<WriteFPDiv, [A53UnitFPMDS]> { let Latency = 33;
                                             let ResourceCycles = [29]; }
def : WriteRes<WriteFPSqrt, [A53UnitFPMDS]> { let Latency = 32;
                                              let ResourceCycles = [28]; }


//===----------------------------------------------------------------------===//
// Subtarget-specific SchedRead types.

// No forwarding defined for ReadALU yet.
def : ReadAdvance<ReadALU, 0>;

// No forwarding defined for ReadCMP yet.
def : ReadAdvance<ReadCMP, 0>;

// No forwarding defined for ReadBr yet.
def : ReadAdvance<ReadBr, 0>;

// No forwarding defined for ReadMAC yet.
def : ReadAdvance<ReadMAC, 0>;

// No forwarding defined for ReadDiv yet.
def : ReadAdvance<ReadDiv, 0>;

// No forwarding defined for ReadLd, ReadPreLd, ReadVecLd yet.
def : ReadAdvance<ReadLd, 0>;
def : ReadAdvance<ReadPreLd, 0>;
def : ReadAdvance<ReadVecLd, 0>;

// No forwarding defined for ReadSt and ReadVecSt yet.
def : ReadAdvance<ReadSt, 0>;
def : ReadAdvance<ReadVecSt, 0>;

// No forwarding defined for ReadFPALU yet.
def : ReadAdvance<ReadFPALU, 0>;

// No forwarding defined for ReadFPMAC/Mul/Div/Sqrt yet.
def : ReadAdvance<ReadFPMAC, 0>;
def : ReadAdvance<ReadFPMul, 0>;
def : ReadAdvance<ReadFPDiv, 0>;
def : ReadAdvance<ReadFPSqrt, 0>;

}
