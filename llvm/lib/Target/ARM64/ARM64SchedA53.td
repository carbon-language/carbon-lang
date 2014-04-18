//=- ARM64SchedA53.td - ARM Cortex-A53 Scheduling Definitions -*- tablegen -*-=//
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
  let MicroOpBufferSize = 0; // Explicitly set to zero since A53 is in-order.
  let IssueWidth = 2;        // 2 micro-ops are dispatched per cycle.
  let MinLatency = 1 ;       // OperandCycles are interpreted as MinLatency.
  let LoadLatency = 2;       // Optimistic load latency assuming bypass.
                             // This is overriden by OperandCycles if the
                             // Itineraries are queried instead.
  let MispredictPenalty = 9; // Based on "Cortex-A53 Software Optimisation
                             // Specification - Instruction Timings"
                             // v 1.0 Spreadsheet
}


//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available.

// Modeling each pipeline as a ProcResource using the BufferSize = 0 since 
// Cortex-A53 is in-order.

def A53UnitALU    : ProcResource<2> { let BufferSize = 0; } // Int ALU
def A53UnitMAC    : ProcResource<1> { let BufferSize = 0; } // Int MAC
def A53UnitDiv    : ProcResource<1> { let BufferSize = 0; } // Int Division
def A53UnitLdSt   : ProcResource<1> { let BufferSize = 0; } // Load/Store
def A53UnitB      : ProcResource<1> { let BufferSize = 0; } // Branch
def A53UnitFPALU  : ProcResource<1> { let BufferSize = 0; } // FP ALU
def A53UnitFPMDS  : ProcResource<1> { let BufferSize = 0; } // FP Mult/Div/Sqrt


//===----------------------------------------------------------------------===//
// Subtarget-specific SchedWrite types which both map the ProcResources and
// set the latency.

let SchedModel = CortexA53Model in {

// ALU - These are reduced to 1 despite a true latency of 4 in order to easily
//       model forwarding logic. Once forwarding is properly modelled, then
//       they'll be corrected.
def : WriteRes<WriteImm, [A53UnitALU]> { let Latency = 1; }
def : WriteRes<WriteI, [A53UnitALU]> { let Latency = 1; }
def : WriteRes<WriteISReg, [A53UnitALU]> { let Latency = 1; }
def : WriteRes<WriteIEReg, [A53UnitALU]> { let Latency = 1; }
def : WriteRes<WriteExtr, [A53UnitALU]> { let Latency = 1; }
def : WriteRes<WriteIS, [A53UnitALU]> { let Latency = 1; }
def : WriteRes<WriteAdr, [A53UnitALU]> { let Latency = 1; }

// MAC
def : WriteRes<WriteIM32, [A53UnitMAC]> { let Latency = 4; }
def : WriteRes<WriteIM64, [A53UnitMAC]> { let Latency = 4; }

// Div
def : WriteRes<WriteID32, [A53UnitDiv]> { let Latency = 4; }
def : WriteRes<WriteID64, [A53UnitDiv]> { let Latency = 4; }

// Load
def : WriteRes<WriteLD, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteLDIdx, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteLDHi, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteVLD, [A53UnitLdSt]> { let Latency = 4; }

// Store
def : WriteRes<WriteST, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteSTP, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteSTIdx, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteSTX, [A53UnitLdSt]> { let Latency = 4; }
def : WriteRes<WriteVST, [A53UnitLdSt]> { let Latency = 4; }

// Branch
def : WriteRes<WriteBr, [A53UnitB]>;
def : WriteRes<WriteBrReg, [A53UnitB]>;
def : WriteRes<WriteSys, [A53UnitB]>;
def : WriteRes<WriteBarrier, [A53UnitB]>;
def : WriteRes<WriteHint, [A53UnitB]>;

// FP ALU
def : WriteRes<WriteF, [A53UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteFCmp, [A53UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteFCvt, [A53UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteFCopy, [A53UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteFImm, [A53UnitFPALU]> { let Latency = 6; }
def : WriteRes<WriteV, [A53UnitFPALU]> { let Latency = 6; }

// FP Mul, Div, Sqrt
def : WriteRes<WriteFMul, [A53UnitFPMDS]> { let Latency = 6; }
def : WriteRes<WriteFDiv, [A53UnitFPMDS]> { let Latency = 33;
                                            let ResourceCycles = [29]; }
def A53WriteFDiv : SchedWriteRes<[A53UnitFPMDS]> { let Latency = 33;
                                                   let ResourceCycles = [29]; }
def A53WriteFSqrt : SchedWriteRes<[A53UnitFPMDS]> { let Latency = 32;
                                                    let ResourceCycles = [28]; }

//===----------------------------------------------------------------------===//
// Subtarget-specific SchedRead types.

// While there is no forwarding information defined for these SchedRead types,
// they are still used by some instruction via a SchedRW list and so these zero
// SchedReadAdvances are required.

def : ReadAdvance<ReadExtrHi, 0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadVLD, 0>;

//===----------------------------------------------------------------------===//
// Subtarget-specific InstRWs.

def : InstRW<[WriteI], (instrs COPY)>;
def : InstRW<[WriteLD], (instregex "LD[1-4]")>;
def : InstRW<[WriteST], (instregex "ST[1-4]")>;
def : InstRW<[A53WriteFDiv], (instregex "^FDIV")>;
def : InstRW<[A53WriteFSqrt], (instregex ".*SQRT.*")>;

}
