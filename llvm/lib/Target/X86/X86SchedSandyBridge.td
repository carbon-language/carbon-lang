//=- X86SchedSandyBridge.td - X86 Sandy Bridge Scheduling ----*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Sandy Bridge to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def SandyBridgeModel : SchedMachineModel {
  // All x86 instructions are modeled as a single micro-op, and SB can decode 4
  // instructions per cycle.
  // FIXME: Identify instructions that aren't a single fused micro-op.
  let IssueWidth = 4;
  let MinLatency = 0; // 0 = Out-of-order execution.
  let LoadLatency = 4;
  let ILPWindow = 30;
  let MispredictPenalty = 16;
}

let SchedModel = SandyBridgeModel in {

// Sandy Bridge can issue micro-ops to 6 different ports in one cycle.

// Ports 0, 1, and 5 handle all computation.
def SBPort0 : ProcResource<1>;
def SBPort1 : ProcResource<1>;
def SBPort5 : ProcResource<1>;

// Ports 2 and 3 are identical. They handle loads and the address half of
// stores.
def SBPort23 : ProcResource<2>;

// Port 4 gets the data half of stores. Store data can be available later than
// the store address, but since we don't model the latency of stores, we can
// ignore that.
def SBPort4 : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def SBPort05  : ProcResGroup<[SBPort0, SBPort5]>;
def SBPort15  : ProcResGroup<[SBPort1, SBPort5]>;
def SBPort015 : ProcResGroup<[SBPort0, SBPort1, SBPort5]>;

// Integer division issued on port 0.
def SBDivider : ProcResource<1>;

// Loads are 4 cycles, so ReadAfterLd registers needn't be available until 4
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 4>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass SBWriteResPair<X86FoldableSchedWrite SchedRW,
                          ProcResourceKind ExePort,
                          int Lat> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, [ExePort]> { let Latency = Lat; }

  // Memory variant also uses a cycle on port 2/3 and adds 4 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, [SBPort23, ExePort]> {
     let Latency = !add(Lat, 4);
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [SBPort4]>;

def : WriteRes<WriteStore, [SBPort23, SBPort4]>;
def : WriteRes<WriteLoad,  [SBPort23]> { let Latency = 4; }
def : WriteRes<WriteMove,  [SBPort015]>;
def : WriteRes<WriteZero,  []>;

defm : SBWriteResPair<WriteALU,   SBPort015, 1>;
defm : SBWriteResPair<WriteIMul,  SBPort1,   3>;
defm : SBWriteResPair<WriteShift, SBPort05,  1>;
defm : SBWriteResPair<WriteJump,  SBPort5,   1>;

// This is for simple LEAs with one or two input operands.
// The complex ones can only execute on port 1, and they require two cycles on
// the port to read all inputs. We don't model that.
def : WriteRes<WriteLEA, [SBPort15]>;

// This is quite rough, latency depends on the dividend.
def : WriteRes<WriteIDiv, [SBPort0, SBDivider]> {
  let Latency = 25;
  let ResourceCycles = [1, 10];
}
def : WriteRes<WriteIDivLd, [SBPort23, SBPort0, SBDivider]> {
  let Latency = 29;
  let ResourceCycles = [1, 1, 10];
}

// Scalar and vector floating point.
defm : SBWriteResPair<WriteFAdd,   SBPort1, 3>;
defm : SBWriteResPair<WriteFMul,   SBPort0, 5>;
defm : SBWriteResPair<WriteFDiv,   SBPort0, 12>; // 10-14 cycles.
defm : SBWriteResPair<WriteFRcp,   SBPort0, 5>;
defm : SBWriteResPair<WriteFSqrt,  SBPort0, 15>;
defm : SBWriteResPair<WriteCvtF2I, SBPort1, 3>;
defm : SBWriteResPair<WriteCvtI2F, SBPort1, 4>;
defm : SBWriteResPair<WriteCvtF2F, SBPort1, 3>;

// Vector integer operations.
defm : SBWriteResPair<WriteVecShift, SBPort05,  1>;
defm : SBWriteResPair<WriteVecLogic, SBPort015, 1>;
defm : SBWriteResPair<WriteVecALU,   SBPort15,  1>;
defm : SBWriteResPair<WriteVecIMul,  SBPort0,   5>;
defm : SBWriteResPair<WriteShuffle,  SBPort15,  1>;

def : WriteRes<WriteSystem,     [SBPort015]> { let Latency = 100; }
def : WriteRes<WriteMicrocoded, [SBPort015]> { let Latency = 100; }
} // SchedModel
