//=- X86SchedHaswell.td - X86 Haswell Scheduling -------------*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for Haswell to support instruction
// scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def HaswellModel : SchedMachineModel {
  // All x86 instructions are modeled as a single micro-op, and HW can decode 4
  // instructions per cycle.
  let IssueWidth = 4;
  let MicroOpBufferSize = 192; // Based on the reorder buffer.
  let LoadLatency = 4;
  let MispredictPenalty = 16;

  // Based on the LSD (loop-stream detector) queue size and benchmarking data.
  let LoopMicroOpBufferSize = 50;

  // FIXME: SSE4 and AVX are unimplemented. This flag is set to allow
  // the scheduler to assign a default model to unrecognized opcodes.
  let CompleteModel = 0;
}

let SchedModel = HaswellModel in {

// Haswell can issue micro-ops to 8 different ports in one cycle.

// Ports 0, 1, 5, and 6 handle all computation.
// Port 4 gets the data half of stores. Store data can be available later than
// the store address, but since we don't model the latency of stores, we can
// ignore that.
// Ports 2 and 3 are identical. They handle loads and the address half of
// stores. Port 7 can handle address calculations.
def HWPort0 : ProcResource<1>;
def HWPort1 : ProcResource<1>;
def HWPort2 : ProcResource<1>;
def HWPort3 : ProcResource<1>;
def HWPort4 : ProcResource<1>;
def HWPort5 : ProcResource<1>;
def HWPort6 : ProcResource<1>;
def HWPort7 : ProcResource<1>;

// Many micro-ops are capable of issuing on multiple ports.
def HWPort23  : ProcResGroup<[HWPort2, HWPort3]>;
def HWPort237 : ProcResGroup<[HWPort2, HWPort3, HWPort7]>;
def HWPort05  : ProcResGroup<[HWPort0, HWPort5]>;
def HWPort06 : ProcResGroup<[HWPort0, HWPort6]>;
def HWPort15  : ProcResGroup<[HWPort1, HWPort5]>;
def HWPort16  : ProcResGroup<[HWPort1, HWPort6]>;
def HWPort015 : ProcResGroup<[HWPort0, HWPort1, HWPort5]>;
def HWPort0156: ProcResGroup<[HWPort0, HWPort1, HWPort5, HWPort6]>;

// 60 Entry Unified Scheduler
def HWPortAny : ProcResGroup<[HWPort0, HWPort1, HWPort2, HWPort3, HWPort4,
                              HWPort5, HWPort6, HWPort7]> {
  let BufferSize=60;
}

// Integer division issued on port 0.
def HWDivider : ProcResource<1>;

// Loads are 4 cycles, so ReadAfterLd registers needn't be available until 4
// cycles after the memory operand.
def : ReadAdvance<ReadAfterLd, 4>;

// Many SchedWrites are defined in pairs with and without a folded load.
// Instructions with folded loads are usually micro-fused, so they only appear
// as two micro-ops when queued in the reservation station.
// This multiclass defines the resource usage for variants with and without
// folded loads.
multiclass HWWriteResPair<X86FoldableSchedWrite SchedRW,
                          ProcResourceKind ExePort,
                          int Lat> {
  // Register variant is using a single cycle on ExePort.
  def : WriteRes<SchedRW, [ExePort]> { let Latency = Lat; }

  // Memory variant also uses a cycle on port 2/3 and adds 4 cycles to the
  // latency.
  def : WriteRes<SchedRW.Folded, [HWPort23, ExePort]> {
     let Latency = !add(Lat, 4);
  }
}

// A folded store needs a cycle on port 4 for the store data, but it does not
// need an extra port 2/3 cycle to recompute the address.
def : WriteRes<WriteRMW, [HWPort4]>;

// Store_addr on 237.
// Store_data on 4.
def : WriteRes<WriteStore, [HWPort237, HWPort4]>;
def : WriteRes<WriteLoad,  [HWPort23]> { let Latency = 4; }
def : WriteRes<WriteMove,  [HWPort0156]>;
def : WriteRes<WriteZero,  []>;

defm : HWWriteResPair<WriteALU,   HWPort0156, 1>;
defm : HWWriteResPair<WriteIMul,  HWPort1,   3>;
def  : WriteRes<WriteIMulH, []> { let Latency = 3; }
defm : HWWriteResPair<WriteShift, HWPort06,  1>;
defm : HWWriteResPair<WriteJump,  HWPort06,   1>;

// This is for simple LEAs with one or two input operands.
// The complex ones can only execute on port 1, and they require two cycles on
// the port to read all inputs. We don't model that.
def : WriteRes<WriteLEA, [HWPort15]>;

// This is quite rough, latency depends on the dividend.
def : WriteRes<WriteIDiv, [HWPort0, HWDivider]> {
  let Latency = 25;
  let ResourceCycles = [1, 10];
}
def : WriteRes<WriteIDivLd, [HWPort23, HWPort0, HWDivider]> {
  let Latency = 29;
  let ResourceCycles = [1, 1, 10];
}

// Scalar and vector floating point.
defm : HWWriteResPair<WriteFAdd,   HWPort1, 3>;
defm : HWWriteResPair<WriteFMul,   HWPort0, 5>;
defm : HWWriteResPair<WriteFDiv,   HWPort0, 12>; // 10-14 cycles.
defm : HWWriteResPair<WriteFRcp,   HWPort0, 5>;
defm : HWWriteResPair<WriteFSqrt,  HWPort0, 15>;
defm : HWWriteResPair<WriteCvtF2I, HWPort1, 3>;
defm : HWWriteResPair<WriteCvtI2F, HWPort1, 4>;
defm : HWWriteResPair<WriteCvtF2F, HWPort1, 3>;
defm : HWWriteResPair<WriteFShuffle,  HWPort5,  1>;
defm : HWWriteResPair<WriteFBlend,  HWPort015,  1>;
defm : HWWriteResPair<WriteFShuffle256,  HWPort5,  3>;

def : WriteRes<WriteFVarBlend, [HWPort5]> {
  let Latency = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteFVarBlendLd, [HWPort5, HWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

// Vector integer operations.
defm : HWWriteResPair<WriteVecShift, HWPort0,  1>;
defm : HWWriteResPair<WriteVecLogic, HWPort015, 1>;
defm : HWWriteResPair<WriteVecALU,   HWPort15,  1>;
defm : HWWriteResPair<WriteVecIMul,  HWPort0,   5>;
defm : HWWriteResPair<WriteShuffle,  HWPort5,  1>;
defm : HWWriteResPair<WriteBlend,  HWPort15,  1>;
defm : HWWriteResPair<WriteShuffle256,  HWPort5,  3>;

def : WriteRes<WriteVarBlend, [HWPort5]> {
  let Latency = 2;
  let ResourceCycles = [2];
}
def : WriteRes<WriteVarBlendLd, [HWPort5, HWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

def : WriteRes<WriteVarVecShift, [HWPort0, HWPort5]> {
  let Latency = 2;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteVarVecShiftLd, [HWPort0, HWPort5, HWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1, 1];
}

def : WriteRes<WriteMPSAD, [HWPort0, HWPort5]> {
  let Latency = 6;
  let ResourceCycles = [1, 2];
}
def : WriteRes<WriteMPSADLd, [HWPort23, HWPort0, HWPort5]> {
  let Latency = 6;
  let ResourceCycles = [1, 1, 2];
}

// String instructions.
// Packed Compare Implicit Length Strings, Return Mask
def : WriteRes<WritePCmpIStrM, [HWPort0]> {
  let Latency = 10;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrMLd, [HWPort0, HWPort23]> {
  let Latency = 10;
  let ResourceCycles = [3, 1];
}

// Packed Compare Explicit Length Strings, Return Mask
def : WriteRes<WritePCmpEStrM, [HWPort0, HWPort16, HWPort5]> {
  let Latency = 10;
  let ResourceCycles = [3, 2, 4];
}
def : WriteRes<WritePCmpEStrMLd, [HWPort05, HWPort16, HWPort23]> {
  let Latency = 10;
  let ResourceCycles = [6, 2, 1];
}

// Packed Compare Implicit Length Strings, Return Index
def : WriteRes<WritePCmpIStrI, [HWPort0]> {
  let Latency = 11;
  let ResourceCycles = [3];
}
def : WriteRes<WritePCmpIStrILd, [HWPort0, HWPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 1];
}

// Packed Compare Explicit Length Strings, Return Index
def : WriteRes<WritePCmpEStrI, [HWPort05, HWPort16]> {
  let Latency = 11;
  let ResourceCycles = [6, 2];
}
def : WriteRes<WritePCmpEStrILd, [HWPort0, HWPort16, HWPort5, HWPort23]> {
  let Latency = 11;
  let ResourceCycles = [3, 2, 2, 1];
}

// AES Instructions.
def : WriteRes<WriteAESDecEnc, [HWPort5]> {
  let Latency = 7;
  let ResourceCycles = [1];
}
def : WriteRes<WriteAESDecEncLd, [HWPort5, HWPort23]> {
  let Latency = 7;
  let ResourceCycles = [1, 1];
}

def : WriteRes<WriteAESIMC, [HWPort5]> {
  let Latency = 14;
  let ResourceCycles = [2];
}
def : WriteRes<WriteAESIMCLd, [HWPort5, HWPort23]> {
  let Latency = 14;
  let ResourceCycles = [2, 1];
}

def : WriteRes<WriteAESKeyGen, [HWPort0, HWPort5]> {
  let Latency = 10;
  let ResourceCycles = [2, 8];
}
def : WriteRes<WriteAESKeyGenLd, [HWPort0, HWPort5, HWPort23]> {
  let Latency = 10;
  let ResourceCycles = [2, 7, 1];
}

// Carry-less multiplication instructions.
def : WriteRes<WriteCLMul, [HWPort0, HWPort5]> {
  let Latency = 7;
  let ResourceCycles = [2, 1];
}
def : WriteRes<WriteCLMulLd, [HWPort0, HWPort5, HWPort23]> {
  let Latency = 7;
  let ResourceCycles = [2, 1, 1];
}

def : WriteRes<WriteSystem,     [HWPort0156]> { let Latency = 100; }
def : WriteRes<WriteMicrocoded, [HWPort0156]> { let Latency = 100; }
def : WriteRes<WriteFence,  [HWPort23, HWPort4]>;
def : WriteRes<WriteNop, []>;

//================ Exceptions ================//

//-- Specific Scheduling Models --//
def Write2P0156_Lat2 : SchedWriteRes<[HWPort0156]> {
  let Latency = 2;
  let ResourceCycles = [2];
}
def Write2P0156_Lat2Ld : SchedWriteRes<[HWPort0156, HWPort23]> {
  let Latency = 6;
  let ResourceCycles = [2, 1];
}

def Write2P237_P4 : SchedWriteRes<[HWPort237, HWPort4]> {
  let Latency = 1;
  let ResourceCycles = [2, 1];
}

def WriteP06 : SchedWriteRes<[HWPort06]>;

// Notation:
// - r: register.
// - mm: 64 bit mmx register.
// - x = 128 bit xmm register.
// - (x)mm = mmx or xmm register.
// - y = 256 bit ymm register.
// - v = any vector register.
// - m = memory.

//=== Integer Instructions ===//
//-- Move instructions --//

// MOV.
// r16,m.
def : InstRW<[WriteALULd], (instregex "MOV16rm")>;

// MOVSX, MOVZX.
// r,m.
def : InstRW<[WriteLoad], (instregex "MOV(S|Z)X32rm(8|16)")>;

// CMOVcc.
// r,r.
def : InstRW<[Write2P0156_Lat2],
      (instregex "CMOV(O|NO|B|AE|E|NE|BE|A|S|NS|P|NP|L|GE|LE|G)(16|32|64)rr")>;
// r,m.
def : InstRW<[Write2P0156_Lat2Ld, ReadAfterLd],
      (instregex "CMOV(O|NO|B|AE|E|NE|BE|A|S|NS|P|NP|L|GE|LE|G)(16|32|64)rm")>;

// XCHG.
// r,r.
def WriteXCHG : SchedWriteRes<[HWPort0156]> {
  let Latency = 2;
  let ResourceCycles = [3];
}

def : InstRW<[WriteXCHG], (instregex "XCHG(8|16|32|64)rr", "XCHG(16|32|64)ar")>;

// r,m.
def WriteXCHGrm : SchedWriteRes<[]> {
  let Latency = 21;
  let NumMicroOps = 8;
}
def : InstRW<[WriteXCHGrm], (instregex "XCHG(8|16|32|64)rm")>;

// XLAT.
def WriteXLAT : SchedWriteRes<[]> {
  let Latency = 7;
  let NumMicroOps = 3;
}
def : InstRW<[WriteXLAT], (instregex "XLAT")>;

// PUSH.
// m.
def : InstRW<[Write2P237_P4], (instregex "PUSH(16|32)rmm")>;

// PUSHF.
def WritePushF : SchedWriteRes<[HWPort1, HWPort4, HWPort237, HWPort06]> {
  let NumMicroOps = 4;
}
def : InstRW<[WritePushF], (instregex "PUSHF(16|32)")>;

// PUSHA.
def WritePushA : SchedWriteRes<[]> {
  let NumMicroOps = 19;
}
def : InstRW<[WritePushA], (instregex "PUSHA(16|32)")>;

// POP.
// m.
def : InstRW<[Write2P237_P4], (instregex "POP(16|32)rmm")>;

// POPF.
def WritePopF : SchedWriteRes<[]> {
  let NumMicroOps = 9;
}
def : InstRW<[WritePopF], (instregex "POPF(16|32)")>;

// POPA.
def WritePopA : SchedWriteRes<[]> {
  let NumMicroOps = 18;
}
def : InstRW<[WritePopA], (instregex "POPA(16|32)")>;

// LAHF SAHF.
def : InstRW<[WriteP06], (instregex "(S|L)AHF")>;

// BSWAP.
// r32.
def WriteBSwap32 : SchedWriteRes<[HWPort15]>;
def : InstRW<[WriteBSwap32], (instregex "BSWAP32r")>;

// r64.
def WriteBSwap64 : SchedWriteRes<[HWPort06, HWPort15]> {
  let NumMicroOps = 2;
}
def : InstRW<[WriteBSwap64], (instregex "BSWAP64r")>;

// MOVBE.
// r16,m16 / r64,m64.
def : InstRW<[Write2P0156_Lat2Ld], (instregex "MOVBE(16|64)rm")>;

// r32, m32.
def WriteMoveBE32rm : SchedWriteRes<[HWPort15, HWPort23]> {
  let NumMicroOps = 2;
}
def : InstRW<[WriteMoveBE32rm], (instregex "MOVBE32rm")>;

// m16,r16.
def WriteMoveBE16mr : SchedWriteRes<[HWPort06, HWPort237, HWPort4]> {
  let NumMicroOps = 3;
}
def : InstRW<[WriteMoveBE16mr], (instregex "MOVBE16mr")>;

// m32,r32.
def WriteMoveBE32mr : SchedWriteRes<[HWPort15, HWPort237, HWPort4]> {
  let NumMicroOps = 3;
}
def : InstRW<[WriteMoveBE32mr], (instregex "MOVBE32mr")>;

// m64,r64.
def WriteMoveBE64mr : SchedWriteRes<[HWPort06, HWPort15, HWPort237, HWPort4]> {
  let NumMicroOps = 4;
}
def : InstRW<[WriteMoveBE64mr], (instregex "MOVBE64mr")>;

} // SchedModel
