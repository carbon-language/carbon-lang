//===-- SISchedule.td - SI Scheduling definitons -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// MachineModel definitions for Southern Islands (SI)
//
//===----------------------------------------------------------------------===//

def WriteBranch : SchedWrite;
def WriteExport : SchedWrite;
def WriteLDS    : SchedWrite;
def WriteSALU   : SchedWrite;
def WriteSMEM   : SchedWrite;
def WriteVMEM   : SchedWrite;
def WriteBarrier : SchedWrite;

// Vector ALU instructions
def Write32Bit         : SchedWrite;
def WriteQuarterRate32 : SchedWrite;

def WriteFloatFMA   : SchedWrite;

def WriteDouble     : SchedWrite;
def WriteDoubleAdd  : SchedWrite;

def SIFullSpeedModel : SchedMachineModel;
def SIQuarterSpeedModel : SchedMachineModel;

// BufferSize = 0 means the processors are in-order.
let BufferSize = 0 in {

// XXX: Are the resource counts correct?
def HWBranch : ProcResource<1>;
def HWExport : ProcResource<7>;   // Taken from S_WAITCNT
def HWLGKM   : ProcResource<31>;  // Taken from S_WAITCNT
def HWSALU   : ProcResource<1>;
def HWVMEM   : ProcResource<15>;  // Taken from S_WAITCNT
def HWVALU   : ProcResource<1>;

}

class HWWriteRes<SchedWrite write, list<ProcResourceKind> resources,
                 int latency> : WriteRes<write, resources> {
  let Latency = latency;
}

class HWVALUWriteRes<SchedWrite write, int latency> :
  HWWriteRes<write, [HWVALU], latency>;


// The latency numbers are taken from AMD Accelerated Parallel Processing
// guide.  They may not be acurate.

// The latency values are 1 / (operations / cycle) / 4.
multiclass SICommonWriteRes {

  def : HWWriteRes<WriteBranch,  [HWBranch], 100>; // XXX: Guessed ???
  def : HWWriteRes<WriteExport,  [HWExport], 100>; // XXX: Guessed ???
  def : HWWriteRes<WriteLDS,     [HWLGKM],    32>; // 2 - 64
  def : HWWriteRes<WriteSALU,    [HWSALU],     1>;
  def : HWWriteRes<WriteSMEM,    [HWLGKM],    10>; // XXX: Guessed ???
  def : HWWriteRes<WriteVMEM,    [HWVMEM],   450>; // 300 - 600
  def : HWWriteRes<WriteBarrier, [HWBranch], 500>; // XXX: Guessed ???

  def : HWVALUWriteRes<Write32Bit,         1>;
  def : HWVALUWriteRes<WriteQuarterRate32, 4>;
}


let SchedModel = SIFullSpeedModel in {

defm : SICommonWriteRes;

def : HWVALUWriteRes<WriteFloatFMA,   1>;
def : HWVALUWriteRes<WriteDouble,     4>;
def : HWVALUWriteRes<WriteDoubleAdd,  2>;

} // End SchedModel = SIFullSpeedModel

let SchedModel = SIQuarterSpeedModel in {

defm : SICommonWriteRes;

def : HWVALUWriteRes<WriteFloatFMA, 16>;
def : HWVALUWriteRes<WriteDouble,   16>;
def : HWVALUWriteRes<WriteDoubleAdd, 8>;

}  // End SchedModel = SIQuarterSpeedModel
