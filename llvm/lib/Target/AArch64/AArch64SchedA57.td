//=- AArch64SchedA57.td - ARM Cortex-A57 Scheduling Defs -----*- tablegen -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the machine model for ARM Cortex-A57 to support
// instruction scheduling and other instruction cost heuristics.
//
//===----------------------------------------------------------------------===//

def CortexA57Model : SchedMachineModel {
  let IssueWidth        =   8; // 3-way decode and 8-way issue
  let MicroOpBufferSize = 128; // 128 micro-op re-order buffer
  let LoadLatency       =   4; // Optimistic load latency
  let MispredictPenalty =  14; // Fetch + Decode/Rename/Dispatch + Branch
}

//===----------------------------------------------------------------------===//
// Define each kind of processor resource and number available on Cortex-A57.
// Cortex A-57 has 8 pipelines that each has its own 8-entry queue where
// micro-ops wait for their operands and then issue out-of-order.

def A57UnitB : ProcResource<1> { let BufferSize = 8; }  // Type B micro-ops
def A57UnitI : ProcResource<2> { let BufferSize = 8; }  // Type I micro-ops
def A57UnitM : ProcResource<1> { let BufferSize = 8; }  // Type M micro-ops
def A57UnitL : ProcResource<1> { let BufferSize = 8; }  // Type L micro-ops
def A57UnitS : ProcResource<1> { let BufferSize = 8; }  // Type S micro-ops
def A57UnitX : ProcResource<1> { let BufferSize = 8; }  // Type X micro-ops
def A57UnitW : ProcResource<1> { let BufferSize = 8; }  // Type W micro-ops
let SchedModel = CortexA57Model in {
  def A57UnitV : ProcResGroup<[A57UnitX, A57UnitW]>;    // Type V micro-ops
}


let SchedModel = CortexA57Model in {

//===----------------------------------------------------------------------===//
// Define customized scheduler read/write types specific to the Cortex-A57.

include "AArch64SchedA57WriteRes.td"

//===----------------------------------------------------------------------===//
// Map the target-defined scheduler read/write resources and latency for
// Cortex-A57. The Cortex-A57 types are directly associated with resources, so
// defining the aliases precludes the need for mapping them using WriteRes. The
// aliases are sufficient for creating a coarse, working model. As the model
// evolves, InstRWs will be used to override these SchedAliases.

def : SchedAlias<WriteImm,   A57Write_1cyc_1I>;
def : SchedAlias<WriteI,     A57Write_1cyc_1I>;
def : SchedAlias<WriteISReg, A57Write_2cyc_1M>;
def : SchedAlias<WriteIEReg, A57Write_2cyc_1M>;
def : SchedAlias<WriteExtr,  A57Write_1cyc_1I>;
def : SchedAlias<WriteIS,    A57Write_1cyc_1I>;
def : SchedAlias<WriteID32,  A57Write_19cyc_1M>;
def : SchedAlias<WriteID64,  A57Write_35cyc_1M>;
def : SchedAlias<WriteIM32,  A57Write_3cyc_1M>;
def : SchedAlias<WriteIM64,  A57Write_5cyc_1M>;
def : SchedAlias<WriteBr,    A57Write_1cyc_1B>;
def : SchedAlias<WriteBrReg, A57Write_1cyc_1B>;
def : SchedAlias<WriteLD,    A57Write_4cyc_1L>;
def : SchedAlias<WriteST,    A57Write_1cyc_1S>;
def : SchedAlias<WriteSTP,   A57Write_1cyc_1S>;
def : SchedAlias<WriteAdr,   A57Write_1cyc_1I>;
def : SchedAlias<WriteLDIdx, A57Write_4cyc_1I_1L>;
def : SchedAlias<WriteSTIdx, A57Write_1cyc_1I_1S>;
def : SchedAlias<WriteF,     A57Write_3cyc_1V>;
def : SchedAlias<WriteFCmp,  A57Write_3cyc_1V>;
def : SchedAlias<WriteFCvt,  A57Write_5cyc_1V>;
def : SchedAlias<WriteFCopy, A57Write_3cyc_1V>;
def : SchedAlias<WriteFImm,  A57Write_3cyc_1V>;
def : SchedAlias<WriteFMul,  A57Write_5cyc_1V>;
def : SchedAlias<WriteFDiv,  A57Write_18cyc_1X>;
def : SchedAlias<WriteV,     A57Write_3cyc_1V>;
def : SchedAlias<WriteVLD,   A57Write_5cyc_1L>;
def : SchedAlias<WriteVST,   A57Write_1cyc_1S>;

def : WriteRes<WriteSys,     []> { let Latency = 1; }
def : WriteRes<WriteBarrier, []> { let Latency = 1; }
def : WriteRes<WriteHint,    []> { let Latency = 1; }

def : WriteRes<WriteLDHi,    []> { let Latency = 4; }

// Forwarding logic is not [yet] explicitly modeled beyond what is captured
// in the latencies of the A57 Generic SchedWriteRes's.
def : ReadAdvance<ReadI,       0>;
def : ReadAdvance<ReadISReg,   0>;
def : ReadAdvance<ReadIEReg,   0>;
def : ReadAdvance<ReadIM,      0>;
def : ReadAdvance<ReadIMA,     0>;
def : ReadAdvance<ReadID,      0>;
def : ReadAdvance<ReadExtrHi,  0>;
def : ReadAdvance<ReadAdrBase, 0>;
def : ReadAdvance<ReadVLD,     0>;


//===----------------------------------------------------------------------===//
// Specialize the coarse model by associating instruction groups with the
// subtarget-defined types. As the modeled is refined, this will override most
// of the above ShchedAlias mappings.

// Miscellaneous
// -----------------------------------------------------------------------------

def : InstRW<[WriteI], (instrs COPY)>;


// Branch Instructions
// -----------------------------------------------------------------------------

def : InstRW<[A57Write_1cyc_1B_1I], (instrs BL)>;
def : InstRW<[A57Write_2cyc_1B_1I], (instrs BLR)>;


// Divide and Multiply Instructions
// -----------------------------------------------------------------------------

// Multiply high
def : InstRW<[A57Write_6cyc_1M], (instrs SMULHrr, UMULHrr)>;


// Miscellaneous Data-Processing Instructions
// -----------------------------------------------------------------------------

def : InstRW<[A57Write_1cyc_1I],    (instrs EXTRWrri)>;
def : InstRW<[A57Write_3cyc_1I_1M], (instrs EXTRXrri)>;
def : InstRW<[A57Write_2cyc_1M],    (instregex "BFM")>;


// Cryptography Extensions
// -----------------------------------------------------------------------------

def : InstRW<[A57Write_3cyc_1W], (instregex "CRC32")>;


// Vector Load
// -----------------------------------------------------------------------------

def : InstRW<[A57Write_8cyc_1L_1V],           (instregex "LD1i(8|16|32)$")>;
def : InstRW<[A57Write_8cyc_1L_1V, WriteAdr], (instregex "LD1i(8|16|32)_POST$")>;
def : InstRW<[A57Write_5cyc_1L],            (instregex "LD1i(64)$")>;
def : InstRW<[A57Write_5cyc_1L, WriteAdr],  (instregex "LD1i(64)_POST$")>;

def : InstRW<[A57Write_8cyc_1L_1V],           (instregex "LD1Rv(8b|4h|2s)$")>;
def : InstRW<[A57Write_8cyc_1L_1V, WriteAdr], (instregex "LD1Rv(8b|4h|2s)_POST$")>;
def : InstRW<[A57Write_5cyc_1L],            (instregex "LD1Rv(1d)$")>;
def : InstRW<[A57Write_5cyc_1L, WriteAdr],  (instregex "LD1Rv(1d)_POST$")>;
def : InstRW<[A57Write_8cyc_1L_1V],           (instregex "LD1Rv(16b|8h|4s|2d)$")>;
def : InstRW<[A57Write_8cyc_1L_1V, WriteAdr], (instregex "LD1Rv(16b|8h|4s|2d)_POST$")>;

def : InstRW<[A57Write_5cyc_1L],              (instregex "LD1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[A57Write_5cyc_1L, WriteAdr],    (instregex "LD1Onev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[A57Write_5cyc_1L],              (instregex "LD1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[A57Write_5cyc_1L, WriteAdr],    (instregex "LD1Onev(16b|8h|4s|2d)_POST$")>;
def : InstRW<[A57Write_5cyc_1L],              (instregex "LD1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[A57Write_5cyc_1L, WriteAdr],    (instregex "LD1Twov(8b|4h|2s|1d)_POST$")>;
def : InstRW<[A57Write_6cyc_2L],             (instregex "LD1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[A57Write_6cyc_2L, WriteAdr],   (instregex "LD1Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[A57Write_6cyc_2L],             (instregex "LD1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[A57Write_6cyc_2L, WriteAdr],   (instregex "LD1Threev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[A57Write_7cyc_3L],            (instregex "LD1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[A57Write_7cyc_3L, WriteAdr],  (instregex "LD1Threev(16b|8h|4s|2d)_POST$")>;
def : InstRW<[A57Write_6cyc_2L],             (instregex "LD1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[A57Write_6cyc_2L, WriteAdr],   (instregex "LD1Fourv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[A57Write_8cyc_4L],           (instregex "LD1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[A57Write_8cyc_4L, WriteAdr], (instregex "LD1Fourv(16b|8h|4s|2d)_POST$")>;

def : InstRW<[A57Write_8cyc_1L_2V],           (instregex "LD2i(8|16)$")>;
def : InstRW<[A57Write_8cyc_1L_2V, WriteAdr], (instregex "LD2i(8|16)_POST$")>;
def : InstRW<[A57Write_6cyc_2L],            (instregex "LD2i(32)$")>;
def : InstRW<[A57Write_6cyc_2L, WriteAdr],  (instregex "LD2i(32)_POST$")>;
def : InstRW<[A57Write_8cyc_1L_1V],            (instregex "LD2i(64)$")>;
def : InstRW<[A57Write_8cyc_1L_1V, WriteAdr],  (instregex "LD2i(64)_POST$")>;

def : InstRW<[A57Write_8cyc_1L_1V],            (instregex "LD2Rv(8b|4h|2s)$")>;
def : InstRW<[A57Write_8cyc_1L_1V, WriteAdr],  (instregex "LD2Rv(8b|4h|2s)_POST$")>;
def : InstRW<[A57Write_5cyc_1L],             (instregex "LD2Rv(1d)$")>;
def : InstRW<[A57Write_5cyc_1L, WriteAdr],   (instregex "LD2Rv(1d)_POST$")>;
def : InstRW<[A57Write_8cyc_1L_2V],           (instregex "LD2Rv(16b|8h|4s|2d)$")>;
def : InstRW<[A57Write_8cyc_1L_2V, WriteAdr], (instregex "LD2Rv(16b|8h|4s|2d)_POST$")>;

def : InstRW<[A57Write_8cyc_1L_1V],             (instregex "LD2Twov(8b|4h|2s)$")>;
def : InstRW<[A57Write_8cyc_1L_1V, WriteAdr],   (instregex "LD2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[A57Write_9cyc_2L_2V],           (instregex "LD2Twov(16b|8h|4s)$")>;
def : InstRW<[A57Write_9cyc_2L_2V, WriteAdr], (instregex "LD2Twov(16b|8h|4s)_POST$")>;
def : InstRW<[A57Write_6cyc_2L],             (instregex "LD2Twov(2d)$")>;
def : InstRW<[A57Write_6cyc_2L, WriteAdr],   (instregex "LD2Twov(2d)_POST$")>;

def : InstRW<[A57Write_9cyc_1L_3V],           (instregex "LD3i(8|16)$")>;
def : InstRW<[A57Write_9cyc_1L_3V, WriteAdr], (instregex "LD3i(8|16)_POST$")>;
def : InstRW<[A57Write_8cyc_1L_2V],            (instregex "LD3i(32)$")>;
def : InstRW<[A57Write_8cyc_1L_2V, WriteAdr],  (instregex "LD3i(32)_POST$")>;
def : InstRW<[A57Write_6cyc_2L],             (instregex "LD3i(64)$")>;
def : InstRW<[A57Write_6cyc_2L, WriteAdr],   (instregex "LD3i(64)_POST$")>;

def : InstRW<[A57Write_8cyc_1L_2V],             (instregex "LD3Rv(8b|4h|2s)$")>;
def : InstRW<[A57Write_8cyc_1L_2V, WriteAdr],   (instregex "LD3Rv(8b|4h|2s)_POST$")>;
def : InstRW<[A57Write_6cyc_2L],              (instregex "LD3Rv(1d)$")>;
def : InstRW<[A57Write_6cyc_2L, WriteAdr],    (instregex "LD3Rv(1d)_POST$")>;
def : InstRW<[A57Write_9cyc_1L_3V],            (instregex "LD3Rv(16b|8h|4s)$")>;
def : InstRW<[A57Write_9cyc_1L_3V, WriteAdr],  (instregex "LD3Rv(16b|8h|4s)_POST$")>;
def : InstRW<[A57Write_9cyc_2L_3V],           (instregex "LD3Rv(2d)$")>;
def : InstRW<[A57Write_9cyc_2L_3V, WriteAdr], (instregex "LD3Rv(2d)_POST$")>;

def : InstRW<[A57Write_9cyc_2L_2V],               (instregex "LD3Threev(8b|4h|2s)$")>;
def : InstRW<[A57Write_9cyc_2L_2V, WriteAdr],     (instregex "LD3Threev(8b|4h|2s)_POST$")>;
def : InstRW<[A57Write_10cyc_3L_4V],           (instregex "LD3Threev(16b|8h|4s)$")>;
def : InstRW<[A57Write_10cyc_3L_4V, WriteAdr], (instregex "LD3Threev(16b|8h|4s)_POST$")>;
def : InstRW<[A57Write_8cyc_4L],               (instregex "LD3Threev(2d)$")>;
def : InstRW<[A57Write_8cyc_4L, WriteAdr],     (instregex "LD3Threev(2d)_POST$")>;

def : InstRW<[A57Write_9cyc_2L_3V],           (instregex "LD4i(8|16)$")>;
def : InstRW<[A57Write_9cyc_2L_3V, WriteAdr], (instregex "LD4i(8|16)_POST$")>;
def : InstRW<[A57Write_8cyc_1L_2V],             (instregex "LD4i(32)$")>;
def : InstRW<[A57Write_8cyc_1L_2V, WriteAdr],   (instregex "LD4i(32)_POST$")>;
def : InstRW<[A57Write_9cyc_2L_3V],           (instregex "LD4i(64)$")>;
def : InstRW<[A57Write_9cyc_2L_3V, WriteAdr], (instregex "LD4i(64)_POST$")>;

def : InstRW<[A57Write_8cyc_1L_2V],              (instregex "LD4Rv(8b|4h|2s)$")>;
def : InstRW<[A57Write_8cyc_1L_2V, WriteAdr],    (instregex "LD4Rv(8b|4h|2s)_POST$")>;
def : InstRW<[A57Write_6cyc_2L],               (instregex "LD4Rv(1d)$")>;
def : InstRW<[A57Write_6cyc_2L, WriteAdr],     (instregex "LD4Rv(1d)_POST$")>;
def : InstRW<[A57Write_9cyc_2L_3V],            (instregex "LD4Rv(16b|8h|4s)$")>;
def : InstRW<[A57Write_9cyc_2L_3V, WriteAdr],  (instregex "LD4Rv(16b|8h|4s)_POST$")>;
def : InstRW<[A57Write_9cyc_2L_4V],           (instregex "LD4Rv(2d)$")>;
def : InstRW<[A57Write_9cyc_2L_4V, WriteAdr], (instregex "LD4Rv(2d)_POST$")>;

def : InstRW<[A57Write_9cyc_2L_2V],                (instregex "LD4Fourv(8b|4h|2s)$")>;
def : InstRW<[A57Write_9cyc_2L_2V, WriteAdr],      (instregex "LD4Fourv(8b|4h|2s)_POST$")>;
def : InstRW<[A57Write_11cyc_4L_4V],           (instregex "LD4Fourv(16b|8h|4s)$")>;
def : InstRW<[A57Write_11cyc_4L_4V, WriteAdr], (instregex "LD4Fourv(16b|8h|4s)_POST$")>;
def : InstRW<[A57Write_8cyc_4L],                (instregex "LD4Fourv(2d)$")>;
def : InstRW<[A57Write_8cyc_4L, WriteAdr],      (instregex "LD4Fourv(2d)_POST$")>;

// Vector Store
// -----------------------------------------------------------------------------

def : InstRW<[A57Write_1cyc_1S],            (instregex "ST1i(8|16|32)$")>;
def : InstRW<[A57Write_1cyc_1S, WriteAdr],  (instregex "ST1i(8|16|32)_POST$")>;
def : InstRW<[A57Write_3cyc_1S_1V],           (instregex "ST1i(64)$")>;
def : InstRW<[A57Write_3cyc_1S_1V, WriteAdr], (instregex "ST1i(64)_POST$")>;

def : InstRW<[A57Write_1cyc_1S],                  (instregex "ST1Onev(8b|4h|2s|1d)$")>;
def : InstRW<[A57Write_1cyc_1S, WriteAdr],        (instregex "ST1Onev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[A57Write_2cyc_2S],                 (instregex "ST1Onev(16b|8h|4s|2d)$")>;
def : InstRW<[A57Write_2cyc_2S, WriteAdr],       (instregex "ST1Onev(16b|8h|4s|2d)_POST$")>;
def : InstRW<[A57Write_2cyc_2S],                 (instregex "ST1Twov(8b|4h|2s|1d)$")>;
def : InstRW<[A57Write_2cyc_2S, WriteAdr],       (instregex "ST1Twov(8b|4h|2s|1d)_POST$")>;
def : InstRW<[A57Write_4cyc_4S],               (instregex "ST1Twov(16b|8h|4s|2d)$")>;
def : InstRW<[A57Write_4cyc_4S, WriteAdr],     (instregex "ST1Twov(16b|8h|4s|2d)_POST$")>;
def : InstRW<[A57Write_3cyc_3S],                (instregex "ST1Threev(8b|4h|2s|1d)$")>;
def : InstRW<[A57Write_3cyc_3S, WriteAdr],      (instregex "ST1Threev(8b|4h|2s|1d)_POST$")>;
def : InstRW<[A57Write_6cyc_6S],             (instregex "ST1Threev(16b|8h|4s|2d)$")>;
def : InstRW<[A57Write_6cyc_6S, WriteAdr],   (instregex "ST1Threev(16b|8h|4s|2d)_POST$")>;
def : InstRW<[A57Write_4cyc_4S],               (instregex "ST1Fourv(8b|4h|2s|1d)$")>;
def : InstRW<[A57Write_4cyc_4S, WriteAdr],     (instregex "ST1Fourv(8b|4h|2s|1d)_POST$")>;
def : InstRW<[A57Write_8cyc_8S],           (instregex "ST1Fourv(16b|8h|4s|2d)$")>;
def : InstRW<[A57Write_8cyc_8S, WriteAdr], (instregex "ST1Fourv(16b|8h|4s|2d)_POST$")>;

def : InstRW<[A57Write_3cyc_1S_1V],           (instregex "ST2i(8|16|32)$")>;
def : InstRW<[A57Write_3cyc_1S_1V, WriteAdr], (instregex "ST2i(8|16|32)_POST$")>;
def : InstRW<[A57Write_2cyc_2S],           (instregex "ST2i(64)$")>;
def : InstRW<[A57Write_2cyc_2S, WriteAdr], (instregex "ST2i(64)_POST$")>;

def : InstRW<[A57Write_3cyc_2S_1V],              (instregex "ST2Twov(8b|4h|2s)$")>;
def : InstRW<[A57Write_3cyc_2S_1V, WriteAdr],    (instregex "ST2Twov(8b|4h|2s)_POST$")>;
def : InstRW<[A57Write_4cyc_4S_2V],           (instregex "ST2Twov(16b|8h|4s)$")>;
def : InstRW<[A57Write_4cyc_4S_2V, WriteAdr], (instregex "ST2Twov(16b|8h|4s)_POST$")>;
def : InstRW<[A57Write_4cyc_4S],             (instregex "ST2Twov(2d)$")>;
def : InstRW<[A57Write_4cyc_4S, WriteAdr],   (instregex "ST2Twov(2d)_POST$")>;

def : InstRW<[A57Write_3cyc_1S_1V],            (instregex "ST3i(8|16)$")>;
def : InstRW<[A57Write_3cyc_1S_1V, WriteAdr],  (instregex "ST3i(8|16)_POST$")>;
def : InstRW<[A57Write_3cyc_3S],           (instregex "ST3i(32)$")>;
def : InstRW<[A57Write_3cyc_3S, WriteAdr], (instregex "ST3i(32)_POST$")>;
def : InstRW<[A57Write_3cyc_2S_1V],           (instregex "ST3i(64)$")>;
def : InstRW<[A57Write_3cyc_2S_1V, WriteAdr], (instregex "ST3i(64)_POST$")>;

def : InstRW<[A57Write_3cyc_3S_2V],                 (instregex "ST3Threev(8b|4h|2s)$")>;
def : InstRW<[A57Write_3cyc_3S_2V, WriteAdr],       (instregex "ST3Threev(8b|4h|2s)_POST$")>;
def : InstRW<[A57Write_6cyc_6S_4V],           (instregex "ST3Threev(16b|8h|4s)$")>;
def : InstRW<[A57Write_6cyc_6S_4V, WriteAdr], (instregex "ST3Threev(16b|8h|4s)_POST$")>;
def : InstRW<[A57Write_6cyc_6S],                (instregex "ST3Threev(2d)$")>;
def : InstRW<[A57Write_6cyc_6S, WriteAdr],      (instregex "ST3Threev(2d)_POST$")>;

def : InstRW<[A57Write_3cyc_1S_1V],             (instregex "ST4i(8|16)$")>;
def : InstRW<[A57Write_3cyc_1S_1V, WriteAdr],   (instregex "ST4i(8|16)_POST$")>;
def : InstRW<[A57Write_4cyc_4S],           (instregex "ST4i(32)$")>;
def : InstRW<[A57Write_4cyc_4S, WriteAdr], (instregex "ST4i(32)_POST$")>;
def : InstRW<[A57Write_3cyc_2S_1V],            (instregex "ST4i(64)$")>;
def : InstRW<[A57Write_3cyc_2S_1V, WriteAdr],  (instregex "ST4i(64)_POST$")>;

def : InstRW<[A57Write_4cyc_4S_2V],                  (instregex "ST4Fourv(8b|4h|2s)$")>;
def : InstRW<[A57Write_4cyc_4S_2V, WriteAdr],        (instregex "ST4Fourv(8b|4h|2s)_POST$")>;
def : InstRW<[A57Write_8cyc_8S_4V],           (instregex "ST4Fourv(16b|8h|4s)$")>;
def : InstRW<[A57Write_8cyc_8S_4V, WriteAdr], (instregex "ST4Fourv(16b|8h|4s)_POST$")>;
def : InstRW<[A57Write_8cyc_8S],                (instregex "ST4Fourv(2d)$")>;
def : InstRW<[A57Write_8cyc_8S, WriteAdr],      (instregex "ST4Fourv(2d)_POST$")>;

} // SchedModel = CortexA57Model
