//===- PTXIntrinsicInstrInfo.td - Defines PTX intrinsics ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the PTX-specific intrinsic instructions.
//
//===----------------------------------------------------------------------===//

// PTX Special Purpose Register Accessor Intrinsics

class PTX_READ_SPECIAL_REGISTER<string regname, Intrinsic intop>
  : InstPTX<(outs RRegu64:$d), (ins),
            !strconcat("mov.u64\t$d, ", regname),
            [(set RRegu64:$d, (intop))]>;

class PTX_READ_SPECIAL_SUB_REGISTER<string regname, Intrinsic intop>
  : InstPTX<(outs RRegu16:$d), (ins),
            !strconcat("mov.u16\t$d, ", regname),
            [(set RRegu16:$d, (intop))]>;

def PTX_READ_TID_R64 : PTX_READ_SPECIAL_REGISTER<"tid", int_ptx_read_tid_r64>;
def PTX_READ_TID_X : PTX_READ_SPECIAL_SUB_REGISTER<"tid.x", int_ptx_read_tid_x>;
def PTX_READ_TID_Y : PTX_READ_SPECIAL_SUB_REGISTER<"tid.y", int_ptx_read_tid_y>;
def PTX_READ_TID_Z : PTX_READ_SPECIAL_SUB_REGISTER<"tid.z", int_ptx_read_tid_z>;
def PTX_READ_TID_W : PTX_READ_SPECIAL_SUB_REGISTER<"tid.w", int_ptx_read_tid_w>;

// PTX Parallel Synchronization and Communication Intrinsics

def PTX_BAR_SYNC : InstPTX<(outs), (ins i32imm:$i), "bar.sync\t$i",
                           [(int_ptx_bar_sync imm:$i)]>;
