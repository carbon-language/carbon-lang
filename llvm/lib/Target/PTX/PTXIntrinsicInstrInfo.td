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

class PTX_READ_SPECIAL_REGISTER_R64<string regname, Intrinsic intop>
  : InstPTX<(outs RegI64:$d), (ins),
            !strconcat("mov.u64\t$d, %", regname),
            [(set RegI64:$d, (intop))]>;

class PTX_READ_SPECIAL_REGISTER_R32<string regname, Intrinsic intop>
  : InstPTX<(outs RegI32:$d), (ins),
            !strconcat("mov.u32\t$d, %", regname),
            [(set RegI32:$d, (intop))]>;

// TODO Add read vector-version of special registers

//def PTX_READ_TID_R64 : PTX_READ_SPECIAL_REGISTER_R64<"tid",
//                                                     int_ptx_read_tid_r64>;
def PTX_READ_TID_X   : PTX_READ_SPECIAL_REGISTER_R32<"tid.x",
                                                     int_ptx_read_tid_x>;
def PTX_READ_TID_Y   : PTX_READ_SPECIAL_REGISTER_R32<"tid.y",
                                                     int_ptx_read_tid_y>;
def PTX_READ_TID_Z   : PTX_READ_SPECIAL_REGISTER_R32<"tid.z",
                                                     int_ptx_read_tid_z>;
def PTX_READ_TID_W   : PTX_READ_SPECIAL_REGISTER_R32<"tid.w",
                                                     int_ptx_read_tid_w>;

//def PTX_READ_NTID_R64 : PTX_READ_SPECIAL_REGISTER_R64<"ntid",
//                                                      int_ptx_read_ntid_r64>;
def PTX_READ_NTID_X   : PTX_READ_SPECIAL_REGISTER_R32<"ntid.x",
                                                      int_ptx_read_ntid_x>;
def PTX_READ_NTID_Y   : PTX_READ_SPECIAL_REGISTER_R32<"ntid.y",
                                                      int_ptx_read_ntid_y>;
def PTX_READ_NTID_Z   : PTX_READ_SPECIAL_REGISTER_R32<"ntid.z",
                                                      int_ptx_read_ntid_z>;
def PTX_READ_NTID_W   : PTX_READ_SPECIAL_REGISTER_R32<"ntid.w",
                                                      int_ptx_read_ntid_w>;

def PTX_READ_LANEID  : PTX_READ_SPECIAL_REGISTER_R32<"laneid",
                                                     int_ptx_read_laneid>;
def PTX_READ_WARPID  : PTX_READ_SPECIAL_REGISTER_R32<"warpid",
                                                     int_ptx_read_warpid>;
def PTX_READ_NWARPID : PTX_READ_SPECIAL_REGISTER_R32<"nwarpid",
                                                     int_ptx_read_nwarpid>;

//def PTX_READ_CTAID_R64 :
//PTX_READ_SPECIAL_REGISTER_R64<"ctaid", int_ptx_read_ctaid_r64>;
def PTX_READ_CTAID_X   : PTX_READ_SPECIAL_REGISTER_R32<"ctaid.x",
                                                       int_ptx_read_ctaid_x>;
def PTX_READ_CTAID_Y   : PTX_READ_SPECIAL_REGISTER_R32<"ctaid.y",
                                                       int_ptx_read_ctaid_y>;
def PTX_READ_CTAID_Z   : PTX_READ_SPECIAL_REGISTER_R32<"ctaid.z",
                                                       int_ptx_read_ctaid_z>;
def PTX_READ_CTAID_W   : PTX_READ_SPECIAL_REGISTER_R32<"ctaid.w",
                                                       int_ptx_read_ctaid_w>;

//def PTX_READ_NCTAID_R64 :
//PTX_READ_SPECIAL_REGISTER_R64<"nctaid", int_ptx_read_nctaid_r64>;
def PTX_READ_NCTAID_X   : PTX_READ_SPECIAL_REGISTER_R32<"nctaid.x",
                                                        int_ptx_read_nctaid_x>;
def PTX_READ_NCTAID_Y   : PTX_READ_SPECIAL_REGISTER_R32<"nctaid.y",
                                                        int_ptx_read_nctaid_y>;
def PTX_READ_NCTAID_Z   : PTX_READ_SPECIAL_REGISTER_R32<"nctaid.z",
                                                        int_ptx_read_nctaid_z>;
def PTX_READ_NCTAID_W   : PTX_READ_SPECIAL_REGISTER_R32<"nctaid.w",
                                                        int_ptx_read_nctaid_w>;

def PTX_READ_SMID  : PTX_READ_SPECIAL_REGISTER_R32<"smid",
                                                   int_ptx_read_smid>;
def PTX_READ_NSMID  : PTX_READ_SPECIAL_REGISTER_R32<"nsmid",
                                                    int_ptx_read_nsmid>;
def PTX_READ_GRIDID  : PTX_READ_SPECIAL_REGISTER_R32<"gridid",
                                                     int_ptx_read_gridid>;

def PTX_READ_LANEMASK_EQ
  : PTX_READ_SPECIAL_REGISTER_R32<"lanemask_eq", int_ptx_read_lanemask_eq>;
def PTX_READ_LANEMASK_LE
  : PTX_READ_SPECIAL_REGISTER_R32<"lanemask_le", int_ptx_read_lanemask_le>;
def PTX_READ_LANEMASK_LT
  : PTX_READ_SPECIAL_REGISTER_R32<"lanemask_lt", int_ptx_read_lanemask_lt>;
def PTX_READ_LANEMASK_GE
  : PTX_READ_SPECIAL_REGISTER_R32<"lanemask_ge", int_ptx_read_lanemask_ge>;
def PTX_READ_LANEMASK_GT
  : PTX_READ_SPECIAL_REGISTER_R32<"lanemask_gt", int_ptx_read_lanemask_gt>;

def PTX_READ_CLOCK
  : PTX_READ_SPECIAL_REGISTER_R32<"clock", int_ptx_read_clock>;
def PTX_READ_CLOCK64
  : PTX_READ_SPECIAL_REGISTER_R64<"clock64", int_ptx_read_clock64>;

def PTX_READ_PM0 : PTX_READ_SPECIAL_REGISTER_R32<"pm0", int_ptx_read_pm0>;
def PTX_READ_PM1 : PTX_READ_SPECIAL_REGISTER_R32<"pm1", int_ptx_read_pm1>;
def PTX_READ_PM2 : PTX_READ_SPECIAL_REGISTER_R32<"pm2", int_ptx_read_pm2>;
def PTX_READ_PM3 : PTX_READ_SPECIAL_REGISTER_R32<"pm3", int_ptx_read_pm3>;

// PTX Parallel Synchronization and Communication Intrinsics

def PTX_BAR_SYNC : InstPTX<(outs), (ins i32imm:$i), "bar.sync\t$i",
                           [(int_ptx_bar_sync imm:$i)]>;
