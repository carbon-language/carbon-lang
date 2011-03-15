//===- IntrinsicsPTX.td - Defines PTX intrinsics -----------*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the PTX-specific intrinsics.
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "ptx" in {
  multiclass PTXReadSpecialRegisterIntrinsic_v4i32 {
// FIXME: Do we need the 128-bit integer type version?
//    def _r64   : Intrinsic<[llvm_i128_ty],   [], [IntrNoMem]>;

// FIXME: Enable this once v4i32 support is enabled in back-end.
//    def _v4i16 : Intrinsic<[llvm_v4i32_ty], [], [IntrNoMem]>;

    def _x     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;
    def _y     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;
    def _z     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;
    def _w     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;
  }

  class PTXReadSpecialRegisterIntrinsic_r32
    : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>;

  class PTXReadSpecialRegisterIntrinsic_r64
    : Intrinsic<[llvm_i64_ty], [], [IntrNoMem]>;
}

defm int_ptx_read_tid        : PTXReadSpecialRegisterIntrinsic_v4i32;
defm int_ptx_read_ntid       : PTXReadSpecialRegisterIntrinsic_v4i32;

def int_ptx_read_laneid      : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_warpid      : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_nwarpid     : PTXReadSpecialRegisterIntrinsic_r32;

defm int_ptx_read_ctaid      : PTXReadSpecialRegisterIntrinsic_v4i32;
defm int_ptx_read_nctaid     : PTXReadSpecialRegisterIntrinsic_v4i32;

def int_ptx_read_smid        : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_nsmid       : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_gridid      : PTXReadSpecialRegisterIntrinsic_r32;

def int_ptx_read_lanemask_eq : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_lanemask_le : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_lanemask_lt : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_lanemask_ge : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_lanemask_gt : PTXReadSpecialRegisterIntrinsic_r32;

def int_ptx_read_clock       : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_clock64     : PTXReadSpecialRegisterIntrinsic_r64;

def int_ptx_read_pm0         : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_pm1         : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_pm2         : PTXReadSpecialRegisterIntrinsic_r32;
def int_ptx_read_pm3         : PTXReadSpecialRegisterIntrinsic_r32;

let TargetPrefix = "ptx" in
  def int_ptx_bar_sync : Intrinsic<[], [llvm_i32_ty], []>;
