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
  multiclass PTXReadSpecialRegisterIntrinsic_v4i32<string prefix> {
// FIXME: Do we need the 128-bit integer type version?
//    def _r64   : Intrinsic<[llvm_i128_ty],   [], [IntrNoMem]>;

// FIXME: Enable this once v4i32 support is enabled in back-end.
//    def _v4i16 : Intrinsic<[llvm_v4i32_ty], [], [IntrNoMem]>;

    def _x     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
                 GCCBuiltin<!strconcat(prefix, "_x")>;
    def _y     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
                 GCCBuiltin<!strconcat(prefix, "_y")>;
    def _z     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
                 GCCBuiltin<!strconcat(prefix, "_z")>;
    def _w     : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
                 GCCBuiltin<!strconcat(prefix, "_w")>;
  }

  class PTXReadSpecialRegisterIntrinsic_r32<string name>
    : Intrinsic<[llvm_i32_ty], [], [IntrNoMem]>,
      GCCBuiltin<name>;

  class PTXReadSpecialRegisterIntrinsic_r64<string name>
    : Intrinsic<[llvm_i64_ty], [], [IntrNoMem]>,
      GCCBuiltin<name>;
}

defm int_ptx_read_tid        : PTXReadSpecialRegisterIntrinsic_v4i32
                               <"__builtin_ptx_read_tid">;
defm int_ptx_read_ntid       : PTXReadSpecialRegisterIntrinsic_v4i32
                               <"__builtin_ptx_read_ntid">;

def int_ptx_read_laneid      : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_laneid">;
def int_ptx_read_warpid      : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_warpid">;
def int_ptx_read_nwarpid     : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_nwarpid">;

defm int_ptx_read_ctaid      : PTXReadSpecialRegisterIntrinsic_v4i32
                               <"__builtin_ptx_read_ctaid">;
defm int_ptx_read_nctaid     : PTXReadSpecialRegisterIntrinsic_v4i32
                               <"__builtin_ptx_read_nctaid">;

def int_ptx_read_smid        : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_smid">;
def int_ptx_read_nsmid       : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_nsmid">;
def int_ptx_read_gridid      : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_gridid">;

def int_ptx_read_lanemask_eq : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_lanemask_eq">;
def int_ptx_read_lanemask_le : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_lanemask_le">;
def int_ptx_read_lanemask_lt : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_lanemask_lt">;
def int_ptx_read_lanemask_ge : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_lanemask_ge">;
def int_ptx_read_lanemask_gt : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_lanemask_gt">;

def int_ptx_read_clock       : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_clock">;
def int_ptx_read_clock64     : PTXReadSpecialRegisterIntrinsic_r64
                               <"__builtin_ptx_read_clock64">;

def int_ptx_read_pm0         : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_pm0">;
def int_ptx_read_pm1         : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_pm1">;
def int_ptx_read_pm2         : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_pm2">;
def int_ptx_read_pm3         : PTXReadSpecialRegisterIntrinsic_r32
                               <"__builtin_ptx_read_pm3">;

let TargetPrefix = "ptx" in
  def int_ptx_bar_sync : Intrinsic<[], [llvm_i32_ty], []>,
                         GCCBuiltin<"__builtin_ptx_bar_sync">;
