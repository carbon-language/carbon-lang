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
  multiclass PTXReadSpecialRegisterIntrinsic {
    def _r64   : Intrinsic<[llvm_i64_ty],   [], [IntrNoMem]>;
    def _v4i16 : Intrinsic<[llvm_v4i16_ty], [], [IntrNoMem]>;
  }

  multiclass PTXReadSpecialSubRegisterIntrinsic {
    def _x : Intrinsic<[llvm_i16_ty], [], [IntrNoMem]>;
    def _y : Intrinsic<[llvm_i16_ty], [], [IntrNoMem]>;
    def _z : Intrinsic<[llvm_i16_ty], [], [IntrNoMem]>;
    def _w : Intrinsic<[llvm_i16_ty], [], [IntrNoMem]>;
  }
}

defm int_ptx_read_tid    : PTXReadSpecialRegisterIntrinsic;
defm int_ptx_read_tid    : PTXReadSpecialSubRegisterIntrinsic;
defm int_ptx_read_ntid   : PTXReadSpecialRegisterIntrinsic;
defm int_ptx_read_ntid   : PTXReadSpecialSubRegisterIntrinsic;
defm int_ptx_read_ctaid  : PTXReadSpecialRegisterIntrinsic;
defm int_ptx_read_ctaid  : PTXReadSpecialSubRegisterIntrinsic;
defm int_ptx_read_nctaid : PTXReadSpecialRegisterIntrinsic;
defm int_ptx_read_nctaid : PTXReadSpecialSubRegisterIntrinsic;

let TargetPrefix = "ptx" in
  def int_ptx_bar_sync : Intrinsic<[], [llvm_i32_ty], []>;
