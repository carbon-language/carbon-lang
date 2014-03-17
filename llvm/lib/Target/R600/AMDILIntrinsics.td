//===- AMDILIntrinsics.td - Defines AMDIL Intrinscs -*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// This file defines all of the amdil-specific intrinsics
//
//===---------------------------------------------------------------===//
//===--------------------------------------------------------------------===//
// Intrinsic classes
// Generic versions of the above classes but for Target specific intrinsics
// instead of SDNode patterns.
//===--------------------------------------------------------------------===//
let TargetPrefix = "AMDIL", isTarget = 1 in {
     class VoidIntLong :
          Intrinsic<[llvm_i64_ty], [], []>;
     class VoidIntInt :
          Intrinsic<[llvm_i32_ty], [], []>;
     class VoidIntBool :
          Intrinsic<[llvm_i32_ty], [], []>;
     class UnaryIntInt :
          Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>], [IntrNoMem]>;
     class UnaryIntFloat :
          Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>], [IntrNoMem]>;
     class ConvertIntFTOI :
          Intrinsic<[llvm_anyint_ty], [llvm_anyfloat_ty], [IntrNoMem]>;
     class ConvertIntITOF :
          Intrinsic<[llvm_anyfloat_ty], [llvm_anyint_ty], [IntrNoMem]>;
     class UnaryIntNoRetInt :
          Intrinsic<[], [llvm_anyint_ty], []>;
     class UnaryIntNoRetFloat :
          Intrinsic<[], [llvm_anyfloat_ty], []>;
     class BinaryIntInt :
          Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
     class BinaryIntFloat :
          Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
     class BinaryIntNoRetInt :
          Intrinsic<[], [llvm_anyint_ty, LLVMMatchType<0>], []>;
     class BinaryIntNoRetFloat :
          Intrinsic<[], [llvm_anyfloat_ty, LLVMMatchType<0>], []>;
     class TernaryIntInt :
          Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>,
          LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
     class TernaryIntFloat :
          Intrinsic<[llvm_anyfloat_ty], [LLVMMatchType<0>,
          LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
     class QuaternaryIntInt :
          Intrinsic<[llvm_anyint_ty], [LLVMMatchType<0>,
          LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>], [IntrNoMem]>;
     class UnaryAtomicInt :
          Intrinsic<[llvm_i32_ty], [llvm_ptr_ty, llvm_i32_ty], [IntrReadWriteArgMem]>;
     class BinaryAtomicInt :
          Intrinsic<[llvm_i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], [IntrReadWriteArgMem]>;
     class TernaryAtomicInt :
          Intrinsic<[llvm_i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty]>;
     class UnaryAtomicIntNoRet :
          Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty], [IntrReadWriteArgMem]>;
     class BinaryAtomicIntNoRet :
          Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty], [IntrReadWriteArgMem]>;
     class TernaryAtomicIntNoRet :
          Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty, llvm_i32_ty, llvm_i32_ty], [IntrReadWriteArgMem]>;
}

let TargetPrefix = "AMDIL", isTarget = 1 in {
  def int_AMDIL_abs : GCCBuiltin<"__amdil_abs">, UnaryIntInt;

  def int_AMDIL_bit_reverse_u32 : GCCBuiltin<"__amdil_ubit_reverse">,
          UnaryIntInt;
  def int_AMDIL_bit_count_i32 : GCCBuiltin<"__amdil_count_bits">,
          UnaryIntInt;
  def int_AMDIL_bit_find_first_lo : GCCBuiltin<"__amdil_ffb_lo">,
          UnaryIntInt;
  def int_AMDIL_bit_find_first_hi : GCCBuiltin<"__amdil_ffb_hi">,
          UnaryIntInt;
  def int_AMDIL_bit_find_first_sgn : GCCBuiltin<"__amdil_ffb_signed">,
          UnaryIntInt;
  def int_AMDIL_media_bitalign : GCCBuiltin<"__amdil_bitalign">,
                    TernaryIntInt;
  def int_AMDIL_media_bytealign : GCCBuiltin<"__amdil_bytealign">,
                    TernaryIntInt;
  def int_AMDIL_bit_insert_u32 : GCCBuiltin<"__amdil_ubit_insert">,
                    QuaternaryIntInt;
  def int_AMDIL_bfi : GCCBuiltin<"__amdil_bfi">,
      TernaryIntInt;
  def int_AMDIL_bfm : GCCBuiltin<"__amdil_bfm">,
      BinaryIntInt;
  def int_AMDIL_mulhi_i32 : GCCBuiltin<"__amdil_imul_high">,
          BinaryIntInt;
  def int_AMDIL_mulhi_u32 : GCCBuiltin<"__amdil_umul_high">,
          BinaryIntInt;
  def int_AMDIL_mul24_i32 : GCCBuiltin<"__amdil_imul24">,
          BinaryIntInt;
  def int_AMDIL_mul24_u32 : GCCBuiltin<"__amdil_umul24">,
          BinaryIntInt;
  def int_AMDIL_mulhi24_i32 : GCCBuiltin<"__amdil_imul24_high">,
          BinaryIntInt;
  def int_AMDIL_mulhi24_u32 : GCCBuiltin<"__amdil_umul24_high">,
          BinaryIntInt;
  def int_AMDIL_carry_i32 : GCCBuiltin<"__amdil_carry">,
          BinaryIntInt;
  def int_AMDIL_borrow_i32 : GCCBuiltin<"__amdil_borrow">,
          BinaryIntInt;
  def int_AMDIL_min_i32 : GCCBuiltin<"__amdil_imin">,
          BinaryIntInt;
  def int_AMDIL_min_u32 : GCCBuiltin<"__amdil_umin">,
          BinaryIntInt;
  def int_AMDIL_min     : GCCBuiltin<"__amdil_min">,
          BinaryIntFloat;
  def int_AMDIL_max_i32 : GCCBuiltin<"__amdil_imax">,
          BinaryIntInt;
  def int_AMDIL_max_u32 : GCCBuiltin<"__amdil_umax">,
          BinaryIntInt;
  def int_AMDIL_max     : GCCBuiltin<"__amdil_max">,
          BinaryIntFloat;
  def int_AMDIL_media_lerp_u4 : GCCBuiltin<"__amdil_u4lerp">,
          TernaryIntInt;
  def int_AMDIL_media_sad : GCCBuiltin<"__amdil_sad">,
          TernaryIntInt;
  def int_AMDIL_media_sad_hi : GCCBuiltin<"__amdil_sadhi">,
          TernaryIntInt;
  def int_AMDIL_fraction : GCCBuiltin<"__amdil_fraction">,
          UnaryIntFloat;
  def int_AMDIL_clamp : GCCBuiltin<"__amdil_clamp">,
          TernaryIntFloat;
  def int_AMDIL_pireduce : GCCBuiltin<"__amdil_pireduce">,
          UnaryIntFloat;
  def int_AMDIL_round_nearest : GCCBuiltin<"__amdil_round_nearest">,
          UnaryIntFloat;
  def int_AMDIL_round_neginf : GCCBuiltin<"__amdil_round_neginf">,
          UnaryIntFloat;
  def int_AMDIL_round_zero : GCCBuiltin<"__amdil_round_zero">,
          UnaryIntFloat;
  def int_AMDIL_acos : GCCBuiltin<"__amdil_acos">,
          UnaryIntFloat;
  def int_AMDIL_atan : GCCBuiltin<"__amdil_atan">,
          UnaryIntFloat;
  def int_AMDIL_asin : GCCBuiltin<"__amdil_asin">,
          UnaryIntFloat;
  def int_AMDIL_cos : GCCBuiltin<"__amdil_cos">,
          UnaryIntFloat;
  def int_AMDIL_cos_vec : GCCBuiltin<"__amdil_cos_vec">,
          UnaryIntFloat;
  def int_AMDIL_tan : GCCBuiltin<"__amdil_tan">,
          UnaryIntFloat;
  def int_AMDIL_sin : GCCBuiltin<"__amdil_sin">,
          UnaryIntFloat;
  def int_AMDIL_sin_vec : GCCBuiltin<"__amdil_sin_vec">,
          UnaryIntFloat;
  def int_AMDIL_pow : GCCBuiltin<"__amdil_pow">, BinaryIntFloat;
  def int_AMDIL_div : GCCBuiltin<"__amdil_div">, BinaryIntFloat;
  def int_AMDIL_udiv : GCCBuiltin<"__amdil_udiv">, BinaryIntInt;
  def int_AMDIL_sqrt: GCCBuiltin<"__amdil_sqrt">,
          UnaryIntFloat;
  def int_AMDIL_sqrt_vec: GCCBuiltin<"__amdil_sqrt_vec">,
          UnaryIntFloat;
  def int_AMDIL_exp : GCCBuiltin<"__amdil_exp">,
          UnaryIntFloat;
  def int_AMDIL_exp_vec : GCCBuiltin<"__amdil_exp_vec">,
          UnaryIntFloat;
  def int_AMDIL_exn : GCCBuiltin<"__amdil_exn">,
          UnaryIntFloat;
  def int_AMDIL_log_vec : GCCBuiltin<"__amdil_log_vec">,
          UnaryIntFloat;
  def int_AMDIL_ln : GCCBuiltin<"__amdil_ln">,
          UnaryIntFloat;
  def int_AMDIL_sign: GCCBuiltin<"__amdil_sign">,
          UnaryIntFloat;
  def int_AMDIL_fma: GCCBuiltin<"__amdil_fma">,
          TernaryIntFloat;
  def int_AMDIL_rsq : GCCBuiltin<"__amdil_rsq">,
          UnaryIntFloat;
  def int_AMDIL_rsq_vec : GCCBuiltin<"__amdil_rsq_vec">,
          UnaryIntFloat;
  def int_AMDIL_length : GCCBuiltin<"__amdil_length">,
          UnaryIntFloat;
  def int_AMDIL_lerp : GCCBuiltin<"__amdil_lerp">,
          TernaryIntFloat;
  def int_AMDIL_media_sad4 : GCCBuiltin<"__amdil_sad4">,
      Intrinsic<[llvm_i32_ty], [llvm_v4i32_ty,
           llvm_v4i32_ty, llvm_i32_ty], []>;

  def int_AMDIL_frexp_f64 : GCCBuiltin<"__amdil_frexp">,
        Intrinsic<[llvm_v2i64_ty], [llvm_double_ty], []>;
 def int_AMDIL_ldexp : GCCBuiltin<"__amdil_ldexp">,
    Intrinsic<[llvm_anyfloat_ty], [llvm_anyfloat_ty, llvm_anyint_ty], []>;
  def int_AMDIL_drcp : GCCBuiltin<"__amdil_rcp">,
      Intrinsic<[llvm_double_ty], [llvm_double_ty], []>;
  def int_AMDIL_convert_f16_f32 : GCCBuiltin<"__amdil_half_to_float">,
      ConvertIntITOF;
  def int_AMDIL_convert_f32_f16 : GCCBuiltin<"__amdil_float_to_half">,
      ConvertIntFTOI;
  def int_AMDIL_convert_f32_i32_rpi : GCCBuiltin<"__amdil_float_to_int_rpi">,
      ConvertIntFTOI;
  def int_AMDIL_convert_f32_i32_flr : GCCBuiltin<"__amdil_float_to_int_flr">,
      ConvertIntFTOI;
  def int_AMDIL_convert_f32_f16_near : GCCBuiltin<"__amdil_float_to_half_near">,
      ConvertIntFTOI;
  def int_AMDIL_convert_f32_f16_neg_inf : GCCBuiltin<"__amdil_float_to_half_neg_inf">,
      ConvertIntFTOI;
  def int_AMDIL_convert_f32_f16_plus_inf : GCCBuiltin<"__amdil_float_to_half_plus_inf">,
      ConvertIntFTOI;
 def int_AMDIL_media_convert_f2v4u8 : GCCBuiltin<"__amdil_f_2_u4">,
      Intrinsic<[llvm_i32_ty], [llvm_v4f32_ty], []>;
  def int_AMDIL_media_unpack_byte_0 : GCCBuiltin<"__amdil_unpack_0">,
      ConvertIntITOF;
  def int_AMDIL_media_unpack_byte_1 : GCCBuiltin<"__amdil_unpack_1">,
      ConvertIntITOF;
  def int_AMDIL_media_unpack_byte_2 : GCCBuiltin<"__amdil_unpack_2">,
      ConvertIntITOF;
  def int_AMDIL_media_unpack_byte_3 : GCCBuiltin<"__amdil_unpack_3">,
      ConvertIntITOF;
  def int_AMDIL_dp2_add : GCCBuiltin<"__amdil_dp2_add">,
        Intrinsic<[llvm_float_ty], [llvm_v2f32_ty,
          llvm_v2f32_ty, llvm_float_ty], []>;
  def int_AMDIL_dp2 : GCCBuiltin<"__amdil_dp2">,
        Intrinsic<[llvm_float_ty], [llvm_v2f32_ty,
          llvm_v2f32_ty], []>;
  def int_AMDIL_dp3 : GCCBuiltin<"__amdil_dp3">,
        Intrinsic<[llvm_float_ty], [llvm_v4f32_ty,
          llvm_v4f32_ty], []>;
  def int_AMDIL_dp4 : GCCBuiltin<"__amdil_dp4">,
        Intrinsic<[llvm_float_ty], [llvm_v4f32_ty,
          llvm_v4f32_ty], []>;
}
