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

let TargetPrefix = "AMDIL", isTarget = 1 in {
//------------- Synchronization Functions - OpenCL 6.11.9 --------------------//
  def int_AMDIL_fence   : GCCBuiltin<"mem_fence">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_global   : GCCBuiltin<"mem_fence_global">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_local   : GCCBuiltin<"mem_fence_local">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_region   : GCCBuiltin<"mem_fence_region">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_read_only   : GCCBuiltin<"read_mem_fence">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_read_only_global   : GCCBuiltin<"read_mem_fence_global">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_read_only_local   : GCCBuiltin<"read_mem_fence_local">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_read_only_region : GCCBuiltin<"read_mem_fence_region">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_write_only   : GCCBuiltin<"write_mem_fence">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_write_only_global   : GCCBuiltin<"write_mem_fence_global">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_write_only_local   : GCCBuiltin<"write_mem_fence_local">,
        UnaryIntNoRetInt;
  def int_AMDIL_fence_write_only_region : GCCBuiltin<"write_mem_fence_region">,
        UnaryIntNoRetInt;

  def int_AMDIL_early_exit : GCCBuiltin<"__amdil_early_exit">,
        UnaryIntNoRetInt;

  def int_AMDIL_cmov_logical  : GCCBuiltin<"__amdil_cmov_logical">,
          TernaryIntInt;
  def int_AMDIL_fabs : GCCBuiltin<"__amdil_fabs">, UnaryIntFloat;
  def int_AMDIL_abs : GCCBuiltin<"__amdil_abs">, UnaryIntInt;

  def int_AMDIL_bit_extract_i32 : GCCBuiltin<"__amdil_ibit_extract">,
          TernaryIntInt;
  def int_AMDIL_bit_extract_u32 : GCCBuiltin<"__amdil_ubit_extract">,
          TernaryIntInt;
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
  def int_AMDIL_mad_i32 : GCCBuiltin<"__amdil_imad">,
          TernaryIntInt;
  def int_AMDIL_mad_u32 : GCCBuiltin<"__amdil_umad">,
          TernaryIntInt;
  def int_AMDIL_mad     : GCCBuiltin<"__amdil_mad">,
          TernaryIntFloat;
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
  def int_AMDIL_mad24_i32 : GCCBuiltin<"__amdil_imad24">,
          TernaryIntInt;
  def int_AMDIL_mad24_u32 : GCCBuiltin<"__amdil_umad24">,
          TernaryIntInt;
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
  def int_AMDIL_round_posinf : GCCBuiltin<"__amdil_round_posinf">,
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
  def int_AMDIL_log : GCCBuiltin<"__amdil_log">,
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
//===---------------------- Image functions begin ------------------------===//
  def int_AMDIL_image1d_write : GCCBuiltin<"__amdil_image1d_write">,
      Intrinsic<[], [llvm_ptr_ty, llvm_v2i32_ty, llvm_v4i32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image1d_read_norm  : GCCBuiltin<"__amdil_image1d_read_norm">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_v4f32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image1d_read_unnorm  : GCCBuiltin<"__amdil_image1d_read_unnorm">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_v4f32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image1d_info0 : GCCBuiltin<"__amdil_image1d_info0">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], []>;

  def int_AMDIL_image1d_info1 : GCCBuiltin<"__amdil_image1d_info1">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], []>;

 def int_AMDIL_image1d_array_write : GCCBuiltin<"__amdil_image1d_array_write">,
      Intrinsic<[], [llvm_ptr_ty, llvm_v2i32_ty, llvm_v4i32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image1d_array_read_norm  : GCCBuiltin<"__amdil_image1d_array_read_norm">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_v4f32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image1d_array_read_unnorm  : GCCBuiltin<"__amdil_image1d_array_read_unnorm">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_v4f32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image1d_array_info0 : GCCBuiltin<"__amdil_image1d_array_info0">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], []>;

  def int_AMDIL_image1d_array_info1 : GCCBuiltin<"__amdil_image1d_array_info1">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], []>;

 def int_AMDIL_image2d_write : GCCBuiltin<"__amdil_image2d_write">,
      Intrinsic<[], [llvm_ptr_ty, llvm_v2i32_ty, llvm_v4i32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image2d_read_norm  : GCCBuiltin<"__amdil_image2d_read_norm">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_v4f32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image2d_read_unnorm  : GCCBuiltin<"__amdil_image2d_read_unnorm">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_v4f32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image2d_info0 : GCCBuiltin<"__amdil_image2d_info0">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], []>;

  def int_AMDIL_image2d_info1 : GCCBuiltin<"__amdil_image2d_info1">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], []>;

 def int_AMDIL_image2d_array_write : GCCBuiltin<"__amdil_image2d_array_write">,
      Intrinsic<[], [llvm_ptr_ty, llvm_v2i32_ty, llvm_v4i32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image2d_array_read_norm  : GCCBuiltin<"__amdil_image2d_array_read_norm">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_v4f32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image2d_array_read_unnorm  : GCCBuiltin<"__amdil_image2d_array_read_unnorm">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_v4f32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image2d_array_info0 : GCCBuiltin<"__amdil_image2d_array_info0">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], []>;

  def int_AMDIL_image2d_array_info1 : GCCBuiltin<"__amdil_image2d_array_info1">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], []>;

  def int_AMDIL_image3d_write : GCCBuiltin<"__amdil_image3d_write">,
         Intrinsic<[], [llvm_ptr_ty, llvm_v4i32_ty, llvm_v4i32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image3d_read_norm  : GCCBuiltin<"__amdil_image3d_read_norm">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_v4f32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image3d_read_unnorm  : GCCBuiltin<"__amdil_image3d_read_unnorm">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty, llvm_i32_ty, llvm_v4f32_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_image3d_info0 : GCCBuiltin<"__amdil_image3d_info0">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], []>;

  def int_AMDIL_image3d_info1 : GCCBuiltin<"__amdil_image3d_info1">,
      Intrinsic<[llvm_v4i32_ty], [llvm_ptr_ty], []>;

//===---------------------- Image functions end --------------------------===//

  def int_AMDIL_append_alloc_i32 : GCCBuiltin<"__amdil_append_alloc">,
      Intrinsic<[llvm_i32_ty], [llvm_ptr_ty], [IntrReadWriteArgMem]>;
  def int_AMDIL_append_consume_i32 : GCCBuiltin<"__amdil_append_consume">,
      Intrinsic<[llvm_i32_ty], [llvm_ptr_ty], [IntrReadWriteArgMem]>;
  def int_AMDIL_append_alloc_i32_noret : GCCBuiltin<"__amdil_append_alloc_noret">,
      Intrinsic<[llvm_i32_ty], [llvm_ptr_ty], [IntrReadWriteArgMem]>;
  def int_AMDIL_append_consume_i32_noret : GCCBuiltin<"__amdil_append_consume_noret">,
      Intrinsic<[llvm_i32_ty], [llvm_ptr_ty], [IntrReadWriteArgMem]>;

  def int_AMDIL_get_global_id : GCCBuiltin<"__amdil_get_global_id_int">,
      Intrinsic<[llvm_v4i32_ty], [], []>;
  def int_AMDIL_get_local_id : GCCBuiltin<"__amdil_get_local_id_int">,
      Intrinsic<[llvm_v4i32_ty], [], []>;
  def int_AMDIL_get_group_id : GCCBuiltin<"__amdil_get_group_id_int">,
      Intrinsic<[llvm_v4i32_ty], [], []>;
  def int_AMDIL_get_num_groups : GCCBuiltin<"__amdil_get_num_groups_int">,
      Intrinsic<[llvm_v4i32_ty], [], []>;
  def int_AMDIL_get_local_size : GCCBuiltin<"__amdil_get_local_size_int">,
      Intrinsic<[llvm_v4i32_ty], [], []>;
  def int_AMDIL_get_global_size : GCCBuiltin<"__amdil_get_global_size_int">,
      Intrinsic<[llvm_v4i32_ty], [], []>;
  def int_AMDIL_get_global_offset : GCCBuiltin<"__amdil_get_global_offset_int">,
      Intrinsic<[llvm_v4i32_ty], [], []>;
  def int_AMDIL_get_work_dim : GCCBuiltin<"get_work_dim">,
      Intrinsic<[llvm_i32_ty], [], []>;
  def int_AMDIL_get_printf_offset : GCCBuiltin<"__amdil_get_printf_offset">,
      Intrinsic<[llvm_i32_ty], []>;
  def int_AMDIL_get_printf_size : GCCBuiltin<"__amdil_get_printf_size">,
      Intrinsic<[llvm_i32_ty], []>;

/// Intrinsics for atomic instructions with no return value
/// Signed 32 bit integer atomics for global address space
def int_AMDIL_atomic_add_gi32_noret : GCCBuiltin<"__atomic_add_gi32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_sub_gi32_noret : GCCBuiltin<"__atomic_sub_gi32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_rsub_gi32_noret : GCCBuiltin<"__atomic_rsub_gi32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_xchg_gi32_noret : GCCBuiltin<"__atomic_xchg_gi32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_inc_gi32_noret : GCCBuiltin<"__atomic_inc_gi32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_dec_gi32_noret : GCCBuiltin<"__atomic_dec_gi32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_cmpxchg_gi32_noret : GCCBuiltin<"__atomic_cmpxchg_gi32_noret">,
    TernaryAtomicIntNoRet;
def int_AMDIL_atomic_min_gi32_noret : GCCBuiltin<"__atomic_min_gi32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_max_gi32_noret : GCCBuiltin<"__atomic_max_gi32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_and_gi32_noret : GCCBuiltin<"__atomic_and_gi32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_or_gi32_noret : GCCBuiltin<"__atomic_or_gi32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_xor_gi32_noret : GCCBuiltin<"__atomic_xor_gi32_noret">,
    BinaryAtomicIntNoRet;



/// Unsigned 32 bit integer atomics for global address space
def int_AMDIL_atomic_add_gu32_noret : GCCBuiltin<"__atomic_add_gu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_sub_gu32_noret : GCCBuiltin<"__atomic_sub_gu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_rsub_gu32_noret : GCCBuiltin<"__atomic_rsub_gu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_xchg_gu32_noret : GCCBuiltin<"__atomic_xchg_gu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_inc_gu32_noret : GCCBuiltin<"__atomic_inc_gu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_dec_gu32_noret : GCCBuiltin<"__atomic_dec_gu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_cmpxchg_gu32_noret : GCCBuiltin<"__atomic_cmpxchg_gu32_noret">,
    TernaryAtomicIntNoRet;
def int_AMDIL_atomic_min_gu32_noret : GCCBuiltin<"__atomic_min_gu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_max_gu32_noret : GCCBuiltin<"__atomic_max_gu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_and_gu32_noret : GCCBuiltin<"__atomic_and_gu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_or_gu32_noret : GCCBuiltin<"__atomic_or_gu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_xor_gu32_noret : GCCBuiltin<"__atomic_xor_gu32_noret">,
    BinaryAtomicIntNoRet;


/// Intrinsics for atomic instructions with a return value
/// Signed 32 bit integer atomics for global address space
def int_AMDIL_atomic_add_gi32 : GCCBuiltin<"__atomic_add_gi32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_sub_gi32 : GCCBuiltin<"__atomic_sub_gi32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_rsub_gi32 : GCCBuiltin<"__atomic_rsub_gi32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xchg_gi32 : GCCBuiltin<"__atomic_xchg_gi32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_inc_gi32 : GCCBuiltin<"__atomic_inc_gi32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_dec_gi32 : GCCBuiltin<"__atomic_dec_gi32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_cmpxchg_gi32 : GCCBuiltin<"__atomic_cmpxchg_gi32">,
    TernaryAtomicInt;
def int_AMDIL_atomic_min_gi32 : GCCBuiltin<"__atomic_min_gi32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_max_gi32 : GCCBuiltin<"__atomic_max_gi32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_and_gi32 : GCCBuiltin<"__atomic_and_gi32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_or_gi32 : GCCBuiltin<"__atomic_or_gi32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xor_gi32 : GCCBuiltin<"__atomic_xor_gi32">,
    BinaryAtomicInt;

/// 32 bit float atomics required by OpenCL
def int_AMDIL_atomic_xchg_gf32 : GCCBuiltin<"__atomic_xchg_gf32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xchg_gf32_noret : GCCBuiltin<"__atomic_xchg_gf32_noret">,
    BinaryAtomicIntNoRet;

/// Unsigned 32 bit integer atomics for global address space
def int_AMDIL_atomic_add_gu32 : GCCBuiltin<"__atomic_add_gu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_sub_gu32 : GCCBuiltin<"__atomic_sub_gu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_rsub_gu32 : GCCBuiltin<"__atomic_rsub_gu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xchg_gu32 : GCCBuiltin<"__atomic_xchg_gu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_inc_gu32 : GCCBuiltin<"__atomic_inc_gu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_dec_gu32 : GCCBuiltin<"__atomic_dec_gu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_cmpxchg_gu32 : GCCBuiltin<"__atomic_cmpxchg_gu32">,
    TernaryAtomicInt;
def int_AMDIL_atomic_min_gu32 : GCCBuiltin<"__atomic_min_gu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_max_gu32 : GCCBuiltin<"__atomic_max_gu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_and_gu32 : GCCBuiltin<"__atomic_and_gu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_or_gu32 : GCCBuiltin<"__atomic_or_gu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xor_gu32 : GCCBuiltin<"__atomic_xor_gu32">,
    BinaryAtomicInt;


/// Intrinsics for atomic instructions with no return value
/// Signed 32 bit integer atomics for local address space
def int_AMDIL_atomic_add_li32_noret : GCCBuiltin<"__atomic_add_li32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_sub_li32_noret : GCCBuiltin<"__atomic_sub_li32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_rsub_li32_noret : GCCBuiltin<"__atomic_rsub_li32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_xchg_li32_noret : GCCBuiltin<"__atomic_xchg_li32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_inc_li32_noret : GCCBuiltin<"__atomic_inc_li32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_dec_li32_noret : GCCBuiltin<"__atomic_dec_li32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_cmpxchg_li32_noret : GCCBuiltin<"__atomic_cmpxchg_li32_noret">,
    TernaryAtomicIntNoRet;
def int_AMDIL_atomic_min_li32_noret : GCCBuiltin<"__atomic_min_li32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_max_li32_noret : GCCBuiltin<"__atomic_max_li32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_and_li32_noret : GCCBuiltin<"__atomic_and_li32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_or_li32_noret : GCCBuiltin<"__atomic_or_li32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_mskor_li32_noret : GCCBuiltin<"__atomic_mskor_li32_noret">,
    TernaryAtomicIntNoRet;
def int_AMDIL_atomic_xor_li32_noret : GCCBuiltin<"__atomic_xor_li32_noret">,
    BinaryAtomicIntNoRet;

/// Signed 32 bit integer atomics for region address space
def int_AMDIL_atomic_add_ri32_noret : GCCBuiltin<"__atomic_add_ri32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_sub_ri32_noret : GCCBuiltin<"__atomic_sub_ri32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_rsub_ri32_noret : GCCBuiltin<"__atomic_rsub_ri32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_xchg_ri32_noret : GCCBuiltin<"__atomic_xchg_ri32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_inc_ri32_noret : GCCBuiltin<"__atomic_inc_ri32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_dec_ri32_noret : GCCBuiltin<"__atomic_dec_ri32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_cmpxchg_ri32_noret : GCCBuiltin<"__atomic_cmpxchg_ri32_noret">,
    TernaryAtomicIntNoRet;
def int_AMDIL_atomic_min_ri32_noret : GCCBuiltin<"__atomic_min_ri32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_max_ri32_noret : GCCBuiltin<"__atomic_max_ri32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_and_ri32_noret : GCCBuiltin<"__atomic_and_ri32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_or_ri32_noret : GCCBuiltin<"__atomic_or_ri32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_mskor_ri32_noret : GCCBuiltin<"__atomic_mskor_ri32_noret">,
    TernaryAtomicIntNoRet;
def int_AMDIL_atomic_xor_ri32_noret : GCCBuiltin<"__atomic_xor_ri32_noret">,
    BinaryAtomicIntNoRet;



/// Unsigned 32 bit integer atomics for local address space
def int_AMDIL_atomic_add_lu32_noret : GCCBuiltin<"__atomic_add_lu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_sub_lu32_noret : GCCBuiltin<"__atomic_sub_lu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_rsub_lu32_noret : GCCBuiltin<"__atomic_rsub_lu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_xchg_lu32_noret : GCCBuiltin<"__atomic_xchg_lu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_inc_lu32_noret : GCCBuiltin<"__atomic_inc_lu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_dec_lu32_noret : GCCBuiltin<"__atomic_dec_lu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_cmpxchg_lu32_noret : GCCBuiltin<"__atomic_cmpxchg_lu32_noret">,
    TernaryAtomicIntNoRet;
def int_AMDIL_atomic_min_lu32_noret : GCCBuiltin<"__atomic_min_lu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_max_lu32_noret : GCCBuiltin<"__atomic_max_lu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_and_lu32_noret : GCCBuiltin<"__atomic_and_lu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_or_lu32_noret : GCCBuiltin<"__atomic_or_lu32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_mskor_lu32_noret : GCCBuiltin<"__atomic_mskor_lu32_noret">,
    TernaryAtomicIntNoRet;
def int_AMDIL_atomic_xor_lu32_noret : GCCBuiltin<"__atomic_xor_lu32_noret">,
    BinaryAtomicIntNoRet;

/// Unsigned 32 bit integer atomics for region address space
def int_AMDIL_atomic_add_ru32_noret : GCCBuiltin<"__atomic_add_ru32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_sub_ru32_noret : GCCBuiltin<"__atomic_sub_ru32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_rsub_ru32_noret : GCCBuiltin<"__atomic_rsub_ru32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_xchg_ru32_noret : GCCBuiltin<"__atomic_xchg_ru32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_inc_ru32_noret : GCCBuiltin<"__atomic_inc_ru32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_dec_ru32_noret : GCCBuiltin<"__atomic_dec_ru32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_cmpxchg_ru32_noret : GCCBuiltin<"__atomic_cmpxchg_ru32_noret">,
    TernaryAtomicIntNoRet;
def int_AMDIL_atomic_min_ru32_noret : GCCBuiltin<"__atomic_min_ru32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_max_ru32_noret : GCCBuiltin<"__atomic_max_ru32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_and_ru32_noret : GCCBuiltin<"__atomic_and_ru32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_or_ru32_noret : GCCBuiltin<"__atomic_or_ru32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_mskor_ru32_noret : GCCBuiltin<"__atomic_mskor_ru32_noret">,
    TernaryAtomicIntNoRet;
def int_AMDIL_atomic_xor_ru32_noret : GCCBuiltin<"__atomic_xor_ru32_noret">,
    BinaryAtomicIntNoRet;

def int_AMDIL_get_cycle_count : GCCBuiltin<"__amdil_get_cycle_count">,
    VoidIntLong;

def int_AMDIL_compute_unit_id : GCCBuiltin<"__amdil_compute_unit_id">,
    VoidIntInt;

def int_AMDIL_wavefront_id : GCCBuiltin<"__amdil_wavefront_id">,
    VoidIntInt;


/// Intrinsics for atomic instructions with a return value
/// Signed 32 bit integer atomics for local address space
def int_AMDIL_atomic_add_li32 : GCCBuiltin<"__atomic_add_li32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_sub_li32 : GCCBuiltin<"__atomic_sub_li32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_rsub_li32 : GCCBuiltin<"__atomic_rsub_li32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xchg_li32 : GCCBuiltin<"__atomic_xchg_li32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_inc_li32 : GCCBuiltin<"__atomic_inc_li32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_dec_li32 : GCCBuiltin<"__atomic_dec_li32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_cmpxchg_li32 : GCCBuiltin<"__atomic_cmpxchg_li32">,
    TernaryAtomicInt;
def int_AMDIL_atomic_min_li32 : GCCBuiltin<"__atomic_min_li32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_max_li32 : GCCBuiltin<"__atomic_max_li32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_and_li32 : GCCBuiltin<"__atomic_and_li32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_or_li32 : GCCBuiltin<"__atomic_or_li32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_mskor_li32 : GCCBuiltin<"__atomic_mskor_li32">,
    TernaryAtomicInt;
def int_AMDIL_atomic_xor_li32 : GCCBuiltin<"__atomic_xor_li32">,
    BinaryAtomicInt;

/// Signed 32 bit integer atomics for region address space
def int_AMDIL_atomic_add_ri32 : GCCBuiltin<"__atomic_add_ri32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_sub_ri32 : GCCBuiltin<"__atomic_sub_ri32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_rsub_ri32 : GCCBuiltin<"__atomic_rsub_ri32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xchg_ri32 : GCCBuiltin<"__atomic_xchg_ri32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_inc_ri32 : GCCBuiltin<"__atomic_inc_ri32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_dec_ri32 : GCCBuiltin<"__atomic_dec_ri32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_cmpxchg_ri32 : GCCBuiltin<"__atomic_cmpxchg_ri32">,
    TernaryAtomicInt;
def int_AMDIL_atomic_min_ri32 : GCCBuiltin<"__atomic_min_ri32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_max_ri32 : GCCBuiltin<"__atomic_max_ri32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_and_ri32 : GCCBuiltin<"__atomic_and_ri32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_or_ri32 : GCCBuiltin<"__atomic_or_ri32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_mskor_ri32 : GCCBuiltin<"__atomic_mskor_ri32">,
    TernaryAtomicInt;
def int_AMDIL_atomic_xor_ri32 : GCCBuiltin<"__atomic_xor_ri32">,
    BinaryAtomicInt;

/// 32 bit float atomics required by OpenCL
def int_AMDIL_atomic_xchg_lf32 : GCCBuiltin<"__atomic_xchg_lf32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xchg_lf32_noret : GCCBuiltin<"__atomic_xchg_lf32_noret">,
    BinaryAtomicIntNoRet;
def int_AMDIL_atomic_xchg_rf32 : GCCBuiltin<"__atomic_xchg_rf32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xchg_rf32_noret : GCCBuiltin<"__atomic_xchg_rf32_noret">,
    BinaryAtomicIntNoRet;

/// Unsigned 32 bit integer atomics for local address space
def int_AMDIL_atomic_add_lu32 : GCCBuiltin<"__atomic_add_lu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_sub_lu32 : GCCBuiltin<"__atomic_sub_lu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_rsub_lu32 : GCCBuiltin<"__atomic_rsub_lu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xchg_lu32 : GCCBuiltin<"__atomic_xchg_lu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_inc_lu32 : GCCBuiltin<"__atomic_inc_lu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_dec_lu32 : GCCBuiltin<"__atomic_dec_lu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_cmpxchg_lu32 : GCCBuiltin<"__atomic_cmpxchg_lu32">,
    TernaryAtomicInt;
def int_AMDIL_atomic_min_lu32 : GCCBuiltin<"__atomic_min_lu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_max_lu32 : GCCBuiltin<"__atomic_max_lu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_and_lu32 : GCCBuiltin<"__atomic_and_lu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_or_lu32 : GCCBuiltin<"__atomic_or_lu32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_mskor_lu32 : GCCBuiltin<"__atomic_mskor_lu32">,
    TernaryAtomicInt;
def int_AMDIL_atomic_xor_lu32 : GCCBuiltin<"__atomic_xor_lu32">,
    BinaryAtomicInt;

/// Unsigned 32 bit integer atomics for region address space
def int_AMDIL_atomic_add_ru32 : GCCBuiltin<"__atomic_add_ru32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_sub_ru32 : GCCBuiltin<"__atomic_sub_ru32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_rsub_ru32 : GCCBuiltin<"__atomic_rsub_ru32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_xchg_ru32 : GCCBuiltin<"__atomic_xchg_ru32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_inc_ru32 : GCCBuiltin<"__atomic_inc_ru32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_dec_ru32 : GCCBuiltin<"__atomic_dec_ru32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_cmpxchg_ru32 : GCCBuiltin<"__atomic_cmpxchg_ru32">,
    TernaryAtomicInt;
def int_AMDIL_atomic_min_ru32 : GCCBuiltin<"__atomic_min_ru32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_max_ru32 : GCCBuiltin<"__atomic_max_ru32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_and_ru32 : GCCBuiltin<"__atomic_and_ru32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_or_ru32 : GCCBuiltin<"__atomic_or_ru32">,
    BinaryAtomicInt;
def int_AMDIL_atomic_mskor_ru32 : GCCBuiltin<"__atomic_mskor_ru32">,
    TernaryAtomicInt;
def int_AMDIL_atomic_xor_ru32 : GCCBuiltin<"__atomic_xor_ru32">,
    BinaryAtomicInt;

/// Semaphore signal/wait/init
def int_AMDIL_semaphore_init : GCCBuiltin<"__amdil_semaphore_init">,
    Intrinsic<[], [llvm_ptr_ty, llvm_i32_ty]>;
def int_AMDIL_semaphore_wait : GCCBuiltin<"__amdil_semaphore_wait">,
    Intrinsic<[], [llvm_ptr_ty]>;
def int_AMDIL_semaphore_signal : GCCBuiltin<"__amdil_semaphore_signal">,
    Intrinsic<[], [llvm_ptr_ty]>;
def int_AMDIL_semaphore_size   : GCCBuiltin<"__amdil_max_semaphore_size">,
    Intrinsic<[llvm_i32_ty], []>;
}
