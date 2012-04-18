//===- IntrinsicsHexagon.td - Defines Hexagon intrinsics ---*- tablegen -*-===//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the Hexagon-specific intrinsics.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Definitions for all Hexagon intrinsics.
//
// All Hexagon intrinsics start with "llvm.hexagon.".
let TargetPrefix = "hexagon" in {
  /// Hexagon_Intrinsic - Base class for all altivec intrinsics.
  class Hexagon_Intrinsic<string GCCIntSuffix, list<LLVMType> ret_types,
                              list<LLVMType> param_types,
                              list<IntrinsicProperty> properties>
    : GCCBuiltin<!strconcat("__builtin_", GCCIntSuffix)>,
      Intrinsic<ret_types, param_types, properties>;
}

//===----------------------------------------------------------------------===//
//
// DEF_FUNCTION_TYPE_1(QI_ftype_MEM,BT_BOOL,BT_PTR) ->
// Hexagon_qi_mem_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_mem_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_ptr_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(void_ftype_SI,BT_VOID,BT_INT) ->
// Hexagon_void_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_void_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_void_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(HI_ftype_SI,BT_I16,BT_INT) ->
// Hexagon_hi_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_hi_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i16_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(SI_ftype_SI,BT_INT,BT_INT) ->
// Hexagon_si_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(DI_ftype_SI,BT_LONGLONG,BT_INT) ->
// Hexagon_di_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(SI_ftype_DI,BT_INT,BT_LONGLONG) ->
// Hexagon_si_di_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_di_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(DI_ftype_DI,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_di_di_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_di_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(QI_ftype_QI,BT_BOOL,BT_BOOL) ->
// Hexagon_qi_qi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_qi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(QI_ftype_SI,BT_BOOL,BT_INT) ->
// Hexagon_qi_si_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_si_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(DI_ftype_QI,BT_LONGLONG,BT_BOOL) ->
// Hexagon_di_qi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_qi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_1(SI_ftype_QI,BT_INT,BT_BOOL) ->
// Hexagon_si_qi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_SISI,BT_BOOL,BT_INT,BT_INT) ->
// Hexagon_qi_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(void_ftype_SISI,BT_VOID,BT_INT,BT_INT) ->
// Hexagon_void_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_void_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_void_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_SISI,BT_INT,BT_INT,BT_INT) ->
// Hexagon_si_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(USI_ftype_SISI,BT_UINT,BT_INT,BT_INT) ->
// Hexagon_usi_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_usi_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_SISI,BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_di_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(UDI_ftype_SISI,BT_ULONGLONG,BT_INT,BT_INT) ->
// Hexagon_udi_sisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_udi_sisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_SIDI,BT_LONGLONG,BT_INT,BT_LONGLONG) ->
// Hexagon_di_sidi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_DISI,BT_LONGLONG,BT_LONGLONG,BT_INT) ->
// Hexagon_di_disi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_disi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_SIDI,BT_INT,BT_INT,BT_LONGLONG) ->
// Hexagon_si_sidi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_DIDI,BT_INT,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_si_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(DI_ftype_DIDI,BT_LONGLONG,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_di_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(UDI_ftype_DIDI,BT_ULONGLONG,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_udi_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_udi_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_DISI,BT_INT,BT_LONGLONG,BT_INT) ->
// Hexagon_si_disi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_disi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_DIDI,BT_BOOL,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_qi_didi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_didi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i64_ty, llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_QIQI,BT_BOOL,BT_BOOL,BT_BOOL) ->
// Hexagon_qi_qiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_qiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(QI_ftype_QIQIQI,BT_BOOL,BT_BOOL,BT_BOOL) ->
// Hexagon_qi_qiqiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_qi_qiqiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i1_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_QIQI,BT_INT,BT_BOOL,BT_BOOL) ->
// Hexagon_si_qiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_2(SI_ftype_QISI,BT_INT,BT_BOOL,BT_INT) ->
// Hexagon_si_qisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i1_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(void_ftype_SISISI,BT_VOID,BT_INT,BT_INT,BT_INT) ->
// Hexagon_void_sisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_void_sisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_void_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SISISI,BT_INT,BT_INT,BT_INT,BT_INT) ->
// Hexagon_si_sisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_SISISI,BT_LONGLONG,BT_INT,BT_INT,BT_INT) ->
// Hexagon_di_sisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_sisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_DISISI,BT_INT,BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_si_disisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_disisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i64_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DISISI,BT_LONGLONG,BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_di_disisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_disisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SIDISI,BT_INT,BT_INT,BT_LONGLONG,BT_INT) ->
// Hexagon_si_sidisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sidisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DIDISI,BT_LONGLONG,BT_LONGLONG,
//                     BT_LONGLONG,BT_INT) ->
// Hexagon_di_didisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SIDIDI,BT_INT,BT_INT,BT_LONGLONG,BT_LONGLONG) ->
// Hexagon_si_sididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sididi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i64_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DIDIDI,BT_LONGLONG,BT_LONGLONG,BT_LONGLONG,
//                     BT_LONGLONG) ->
// Hexagon_di_dididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_dididi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_SISIDI,BT_INT,BT_INT,BT_INT,BT_LONGLONG) ->
// Hexagon_si_sisidi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisidi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(SI_ftype_QISISI,BT_INT,BT_BOOL,BT_INT,BT_INT) ->
// Hexagon_si_qisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_qisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_QISISI,BT_LONGLONG,BT_BOOL,BT_INT,BT_INT) ->
// Hexagon_di_qisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_qisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i1_ty, llvm_i32_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_QIDIDI,BT_LONGLONG,BT_BOOL,BT_LONGLONG,
//                     BT_LONGLONG) ->
// Hexagon_di_qididi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_qididi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i32_ty, llvm_i64_ty,
                           llvm_i64_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_3(DI_ftype_DIDIQI,BT_LONGLONG,BT_LONGLONG,BT_LONGLONG,
//                     BT_BOOL) ->
// Hexagon_di_didiqi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didiqi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_4(SI_ftype_SISISISI,BT_INT,BT_INT,BT_INT,BT_INT,BT_INT) ->
// Hexagon_si_sisisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_si_sisisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;
//
// DEF_FUNCTION_TYPE_4(DI_ftype_DIDISISI,BT_LONGLONG,BT_LONGLONG,
//                     BT_LONGLONG,BT_INT,BT_INT) ->
// Hexagon_di_didisisi_Intrinsic<string GCCIntSuffix>
//
class Hexagon_di_didisisi_Intrinsic<string GCCIntSuffix>
  : Hexagon_Intrinsic<GCCIntSuffix,
                          [llvm_i64_ty], [llvm_i64_ty, llvm_i64_ty,
                           llvm_i32_ty, llvm_i32_ty],
                          [IntrNoMem]>;

//
// BUILTIN_INFO(HEXAGON.C2_cmpeq,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpeq : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.cmpeq">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgt,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgt : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.cmpgt">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtu,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgtu : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.cmpgtu">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpeqp,QI_ftype_DIDI,2)
//
def int_hexagon_C2_cmpeqp : Hexagon_qi_didi_Intrinsic<"HEXAGON.C2.cmpeqp">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtp,QI_ftype_DIDI,2)
//
def int_hexagon_C2_cmpgtp : Hexagon_qi_didi_Intrinsic<"HEXAGON.C2.cmpgtp">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtup,QI_ftype_DIDI,2)
//
def int_hexagon_C2_cmpgtup : Hexagon_qi_didi_Intrinsic<"HEXAGON.C2.cmpgtup">;
//
// BUILTIN_INFO(HEXAGON.C2_bitsset,QI_ftype_SISI,2)
//
def int_hexagon_C2_bitsset : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.bitsset">;
//
// BUILTIN_INFO(HEXAGON.C2_bitsclr,QI_ftype_SISI,2)
//
def int_hexagon_C2_bitsclr : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.bitsclr">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpeqi,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpeqi : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.cmpeqi">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgti,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgti : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.cmpgti">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgtui,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgtui : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.cmpgtui">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgei,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgei : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.cmpgei">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpgeui,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpgeui : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.cmpgeui">;
//
// BUILTIN_INFO(HEXAGON.C2_cmplt,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmplt : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.cmplt">;
//
// BUILTIN_INFO(HEXAGON.C2_cmpltu,QI_ftype_SISI,2)
//
def int_hexagon_C2_cmpltu : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.cmpltu">;
//
// BUILTIN_INFO(HEXAGON.C2_bitsclri,QI_ftype_SISI,2)
//
def int_hexagon_C2_bitsclri : Hexagon_qi_sisi_Intrinsic<"HEXAGON.C2.bitsclri">;
//
// BUILTIN_INFO(HEXAGON.C2_and,QI_ftype_QIQI,2)
//
def int_hexagon_C2_and : Hexagon_qi_qiqi_Intrinsic<"HEXAGON.C2.and">;
//
// BUILTIN_INFO(HEXAGON.C2_or,QI_ftype_QIQI,2)
//
def int_hexagon_C2_or : Hexagon_qi_qiqi_Intrinsic<"HEXAGON.C2.or">;
//
// BUILTIN_INFO(HEXAGON.C2_xor,QI_ftype_QIQI,2)
//
def int_hexagon_C2_xor : Hexagon_qi_qiqi_Intrinsic<"HEXAGON.C2.xor">;
//
// BUILTIN_INFO(HEXAGON.C2_andn,QI_ftype_QIQI,2)
//
def int_hexagon_C2_andn : Hexagon_qi_qiqi_Intrinsic<"HEXAGON.C2.andn">;
//
// BUILTIN_INFO(HEXAGON.C2_not,QI_ftype_QI,1)
//
def int_hexagon_C2_not : Hexagon_qi_qi_Intrinsic<"HEXAGON.C2.not">;
//
// BUILTIN_INFO(HEXAGON.C2_orn,QI_ftype_QIQI,2)
//
def int_hexagon_C2_orn : Hexagon_qi_qiqi_Intrinsic<"HEXAGON.C2.orn">;
//
// BUILTIN_INFO(HEXAGON.C2_pxfer_map,QI_ftype_QI,1)
//
def int_hexagon_C2_pxfer_map : Hexagon_qi_qi_Intrinsic<"HEXAGON.C2.pxfer.map">;
//
// BUILTIN_INFO(HEXAGON.C2_any8,QI_ftype_QI,1)
//
def int_hexagon_C2_any8 : Hexagon_qi_qi_Intrinsic<"HEXAGON.C2.any8">;
//
// BUILTIN_INFO(HEXAGON.C2_all8,QI_ftype_QI,1)
//
def int_hexagon_C2_all8 : Hexagon_qi_qi_Intrinsic<"HEXAGON.C2.all8">;
//
// BUILTIN_INFO(HEXAGON.C2_vitpack,SI_ftype_QIQI,2)
//
def int_hexagon_C2_vitpack : Hexagon_si_qiqi_Intrinsic<"HEXAGON.C2.vitpack">;
//
// BUILTIN_INFO(HEXAGON.C2_mux,SI_ftype_QISISI,3)
//
def int_hexagon_C2_mux : Hexagon_si_qisisi_Intrinsic<"HEXAGON.C2.mux">;
//
// BUILTIN_INFO(HEXAGON.C2_muxii,SI_ftype_QISISI,3)
//
def int_hexagon_C2_muxii : Hexagon_si_qisisi_Intrinsic<"HEXAGON.C2.muxii">;
//
// BUILTIN_INFO(HEXAGON.C2_muxir,SI_ftype_QISISI,3)
//
def int_hexagon_C2_muxir : Hexagon_si_qisisi_Intrinsic<"HEXAGON.C2.muxir">;
//
// BUILTIN_INFO(HEXAGON.C2_muxri,SI_ftype_QISISI,3)
//
def int_hexagon_C2_muxri : Hexagon_si_qisisi_Intrinsic<"HEXAGON.C2.muxri">;
//
// BUILTIN_INFO(HEXAGON.C2_vmux,DI_ftype_QIDIDI,3)
//
def int_hexagon_C2_vmux : Hexagon_di_qididi_Intrinsic<"HEXAGON.C2.vmux">;
//
// BUILTIN_INFO(HEXAGON.C2_mask,DI_ftype_QI,1)
//
def int_hexagon_C2_mask : Hexagon_di_qi_Intrinsic<"HEXAGON.C2.mask">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpbeq,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpbeq : Hexagon_qi_didi_Intrinsic<"HEXAGON.A2.vcmpbeq">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpbgtu,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpbgtu : Hexagon_qi_didi_Intrinsic<"HEXAGON.A2.vcmpbgtu">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpheq,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpheq : Hexagon_qi_didi_Intrinsic<"HEXAGON.A2.vcmpheq">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmphgt,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmphgt : Hexagon_qi_didi_Intrinsic<"HEXAGON.A2.vcmphgt">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmphgtu,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmphgtu : Hexagon_qi_didi_Intrinsic<"HEXAGON.A2.vcmphgtu">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpweq,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpweq : Hexagon_qi_didi_Intrinsic<"HEXAGON.A2.vcmpweq">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpwgt,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpwgt : Hexagon_qi_didi_Intrinsic<"HEXAGON.A2.vcmpwgt">;
//
// BUILTIN_INFO(HEXAGON.A2_vcmpwgtu,QI_ftype_DIDI,2)
//
def int_hexagon_A2_vcmpwgtu : Hexagon_qi_didi_Intrinsic<"HEXAGON.A2.vcmpwgtu">;
//
// BUILTIN_INFO(HEXAGON.C2_tfrpr,SI_ftype_QI,1)
//
def int_hexagon_C2_tfrpr : Hexagon_si_qi_Intrinsic<"HEXAGON.C2.tfrpr">;
//
// BUILTIN_INFO(HEXAGON.C2_tfrrp,QI_ftype_SI,1)
//
def int_hexagon_C2_tfrrp : Hexagon_qi_si_Intrinsic<"HEXAGON.C2.tfrrp">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.sat.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.sat.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.sat.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.sat.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.sat.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.sat.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.sat.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_acc_sat_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_acc_sat_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.acc.sat.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.sat.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.sat.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.sat.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.sat.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.sat.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.sat.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.sat.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_nac_sat_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpy_nac_sat_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpy.nac.sat.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.rnd.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.rnd.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.rnd.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.rnd.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.rnd.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.rnd.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.rnd.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_rnd_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_rnd_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.rnd.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.rnd.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.rnd.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hl_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hl_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.rnd.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_hl_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_hl_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.rnd.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_lh_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_lh_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.rnd.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_lh_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_lh_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.rnd.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_ll_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_ll_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.rnd.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_sat_rnd_ll_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_sat_rnd_ll_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.sat.rnd.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.acc.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.acc.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.acc.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.acc.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.acc.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.acc.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.acc.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_acc_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_acc_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.acc.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.nac.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.nac.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.nac.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.nac.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.nac.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.nac.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.nac.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_nac_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyd_nac_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyd.nac.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hl_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hl_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_hl_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_hl_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_lh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_lh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_lh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_lh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_ll_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_ll_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_ll_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_ll_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.rnd.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.rnd.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hl_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hl_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.rnd.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_hl_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_hl_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.rnd.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_lh_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_lh_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.rnd.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_lh_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_lh_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.rnd.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_ll_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_ll_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.rnd.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyd_rnd_ll_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_mpyd_rnd_ll_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.mpyd.rnd.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.acc.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.acc.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.acc.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.acc.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.acc.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.acc.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.acc.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_acc_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_acc_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.acc.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.nac.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.nac.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hl_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hl_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.nac.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_hl_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_hl_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.nac.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_lh_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_lh_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.nac.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_lh_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_lh_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.nac.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_ll_s0,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_ll_s0 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.nac.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_nac_ll_s1,SI_ftype_SISISI,3)
//
def int_hexagon_M2_mpyu_nac_ll_s1 :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.mpyu.nac.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hh_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hh_s0 :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.M2.mpyu.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hh_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hh_s1 :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.M2.mpyu.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hl_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hl_s0 :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.M2.mpyu.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_hl_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_hl_s1 :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.M2.mpyu.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_lh_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_lh_s0 :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.M2.mpyu.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_lh_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_lh_s1 :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.M2.mpyu.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_ll_s0,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_ll_s0 :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.M2.mpyu.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_ll_s1,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_ll_s1 :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.M2.mpyu.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.acc.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.acc.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.acc.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.acc.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.acc.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.acc.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.acc.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_acc_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_acc_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.acc.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.nac.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.nac.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hl_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hl_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.nac.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_hl_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_hl_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.nac.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_lh_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_lh_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.nac.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_lh_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_lh_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.nac.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_ll_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_ll_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.nac.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_nac_ll_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_mpyud_nac_ll_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.mpyud.nac.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hh_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hh_s0 :
Hexagon_udi_sisi_Intrinsic<"HEXAGON.M2.mpyud.hh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hh_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hh_s1 :
Hexagon_udi_sisi_Intrinsic<"HEXAGON.M2.mpyud.hh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hl_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hl_s0 :
Hexagon_udi_sisi_Intrinsic<"HEXAGON.M2.mpyud.hl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_hl_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_hl_s1 :
Hexagon_udi_sisi_Intrinsic<"HEXAGON.M2.mpyud.hl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_lh_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_lh_s0 :
Hexagon_udi_sisi_Intrinsic<"HEXAGON.M2.mpyud.lh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_lh_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_lh_s1 :
Hexagon_udi_sisi_Intrinsic<"HEXAGON.M2.mpyud.lh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_ll_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_ll_s0 :
Hexagon_udi_sisi_Intrinsic<"HEXAGON.M2.mpyud.ll.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyud_ll_s1,UDI_ftype_SISI,2)
//
def int_hexagon_M2_mpyud_ll_s1 :
Hexagon_udi_sisi_Intrinsic<"HEXAGON.M2.mpyud.ll.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mpysmi,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpysmi :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpysmi">;
//
// BUILTIN_INFO(HEXAGON.M2_macsip,SI_ftype_SISISI,3)
//
def int_hexagon_M2_macsip :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.macsip">;
//
// BUILTIN_INFO(HEXAGON.M2_macsin,SI_ftype_SISISI,3)
//
def int_hexagon_M2_macsin :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.macsin">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_dpmpyss_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.dpmpyss.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_acc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyss_acc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.dpmpyss.acc.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_nac_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyss_nac_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.dpmpyss.nac.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyuu_s0,UDI_ftype_SISI,2)
//
def int_hexagon_M2_dpmpyuu_s0 :
Hexagon_udi_sisi_Intrinsic<"HEXAGON.M2.dpmpyuu.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyuu_acc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyuu_acc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.dpmpyuu.acc.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyuu_nac_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_dpmpyuu_nac_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.dpmpyuu.nac.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpy_up,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpy_up :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpy.up">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyu_up,USI_ftype_SISI,2)
//
def int_hexagon_M2_mpyu_up :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.M2.mpyu.up">;
//
// BUILTIN_INFO(HEXAGON.M2_dpmpyss_rnd_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_dpmpyss_rnd_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.dpmpyss.rnd.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyi,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpyi :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpyi">;
//
// BUILTIN_INFO(HEXAGON.M2_mpyui,SI_ftype_SISI,2)
//
def int_hexagon_M2_mpyui :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.mpyui">;
//
// BUILTIN_INFO(HEXAGON.M2_maci,SI_ftype_SISISI,3)
//
def int_hexagon_M2_maci :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.maci">;
//
// BUILTIN_INFO(HEXAGON.M2_acci,SI_ftype_SISISI,3)
//
def int_hexagon_M2_acci :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.acci">;
//
// BUILTIN_INFO(HEXAGON.M2_accii,SI_ftype_SISISI,3)
//
def int_hexagon_M2_accii :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.accii">;
//
// BUILTIN_INFO(HEXAGON.M2_nacci,SI_ftype_SISISI,3)
//
def int_hexagon_M2_nacci :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.nacci">;
//
// BUILTIN_INFO(HEXAGON.M2_naccii,SI_ftype_SISISI,3)
//
def int_hexagon_M2_naccii :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.naccii">;
//
// BUILTIN_INFO(HEXAGON.M2_subacc,SI_ftype_SISISI,3)
//
def int_hexagon_M2_subacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.subacc">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.vmpy2s.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.vmpy2s.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2s_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2s_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.vmac2s.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2s_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2s_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.vmac2s.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s0pack,SI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s0pack :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.vmpy2s.s0pack">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2s_s1pack,SI_ftype_SISI,2)
//
def int_hexagon_M2_vmpy2s_s1pack :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.vmpy2s.s1pack">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2,DI_ftype_DISISI,3)
//
def int_hexagon_M2_vmac2 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.vmac2">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2es_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vmpy2es_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vmpy2es.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmpy2es_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vmpy2es_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vmpy2es.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2es_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vmac2es_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vmac2es.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2es_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vmac2es_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vmac2es.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vmac2es,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vmac2es :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vmac2es">;
//
// BUILTIN_INFO(HEXAGON.M2_vrmac_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrmac_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vrmac.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrmpy_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrmpy_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vrmpy.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpyrs_s0,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpyrs_s0 :
Hexagon_si_didi_Intrinsic<"HEXAGON.M2.vdmpyrs.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpyrs_s1,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpyrs_s1 :
Hexagon_si_didi_Intrinsic<"HEXAGON.M2.vdmpyrs.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmacs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vdmacs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vdmacs.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmacs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vdmacs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vdmacs.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpys_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpys_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vdmpys.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vdmpys_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vdmpys_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vdmpys.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrs_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrs_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.cmpyrs.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrs_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrs_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.cmpyrs.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrsc_s0,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrsc_s0 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.cmpyrsc.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyrsc_s1,SI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyrsc_s1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.cmpyrsc.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacs_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacs_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.cmacs.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacs_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacs_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.cmacs.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacsc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacsc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.cmacsc.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacsc_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacsc_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.cmacsc.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpys_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpys_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.cmpys.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpys_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpys_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.cmpys.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpysc_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpysc_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.cmpysc.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpysc_s1,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpysc_s1 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.cmpysc.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacs_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacs_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.cnacs.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacs_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacs_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.cnacs.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacsc_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacsc_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.cnacsc.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cnacsc_s1,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cnacsc_s1 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.cnacsc.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpys_s1,DI_ftype_DISI,2)
//
def int_hexagon_M2_vrcmpys_s1 :
Hexagon_di_disi_Intrinsic<"HEXAGON.M2.vrcmpys.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpys_acc_s1,DI_ftype_DIDISI,3)
//
def int_hexagon_M2_vrcmpys_acc_s1 :
Hexagon_di_didisi_Intrinsic<"HEXAGON.M2.vrcmpys.acc.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpys_s1rp,SI_ftype_DISI,2)
//
def int_hexagon_M2_vrcmpys_s1rp :
Hexagon_si_disi_Intrinsic<"HEXAGON.M2.vrcmpys.s1rp">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmacls.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmacls.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmachs.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmachs.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyl.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyl.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmacls.rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacls_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacls_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmacls.rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmachs.rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmachs_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmachs_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmachs.rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyl.rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyl_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyl_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyl.rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyh.rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyh_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyh_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyh.rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_hmmpyl_rs1,SI_ftype_SISI,2)
//
def int_hexagon_M2_hmmpyl_rs1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.hmmpyl.rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_hmmpyh_rs1,SI_ftype_SISI,2)
//
def int_hexagon_M2_hmmpyh_rs1 :
Hexagon_si_sisi_Intrinsic<"HEXAGON.M2.hmmpyh.rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmaculs.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmaculs.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmacuhs.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_s1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_s1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmacuhs.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyul.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyul.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyuh.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_s1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_s1 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyuh.s1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmaculs.rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmaculs_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmaculs_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmaculs.rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_rs0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_rs0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmacuhs.rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmacuhs_rs1,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_mmacuhs_rs1 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.mmacuhs.rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyul.rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyul_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyul_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyul.rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_rs0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_rs0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyuh.rs0">;
//
// BUILTIN_INFO(HEXAGON.M2_mmpyuh_rs1,DI_ftype_DIDI,2)
//
def int_hexagon_M2_mmpyuh_rs1 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.mmpyuh.rs1">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmaci_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmaci_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vrcmaci.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmacr_s0,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmacr_s0 :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vrcmacr.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmaci_s0c,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmaci_s0c :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vrcmaci.s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmacr_s0c,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vrcmacr_s0c :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vrcmacr.s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_cmaci_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmaci_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.cmaci.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmacr_s0,DI_ftype_DISISI,3)
//
def int_hexagon_M2_cmacr_s0 :
Hexagon_di_disisi_Intrinsic<"HEXAGON.M2.cmacr.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyi_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyi_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vrcmpyi.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyr_s0,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyr_s0 :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vrcmpyr.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyi_s0c,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyi_s0c :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vrcmpyi.s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_vrcmpyr_s0c,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vrcmpyr_s0c :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vrcmpyr.s0c">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyi_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyi_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.cmpyi.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_cmpyr_s0,DI_ftype_SISI,2)
//
def int_hexagon_M2_cmpyr_s0 :
Hexagon_di_sisi_Intrinsic<"HEXAGON.M2.cmpyr.s0">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s0_sat_i,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s0_sat_i :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vcmpy.s0.sat.i">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s0_sat_r,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s0_sat_r :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vcmpy.s0.sat.r">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s1_sat_i,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s1_sat_i :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vcmpy.s1.sat.i">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmpy_s1_sat_r,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vcmpy_s1_sat_r :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vcmpy.s1.sat.r">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmac_s0_sat_i,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vcmac_s0_sat_i :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vcmac.s0.sat.i">;
//
// BUILTIN_INFO(HEXAGON.M2_vcmac_s0_sat_r,DI_ftype_DIDIDI,3)
//
def int_hexagon_M2_vcmac_s0_sat_r :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M2.vcmac.s0.sat.r">;
//
// BUILTIN_INFO(HEXAGON.S2_vcrotate,DI_ftype_DISI,2)
//
def int_hexagon_S2_vcrotate :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.vcrotate">;
//
// BUILTIN_INFO(HEXAGON.A2_add,SI_ftype_SISI,2)
//
def int_hexagon_A2_add :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.add">;
//
// BUILTIN_INFO(HEXAGON.A2_sub,SI_ftype_SISI,2)
//
def int_hexagon_A2_sub :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.sub">;
//
// BUILTIN_INFO(HEXAGON.A2_addsat,SI_ftype_SISI,2)
//
def int_hexagon_A2_addsat :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addsat">;
//
// BUILTIN_INFO(HEXAGON.A2_subsat,SI_ftype_SISI,2)
//
def int_hexagon_A2_subsat :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subsat">;
//
// BUILTIN_INFO(HEXAGON.A2_addi,SI_ftype_SISI,2)
//
def int_hexagon_A2_addi :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addi">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.l16.ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.l16.hl">;
def int_hexagon_A2_addh_l16_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.l16.lh">;
def int_hexagon_A2_addh_l16_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.l16.hh">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.l16.sat.ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_l16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_l16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.l16.sat.hl">;
def int_hexagon_A2_addh_l16_sat_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.l16.sat.lh">;
def int_hexagon_A2_addh_l16_sat_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.l16.sat.hh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.l16.ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.l16.hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.l16.sat.ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_l16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_l16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.l16.sat.hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.h16.ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.h16.lh">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.h16.hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.h16.hh">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.h16.sat.ll">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.h16.sat.lh">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.h16.sat.hl">;
//
// BUILTIN_INFO(HEXAGON.A2_addh_h16_sat_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_addh_h16_sat_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.addh.h16.sat.hh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.h16.ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.h16.lh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.h16.hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.h16.hh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.h16.sat.ll">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.h16.sat.lh">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.h16.sat.hl">;
//
// BUILTIN_INFO(HEXAGON.A2_subh_h16_sat_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_subh_h16_sat_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subh.h16.sat.hh">;
//
// BUILTIN_INFO(HEXAGON.A2_aslh,SI_ftype_SI,1)
//
def int_hexagon_A2_aslh :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.aslh">;
//
// BUILTIN_INFO(HEXAGON.A2_asrh,SI_ftype_SI,1)
//
def int_hexagon_A2_asrh :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.asrh">;
//
// BUILTIN_INFO(HEXAGON.A2_addp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_addp :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.addp">;
//
// BUILTIN_INFO(HEXAGON.A2_addpsat,DI_ftype_DIDI,2)
//
def int_hexagon_A2_addpsat :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.addpsat">;
//
// BUILTIN_INFO(HEXAGON.A2_addsp,DI_ftype_SIDI,2)
//
def int_hexagon_A2_addsp :
Hexagon_di_sidi_Intrinsic<"HEXAGON.A2.addsp">;
//
// BUILTIN_INFO(HEXAGON.A2_subp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_subp :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.subp">;
//
// BUILTIN_INFO(HEXAGON.A2_neg,SI_ftype_SI,1)
//
def int_hexagon_A2_neg :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.neg">;
//
// BUILTIN_INFO(HEXAGON.A2_negsat,SI_ftype_SI,1)
//
def int_hexagon_A2_negsat :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.negsat">;
//
// BUILTIN_INFO(HEXAGON.A2_abs,SI_ftype_SI,1)
//
def int_hexagon_A2_abs :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.abs">;
//
// BUILTIN_INFO(HEXAGON.A2_abssat,SI_ftype_SI,1)
//
def int_hexagon_A2_abssat :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.abssat">;
//
// BUILTIN_INFO(HEXAGON.A2_vconj,DI_ftype_DI,1)
//
def int_hexagon_A2_vconj :
Hexagon_di_di_Intrinsic<"HEXAGON.A2.vconj">;
//
// BUILTIN_INFO(HEXAGON.A2_negp,DI_ftype_DI,1)
//
def int_hexagon_A2_negp :
Hexagon_di_di_Intrinsic<"HEXAGON.A2.negp">;
//
// BUILTIN_INFO(HEXAGON.A2_absp,DI_ftype_DI,1)
//
def int_hexagon_A2_absp :
Hexagon_di_di_Intrinsic<"HEXAGON.A2.absp">;
//
// BUILTIN_INFO(HEXAGON.A2_max,SI_ftype_SISI,2)
//
def int_hexagon_A2_max :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.max">;
//
// BUILTIN_INFO(HEXAGON.A2_maxu,USI_ftype_SISI,2)
//
def int_hexagon_A2_maxu :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.A2.maxu">;
//
// BUILTIN_INFO(HEXAGON.A2_min,SI_ftype_SISI,2)
//
def int_hexagon_A2_min :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.min">;
//
// BUILTIN_INFO(HEXAGON.A2_minu,USI_ftype_SISI,2)
//
def int_hexagon_A2_minu :
Hexagon_usi_sisi_Intrinsic<"HEXAGON.A2.minu">;
//
// BUILTIN_INFO(HEXAGON.A2_maxp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_maxp :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.maxp">;
//
// BUILTIN_INFO(HEXAGON.A2_maxup,UDI_ftype_DIDI,2)
//
def int_hexagon_A2_maxup :
Hexagon_udi_didi_Intrinsic<"HEXAGON.A2.maxup">;
//
// BUILTIN_INFO(HEXAGON.A2_minp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_minp :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.minp">;
//
// BUILTIN_INFO(HEXAGON.A2_minup,UDI_ftype_DIDI,2)
//
def int_hexagon_A2_minup :
Hexagon_udi_didi_Intrinsic<"HEXAGON.A2.minup">;
//
// BUILTIN_INFO(HEXAGON.A2_tfr,SI_ftype_SI,1)
//
def int_hexagon_A2_tfr :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.tfr">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrsi,SI_ftype_SI,1)
//
def int_hexagon_A2_tfrsi :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.tfrsi">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrp,DI_ftype_DI,1)
//
def int_hexagon_A2_tfrp :
Hexagon_di_di_Intrinsic<"HEXAGON.A2.tfrp">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrpi,DI_ftype_SI,1)
//
def int_hexagon_A2_tfrpi :
Hexagon_di_si_Intrinsic<"HEXAGON.A2.tfrpi">;
//
// BUILTIN_INFO(HEXAGON.A2_zxtb,SI_ftype_SI,1)
//
def int_hexagon_A2_zxtb :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.zxtb">;
//
// BUILTIN_INFO(HEXAGON.A2_sxtb,SI_ftype_SI,1)
//
def int_hexagon_A2_sxtb :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.sxtb">;
//
// BUILTIN_INFO(HEXAGON.A2_zxth,SI_ftype_SI,1)
//
def int_hexagon_A2_zxth :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.zxth">;
//
// BUILTIN_INFO(HEXAGON.A2_sxth,SI_ftype_SI,1)
//
def int_hexagon_A2_sxth :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.sxth">;
//
// BUILTIN_INFO(HEXAGON.A2_combinew,DI_ftype_SISI,2)
//
def int_hexagon_A2_combinew :
Hexagon_di_sisi_Intrinsic<"HEXAGON.A2.combinew">;
//
// BUILTIN_INFO(HEXAGON.A2_combineii,DI_ftype_SISI,2)
//
def int_hexagon_A2_combineii :
Hexagon_di_sisi_Intrinsic<"HEXAGON.A2.combineii">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_hh,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_hh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.combine.hh">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_hl,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_hl :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.combine.hl">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_lh,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_lh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.combine.lh">;
//
// BUILTIN_INFO(HEXAGON.A2_combine_ll,SI_ftype_SISI,2)
//
def int_hexagon_A2_combine_ll :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.combine.ll">;
//
// BUILTIN_INFO(HEXAGON.A2_tfril,SI_ftype_SISI,2)
//
def int_hexagon_A2_tfril :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.tfril">;
//
// BUILTIN_INFO(HEXAGON.A2_tfrih,SI_ftype_SISI,2)
//
def int_hexagon_A2_tfrih :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.tfrih">;
//
// BUILTIN_INFO(HEXAGON.A2_and,SI_ftype_SISI,2)
//
def int_hexagon_A2_and :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.and">;
//
// BUILTIN_INFO(HEXAGON.A2_or,SI_ftype_SISI,2)
//
def int_hexagon_A2_or :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.or">;
//
// BUILTIN_INFO(HEXAGON.A2_xor,SI_ftype_SISI,2)
//
def int_hexagon_A2_xor :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.xor">;
//
// BUILTIN_INFO(HEXAGON.A2_not,SI_ftype_SI,1)
//
def int_hexagon_A2_not :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.not">;
//
// BUILTIN_INFO(HEXAGON.M2_xor_xacc,SI_ftype_SISISI,3)
//
def int_hexagon_M2_xor_xacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M2.xor.xacc">;
//
// BUILTIN_INFO(HEXAGON.A2_subri,SI_ftype_SISI,2)
//
def int_hexagon_A2_subri :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.subri">;
//
// BUILTIN_INFO(HEXAGON.A2_andir,SI_ftype_SISI,2)
//
def int_hexagon_A2_andir :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.andir">;
//
// BUILTIN_INFO(HEXAGON.A2_orir,SI_ftype_SISI,2)
//
def int_hexagon_A2_orir :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.orir">;
//
// BUILTIN_INFO(HEXAGON.A2_andp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_andp :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.andp">;
//
// BUILTIN_INFO(HEXAGON.A2_orp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_orp :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.orp">;
//
// BUILTIN_INFO(HEXAGON.A2_xorp,DI_ftype_DIDI,2)
//
def int_hexagon_A2_xorp :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.xorp">;
//
// BUILTIN_INFO(HEXAGON.A2_notp,DI_ftype_DI,1)
//
def int_hexagon_A2_notp :
Hexagon_di_di_Intrinsic<"HEXAGON.A2.notp">;
//
// BUILTIN_INFO(HEXAGON.A2_sxtw,DI_ftype_SI,1)
//
def int_hexagon_A2_sxtw :
Hexagon_di_si_Intrinsic<"HEXAGON.A2.sxtw">;
//
// BUILTIN_INFO(HEXAGON.A2_sat,SI_ftype_DI,1)
//
def int_hexagon_A2_sat :
Hexagon_si_di_Intrinsic<"HEXAGON.A2.sat">;
//
// BUILTIN_INFO(HEXAGON.A2_sath,SI_ftype_SI,1)
//
def int_hexagon_A2_sath :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.sath">;
//
// BUILTIN_INFO(HEXAGON.A2_satuh,SI_ftype_SI,1)
//
def int_hexagon_A2_satuh :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.satuh">;
//
// BUILTIN_INFO(HEXAGON.A2_satub,SI_ftype_SI,1)
//
def int_hexagon_A2_satub :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.satub">;
//
// BUILTIN_INFO(HEXAGON.A2_satb,SI_ftype_SI,1)
//
def int_hexagon_A2_satb :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.satb">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddub :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vaddub">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddubs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddubs :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vaddubs">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddh  :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vaddh">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddhs :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vaddhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vadduhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vadduhs :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vadduhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddw :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vaddw">;
//
// BUILTIN_INFO(HEXAGON.A2_vaddws,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vaddws :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vaddws">;
//
// BUILTIN_INFO(HEXAGON.A2_svavgh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svavgh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.svavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_svavghs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svavghs :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.svavghs">;
//
// BUILTIN_INFO(HEXAGON.A2_svnavgh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svnavgh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.svnavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_svaddh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svaddh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.svaddh">;
//
// BUILTIN_INFO(HEXAGON.A2_svaddhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svaddhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.svaddhs">;
//
// BUILTIN_INFO(HEXAGON.A2_svadduhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svadduhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.svadduhs">;
//
// BUILTIN_INFO(HEXAGON.A2_svsubh,SI_ftype_SISI,2)
//
def int_hexagon_A2_svsubh :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.svsubh">;
//
// BUILTIN_INFO(HEXAGON.A2_svsubhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svsubhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.svsubhs">;
//
// BUILTIN_INFO(HEXAGON.A2_svsubuhs,SI_ftype_SISI,2)
//
def int_hexagon_A2_svsubuhs :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A2.svsubuhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vraddub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vraddub :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vraddub">;
//
// BUILTIN_INFO(HEXAGON.A2_vraddub_acc,DI_ftype_DIDIDI,3)
//
def int_hexagon_A2_vraddub_acc :
Hexagon_di_dididi_Intrinsic<"HEXAGON.A2.vraddub.acc">;
//
// BUILTIN_INFO(HEXAGON.M2_vradduh,SI_ftype_DIDI,2)
//
def int_hexagon_M2_vradduh :
Hexagon_si_didi_Intrinsic<"HEXAGON.M2.vradduh">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubub :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vsubub">;
//
// BUILTIN_INFO(HEXAGON.A2_vsububs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsububs :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vsububs">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubh :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vsubh">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubhs :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vsubhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubuhs,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubuhs :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vsubuhs">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubw :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vsubw">;
//
// BUILTIN_INFO(HEXAGON.A2_vsubws,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vsubws :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vsubws">;
//
// BUILTIN_INFO(HEXAGON.A2_vabsh,DI_ftype_DI,1)
//
def int_hexagon_A2_vabsh :
Hexagon_di_di_Intrinsic<"HEXAGON.A2.vabsh">;
//
// BUILTIN_INFO(HEXAGON.A2_vabshsat,DI_ftype_DI,1)
//
def int_hexagon_A2_vabshsat :
Hexagon_di_di_Intrinsic<"HEXAGON.A2.vabshsat">;
//
// BUILTIN_INFO(HEXAGON.A2_vabsw,DI_ftype_DI,1)
//
def int_hexagon_A2_vabsw :
Hexagon_di_di_Intrinsic<"HEXAGON.A2.vabsw">;
//
// BUILTIN_INFO(HEXAGON.A2_vabswsat,DI_ftype_DI,1)
//
def int_hexagon_A2_vabswsat :
Hexagon_di_di_Intrinsic<"HEXAGON.A2.vabswsat">;
//
// BUILTIN_INFO(HEXAGON.M2_vabsdiffw,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vabsdiffw :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vabsdiffw">;
//
// BUILTIN_INFO(HEXAGON.M2_vabsdiffh,DI_ftype_DIDI,2)
//
def int_hexagon_M2_vabsdiffh :
Hexagon_di_didi_Intrinsic<"HEXAGON.M2.vabsdiffh">;
//
// BUILTIN_INFO(HEXAGON.A2_vrsadub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vrsadub :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vrsadub">;
//
// BUILTIN_INFO(HEXAGON.A2_vrsadub_acc,DI_ftype_DIDIDI,3)
//
def int_hexagon_A2_vrsadub_acc :
Hexagon_di_dididi_Intrinsic<"HEXAGON.A2.vrsadub.acc">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgub :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavgub">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguh :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavguh">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgh :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgh :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vnavgh">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgw :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavgw">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgw :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vnavgw">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgwr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgwr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavgwr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgwr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgwr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vnavgwr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgwcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgwcr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavgwcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavgwcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavgwcr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vnavgwcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavghcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavghcr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavghcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavghcr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavghcr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vnavghcr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguw :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavguw">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguwr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguwr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavguwr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavgubr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavgubr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavgubr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavguhr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavguhr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavguhr">;
//
// BUILTIN_INFO(HEXAGON.A2_vavghr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vavghr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vavghr">;
//
// BUILTIN_INFO(HEXAGON.A2_vnavghr,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vnavghr :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vnavghr">;
//
// BUILTIN_INFO(HEXAGON.A2_vminh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminh :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vminh">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxh :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vmaxh">;
//
// BUILTIN_INFO(HEXAGON.A2_vminub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminub :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vminub">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxub,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxub :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vmaxub">;
//
// BUILTIN_INFO(HEXAGON.A2_vminuh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminuh :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vminuh">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxuh,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxuh :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vmaxuh">;
//
// BUILTIN_INFO(HEXAGON.A2_vminw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminw :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vminw">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxw :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vmaxw">;
//
// BUILTIN_INFO(HEXAGON.A2_vminuw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vminuw :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vminuw">;
//
// BUILTIN_INFO(HEXAGON.A2_vmaxuw,DI_ftype_DIDI,2)
//
def int_hexagon_A2_vmaxuw :
Hexagon_di_didi_Intrinsic<"HEXAGON.A2.vmaxuw">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.asr.r.r">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.asl.r.r">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_lsr_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.lsr.r.r">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_lsl_r_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.lsl.r.r">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asr.r.p">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asl.r.p">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.lsr.r.p">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsl_r_p :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.lsl.r.p">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asr.r.r.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asl.r.r.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsr.r.r.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsl.r.r.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asr.r.p.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asl.r.p.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsr.r.p.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsl.r.p.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asr.r.r.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asl.r.r.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsr.r.r.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsl.r.r.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asr.r.p.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asl.r.p.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsr.r.p.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsl.r.p.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asr.r.r.and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asl.r.r.and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsr.r.r.and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsl.r.r.and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asr.r.r.or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asl.r.r.or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsr.r.r.or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsl_r_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsl.r.r.or">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asr.r.p.and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asl.r.p.and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsr.r.p.and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsl.r.p.and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asr.r.p.or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asl.r.p.or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsr.r.p.or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsl_r_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsl.r.p.or">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_r_sat,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_r_r_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.asr.r.r.sat">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_r_sat,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_r_r_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.asl.r.r.sat">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_i_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.asr.i.r">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_lsr_i_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.lsr.i.r">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_i_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.asl.i.r">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_p :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asr.i.p">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_i_p :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.lsr.i.p">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_i_p :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asl.i.p">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asr.i.r.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsr.i.r.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_acc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_acc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asl.i.r.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asr.i.p.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsr.i.p.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_acc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_acc :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asl.i.p.acc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asr.i.r.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsr.i.r.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_nac,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_nac :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asl.i.r.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asr.i.p.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsr.i.p.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_nac,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_nac :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asl.i.p.nac">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_xacc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_xacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsr.i.r.xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_xacc,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_xacc :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asl.i.r.xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_xacc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_xacc :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsr.i.p.xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_xacc,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_xacc :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asl.i.p.xacc">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asr.i.r.and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsr.i.r.and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_and,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asl.i.r.and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asr_i_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asr.i.r.or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_lsr_i_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.lsr.i.r.or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_or,SI_ftype_SISISI,3)
//
def int_hexagon_S2_asl_i_r_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.asl.i.r.or">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asr.i.p.and">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsr.i.p.and">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_and,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_and :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asl.i.p.and">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asr_i_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asr.i.p.or">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_lsr_i_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.lsr.i.p.or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_p_or,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_asl_i_p_or :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.asl.i.p.or">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_r_sat,SI_ftype_SISI,2)
//
def int_hexagon_S2_asl_i_r_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.asl.i.r.sat">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_rnd,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_i_r_rnd :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.asr.i.r.rnd">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_r_rnd_goodsyntax,SI_ftype_SISI,2)
//
def int_hexagon_S2_asr_i_r_rnd_goodsyntax :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.asr.i.r.rnd.goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_addasl_rrri,SI_ftype_SISISI,3)
//
def int_hexagon_S2_addasl_rrri :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.addasl.rrri">;
//
// BUILTIN_INFO(HEXAGON.S2_valignib,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_valignib :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.valignib">;
//
// BUILTIN_INFO(HEXAGON.S2_valignrb,DI_ftype_DIDIQI,3)
//
def int_hexagon_S2_valignrb :
Hexagon_di_didiqi_Intrinsic<"HEXAGON.S2.valignrb">;
//
// BUILTIN_INFO(HEXAGON.S2_vspliceib,DI_ftype_DIDISI,3)
//
def int_hexagon_S2_vspliceib :
Hexagon_di_didisi_Intrinsic<"HEXAGON.S2.vspliceib">;
//
// BUILTIN_INFO(HEXAGON.S2_vsplicerb,DI_ftype_DIDIQI,3)
//
def int_hexagon_S2_vsplicerb :
Hexagon_di_didiqi_Intrinsic<"HEXAGON.S2.vsplicerb">;
//
// BUILTIN_INFO(HEXAGON.S2_vsplatrh,DI_ftype_SI,1)
//
def int_hexagon_S2_vsplatrh :
Hexagon_di_si_Intrinsic<"HEXAGON.S2.vsplatrh">;
//
// BUILTIN_INFO(HEXAGON.S2_vsplatrb,SI_ftype_SI,1)
//
def int_hexagon_S2_vsplatrb :
Hexagon_si_si_Intrinsic<"HEXAGON.S2.vsplatrb">;
//
// BUILTIN_INFO(HEXAGON.S2_insert,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_insert :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON.S2.insert">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxb_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxb_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON.S2.tableidxb.goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxh_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxh_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON.S2.tableidxh.goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxw_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxw_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON.S2.tableidxw.goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_tableidxd_goodsyntax,SI_ftype_SISISISI,4)
//
def int_hexagon_S2_tableidxd_goodsyntax :
Hexagon_si_sisisisi_Intrinsic<"HEXAGON.S2.tableidxd.goodsyntax">;
//
// BUILTIN_INFO(HEXAGON.S2_extractu,SI_ftype_SISISI,3)
//
def int_hexagon_S2_extractu :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S2.extractu">;
//
// BUILTIN_INFO(HEXAGON.S2_insertp,DI_ftype_DIDISISI,4)
//
def int_hexagon_S2_insertp :
Hexagon_di_didisisi_Intrinsic<"HEXAGON.S2.insertp">;
//
// BUILTIN_INFO(HEXAGON.S2_extractup,DI_ftype_DISISI,3)
//
def int_hexagon_S2_extractup :
Hexagon_di_disisi_Intrinsic<"HEXAGON.S2.extractup">;
//
// BUILTIN_INFO(HEXAGON.S2_insert_rp,SI_ftype_SISIDI,3)
//
def int_hexagon_S2_insert_rp :
Hexagon_si_sisidi_Intrinsic<"HEXAGON.S2.insert.rp">;
//
// BUILTIN_INFO(HEXAGON.S2_extractu_rp,SI_ftype_SIDI,2)
//
def int_hexagon_S2_extractu_rp :
Hexagon_si_sidi_Intrinsic<"HEXAGON.S2.extractu.rp">;
//
// BUILTIN_INFO(HEXAGON.S2_insertp_rp,DI_ftype_DIDIDI,3)
//
def int_hexagon_S2_insertp_rp :
Hexagon_di_dididi_Intrinsic<"HEXAGON.S2.insertp.rp">;
//
// BUILTIN_INFO(HEXAGON.S2_extractup_rp,DI_ftype_DIDI,2)
//
def int_hexagon_S2_extractup_rp :
Hexagon_di_didi_Intrinsic<"HEXAGON.S2.extractup.rp">;
//
// BUILTIN_INFO(HEXAGON.S2_tstbit_i,QI_ftype_SISI,2)
//
def int_hexagon_S2_tstbit_i :
Hexagon_qi_sisi_Intrinsic<"HEXAGON.S2.tstbit.i">;
//
// BUILTIN_INFO(HEXAGON.S2_setbit_i,SI_ftype_SISI,2)
//
def int_hexagon_S2_setbit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.setbit.i">;
//
// BUILTIN_INFO(HEXAGON.S2_togglebit_i,SI_ftype_SISI,2)
//
def int_hexagon_S2_togglebit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.togglebit.i">;
//
// BUILTIN_INFO(HEXAGON.S2_clrbit_i,SI_ftype_SISI,2)
//
def int_hexagon_S2_clrbit_i :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.clrbit.i">;
//
// BUILTIN_INFO(HEXAGON.S2_tstbit_r,QI_ftype_SISI,2)
//
def int_hexagon_S2_tstbit_r :
Hexagon_qi_sisi_Intrinsic<"HEXAGON.S2.tstbit.r">;
//
// BUILTIN_INFO(HEXAGON.S2_setbit_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_setbit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.setbit.r">;
//
// BUILTIN_INFO(HEXAGON.S2_togglebit_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_togglebit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.togglebit.r">;
//
// BUILTIN_INFO(HEXAGON.S2_clrbit_r,SI_ftype_SISI,2)
//
def int_hexagon_S2_clrbit_r :
Hexagon_si_sisi_Intrinsic<"HEXAGON.S2.clrbit.r">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asr.i.vh">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_i_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.lsr.i.vh">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_i_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asl.i.vh">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asr.r.vh">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asl.r.vh">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.lsr.r.vh">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_vh,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsl_r_vh :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.lsl.r.vh">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asr.i.vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_i_svw_trun,SI_ftype_DISI,2)
//
def int_hexagon_S2_asr_i_svw_trun :
Hexagon_si_disi_Intrinsic<"HEXAGON.S2.asr.i.svw.trun">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_svw_trun,SI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_svw_trun :
Hexagon_si_disi_Intrinsic<"HEXAGON.S2.asr.r.svw.trun">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_i_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_i_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.lsr.i.vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_i_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_i_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asl.i.vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asr_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asr_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asr.r.vw">;
//
// BUILTIN_INFO(HEXAGON.S2_asl_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_asl_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.asl.r.vw">;
//
// BUILTIN_INFO(HEXAGON.S2_lsr_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsr_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.lsr.r.vw">;
//
// BUILTIN_INFO(HEXAGON.S2_lsl_r_vw,DI_ftype_DISI,2)
//
def int_hexagon_S2_lsl_r_vw :
Hexagon_di_disi_Intrinsic<"HEXAGON.S2.lsl.r.vw">;
//
// BUILTIN_INFO(HEXAGON.S2_vrndpackwh,SI_ftype_DI,1)
//
def int_hexagon_S2_vrndpackwh :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.vrndpackwh">;
//
// BUILTIN_INFO(HEXAGON.S2_vrndpackwhs,SI_ftype_DI,1)
//
def int_hexagon_S2_vrndpackwhs :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.vrndpackwhs">;
//
// BUILTIN_INFO(HEXAGON.S2_vsxtbh,DI_ftype_SI,1)
//
def int_hexagon_S2_vsxtbh :
Hexagon_di_si_Intrinsic<"HEXAGON.S2.vsxtbh">;
//
// BUILTIN_INFO(HEXAGON.S2_vzxtbh,DI_ftype_SI,1)
//
def int_hexagon_S2_vzxtbh :
Hexagon_di_si_Intrinsic<"HEXAGON.S2.vzxtbh">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathub,SI_ftype_DI,1)
//
def int_hexagon_S2_vsathub :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.vsathub">;
//
// BUILTIN_INFO(HEXAGON.S2_svsathub,SI_ftype_SI,1)
//
def int_hexagon_S2_svsathub :
Hexagon_si_si_Intrinsic<"HEXAGON.S2.svsathub">;
//
// BUILTIN_INFO(HEXAGON.S2_svsathb,SI_ftype_SI,1)
//
def int_hexagon_S2_svsathb :
Hexagon_si_si_Intrinsic<"HEXAGON.S2.svsathb">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathb,SI_ftype_DI,1)
//
def int_hexagon_S2_vsathb :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.vsathb">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunohb,SI_ftype_DI,1)
//
def int_hexagon_S2_vtrunohb :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.vtrunohb">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunewh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_vtrunewh :
Hexagon_di_didi_Intrinsic<"HEXAGON.S2.vtrunewh">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunowh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_vtrunowh :
Hexagon_di_didi_Intrinsic<"HEXAGON.S2.vtrunowh">;
//
// BUILTIN_INFO(HEXAGON.S2_vtrunehb,SI_ftype_DI,1)
//
def int_hexagon_S2_vtrunehb :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.vtrunehb">;
//
// BUILTIN_INFO(HEXAGON.S2_vsxthw,DI_ftype_SI,1)
//
def int_hexagon_S2_vsxthw :
Hexagon_di_si_Intrinsic<"HEXAGON.S2.vsxthw">;
//
// BUILTIN_INFO(HEXAGON.S2_vzxthw,DI_ftype_SI,1)
//
def int_hexagon_S2_vzxthw :
Hexagon_di_si_Intrinsic<"HEXAGON.S2.vzxthw">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwh,SI_ftype_DI,1)
//
def int_hexagon_S2_vsatwh :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.vsatwh">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwuh,SI_ftype_DI,1)
//
def int_hexagon_S2_vsatwuh :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.vsatwuh">;
//
// BUILTIN_INFO(HEXAGON.S2_packhl,DI_ftype_SISI,2)
//
def int_hexagon_S2_packhl :
Hexagon_di_sisi_Intrinsic<"HEXAGON.S2.packhl">;
//
// BUILTIN_INFO(HEXAGON.A2_swiz,SI_ftype_SI,1)
//
def int_hexagon_A2_swiz :
Hexagon_si_si_Intrinsic<"HEXAGON.A2.swiz">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathub_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsathub_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON.S2.vsathub.nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_vsathb_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsathb_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON.S2.vsathb.nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwh_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsatwh_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON.S2.vsatwh.nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_vsatwuh_nopack,DI_ftype_DI,1)
//
def int_hexagon_S2_vsatwuh_nopack :
Hexagon_di_di_Intrinsic<"HEXAGON.S2.vsatwuh.nopack">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffob,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffob :
Hexagon_di_didi_Intrinsic<"HEXAGON.S2.shuffob">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffeb,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffeb :
Hexagon_di_didi_Intrinsic<"HEXAGON.S2.shuffeb">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffoh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffoh :
Hexagon_di_didi_Intrinsic<"HEXAGON.S2.shuffoh">;
//
// BUILTIN_INFO(HEXAGON.S2_shuffeh,DI_ftype_DIDI,2)
//
def int_hexagon_S2_shuffeh :
Hexagon_di_didi_Intrinsic<"HEXAGON.S2.shuffeh">;
//
// BUILTIN_INFO(HEXAGON.S2_parityp,SI_ftype_DIDI,2)
//
def int_hexagon_S2_parityp :
Hexagon_si_didi_Intrinsic<"HEXAGON.S2.parityp">;
//
// BUILTIN_INFO(HEXAGON.S2_lfsp,DI_ftype_DIDI,2)
//
def int_hexagon_S2_lfsp :
Hexagon_di_didi_Intrinsic<"HEXAGON.S2.lfsp">;
//
// BUILTIN_INFO(HEXAGON.S2_clbnorm,SI_ftype_SI,1)
//
def int_hexagon_S2_clbnorm :
Hexagon_si_si_Intrinsic<"HEXAGON.S2.clbnorm">;
//
// BUILTIN_INFO(HEXAGON.S2_clb,SI_ftype_SI,1)
//
def int_hexagon_S2_clb :
Hexagon_si_si_Intrinsic<"HEXAGON.S2.clb">;
//
// BUILTIN_INFO(HEXAGON.S2_cl0,SI_ftype_SI,1)
//
def int_hexagon_S2_cl0 :
Hexagon_si_si_Intrinsic<"HEXAGON.S2.cl0">;
//
// BUILTIN_INFO(HEXAGON.S2_cl1,SI_ftype_SI,1)
//
def int_hexagon_S2_cl1 :
Hexagon_si_si_Intrinsic<"HEXAGON.S2.cl1">;
//
// BUILTIN_INFO(HEXAGON.S2_clbp,SI_ftype_DI,1)
//
def int_hexagon_S2_clbp :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.clbp">;
//
// BUILTIN_INFO(HEXAGON.S2_cl0p,SI_ftype_DI,1)
//
def int_hexagon_S2_cl0p :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.cl0p">;
//
// BUILTIN_INFO(HEXAGON.S2_cl1p,SI_ftype_DI,1)
//
def int_hexagon_S2_cl1p :
Hexagon_si_di_Intrinsic<"HEXAGON.S2.cl1p">;
//
// BUILTIN_INFO(HEXAGON.S2_brev,SI_ftype_SI,1)
//
def int_hexagon_S2_brev :
Hexagon_si_si_Intrinsic<"HEXAGON.S2.brev">;
//
// BUILTIN_INFO(HEXAGON.S2_ct0,SI_ftype_SI,1)
//
def int_hexagon_S2_ct0 :
Hexagon_si_si_Intrinsic<"HEXAGON.S2.ct0">;
//
// BUILTIN_INFO(HEXAGON.S2_ct1,SI_ftype_SI,1)
//
def int_hexagon_S2_ct1 :
Hexagon_si_si_Intrinsic<"HEXAGON.S2.ct1">;
//
// BUILTIN_INFO(HEXAGON.S2_interleave,DI_ftype_DI,1)
//
def int_hexagon_S2_interleave :
Hexagon_di_di_Intrinsic<"HEXAGON.S2.interleave">;
//
// BUILTIN_INFO(HEXAGON.S2_deinterleave,DI_ftype_DI,1)
//
def int_hexagon_S2_deinterleave :
Hexagon_di_di_Intrinsic<"HEXAGON.S2.deinterleave">;

//
// BUILTIN_INFO(SI_to_SXTHI_asrh,SI_ftype_SI,1)
//
def int_hexagon_SI_to_SXTHI_asrh :
Hexagon_si_si_Intrinsic<"SI.to.SXTHI.asrh">;

//
// BUILTIN_INFO(HEXAGON.A4_orn,SI_ftype_SISI,2)
//
def int_hexagon_A4_orn :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.orn">;
//
// BUILTIN_INFO(HEXAGON.A4_andn,SI_ftype_SISI,2)
//
def int_hexagon_A4_andn :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.andn">;
//
// BUILTIN_INFO(HEXAGON.A4_orn,DI_ftype_DIDI,2)
//
def int_hexagon_A4_ornp :
Hexagon_di_didi_Intrinsic<"HEXAGON.A4.ornp">;
//
// BUILTIN_INFO(HEXAGON.A4_andn,DI_ftype_DIDI,2)
//
def int_hexagon_A4_andnp :
Hexagon_di_didi_Intrinsic<"HEXAGON.A4.andnp">;
//
// BUILTIN_INFO(HEXAGON.A4_combineir,DI_ftype_sisi,2)
//
def int_hexagon_A4_combineir :
Hexagon_di_sisi_Intrinsic<"HEXAGON.A4.combineir">;
//
// BUILTIN_INFO(HEXAGON.A4_combineir,DI_ftype_sisi,2)
//
def int_hexagon_A4_combineri :
Hexagon_di_sisi_Intrinsic<"HEXAGON.A4.combineri">;
//
// BUILTIN_INFO(HEXAGON.C4_cmpneq,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmpneq :
Hexagon_qi_sisi_Intrinsic<"HEXAGON.C4.cmpneq">;
//
// BUILTIN_INFO(HEXAGON.C4_cmpneqi,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmpneqi :
Hexagon_qi_sisi_Intrinsic<"HEXAGON.C4.cmpneqi">;
//
// BUILTIN_INFO(HEXAGON.C4_cmplte,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmplte :
Hexagon_qi_sisi_Intrinsic<"HEXAGON.C4.cmplte">;
//
// BUILTIN_INFO(HEXAGON.C4_cmpltei,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmpltei :
Hexagon_qi_sisi_Intrinsic<"HEXAGON.C4.cmpltei">;
//
// BUILTIN_INFO(HEXAGON.C4_cmplteu,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmplteu :
Hexagon_qi_sisi_Intrinsic<"HEXAGON.C4.cmplteu">;
//
// BUILTIN_INFO(HEXAGON.C4_cmplteui,QI_ftype_SISI,2)
//
def int_hexagon_C4_cmplteui :
Hexagon_qi_sisi_Intrinsic<"HEXAGON.C4.cmplteui">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpneq,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpneq :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.rcmpneq">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpneqi,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpneqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.rcmpneqi">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpeq,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpeq :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.rcmpeq">;
//
// BUILTIN_INFO(HEXAGON.A4_rcmpeqi,SI_ftype_SISI,2)
//
def int_hexagon_A4_rcmpeqi :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.rcmpeqi">;
//
// BUILTIN_INFO(HEXAGON.C4_fastcorner9,QI_ftype_QIQI,2)
//
def int_hexagon_C4_fastcorner9 :
Hexagon_qi_qiqi_Intrinsic<"HEXAGON.C4.fastcorner9">;
//
// BUILTIN_INFO(HEXAGON.C4_fastcorner9_not,QI_ftype_QIQI,2)
//
def int_hexagon_C4_fastcorner9_not :
Hexagon_qi_qiqi_Intrinsic<"HEXAGON.C4.fastcorner9_not">;
//
// BUILTIN_INFO(HEXAGON.C4_and_andn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_andn :
Hexagon_qi_qiqiqi_Intrinsic<"HEXAGON.C4.and_andn">;
//
// BUILTIN_INFO(HEXAGON.C4_and_and,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_and :
Hexagon_qi_qiqiqi_Intrinsic<"HEXAGON.C4.and_and">;
//
// BUILTIN_INFO(HEXAGON.C4_and_orn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_orn :
Hexagon_qi_qiqiqi_Intrinsic<"HEXAGON.C4.and_orn">;
//
// BUILTIN_INFO(HEXAGON.C4_and_or,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_and_or :
Hexagon_qi_qiqiqi_Intrinsic<"HEXAGON.C4.and_or">;
//
// BUILTIN_INFO(HEXAGON.C4_or_andn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_andn :
Hexagon_qi_qiqiqi_Intrinsic<"HEXAGON.C4.or_andn">;
//
// BUILTIN_INFO(HEXAGON.C4_or_and,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_and :
Hexagon_qi_qiqiqi_Intrinsic<"HEXAGON.C4.or_and">;
//
// BUILTIN_INFO(HEXAGON.C4_or_orn,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_orn :
Hexagon_qi_qiqiqi_Intrinsic<"HEXAGON.C4.or_orn">;
//
// BUILTIN_INFO(HEXAGON.C4_or_or,QI_ftype_QIQIQI,3)
//
def int_hexagon_C4_or_or :
Hexagon_qi_qiqiqi_Intrinsic<"HEXAGON.C4.or_or">;
//
// BUILTIN_INFO(HEXAGON.S4_addaddi,SI_ftype_SISISI,3)
//
def int_hexagon_S4_addaddi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S4.addaddi">;
//
// BUILTIN_INFO(HEXAGON.S4_subaddi,SI_ftype_SISISI,3)
//
def int_hexagon_S4_subaddi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S4.subaddi">;
//
// BUILTIN_INFO(HEXAGON.S4_andnp,DI_ftype_DIDI,2)
//
def int_hexagon_S4_andnp :
Hexagon_di_didi_Intrinsic<"HEXAGON.S4.andnp">;
//
// BUILTIN_INFO(HEXAGON.S4_ornp,DI_ftype_DIDI,2)
//
def int_hexagon_S4_ornp :
Hexagon_di_didi_Intrinsic<"HEXAGON.S4.ornp">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_xacc,DI_ftype_DIDIDI,3)
//
def int_hexagon_M4_xor_xacc :
Hexagon_di_dididi_Intrinsic<"HEXAGON.M4.xor_xacc">;
//
// BUILTIN_INFO(HEXAGON.M4_and_and,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.and_and">;
//
// BUILTIN_INFO(HEXAGON.M4_and_andn,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.and_andn">;
//
// BUILTIN_INFO(HEXAGON.M4_and_or,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.and_or">;
//
// BUILTIN_INFO(HEXAGON.M4_and_xor,SI_ftype_SISISI,3)
//
def int_hexagon_M4_and_xor :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.and_xor">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_and,SI_ftype_SISISI,3)
//
def int_hexagon_M4_xor_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.xor_or">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_or,SI_ftype_SISISI,3)
//
def int_hexagon_M4_xor_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.xor_and">;
//
// BUILTIN_INFO(HEXAGON.M4_xor_andn,SI_ftype_SISISI,3)
//
def int_hexagon_M4_xor_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.xor_andn">;
//
// BUILTIN_INFO(HEXAGON.M4_or_and,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_and :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.or_and">;
//
// BUILTIN_INFO(HEXAGON.M4_or_or,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_or :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.or_or">;
//
// BUILTIN_INFO(HEXAGON.M4_or_xor,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_xor :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.or_xor">;
//
// BUILTIN_INFO(HEXAGON.M4_or_andn,SI_ftype_SISISI,3)
//
def int_hexagon_M4_or_andn :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.M4.or_andn">;
//
// BUILTIN_INFO(HEXAGON.S4_or_andix,SI_ftype_SISISI,3)
//
def int_hexagon_S4_or_andix :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S4.or_andix">;
//
// BUILTIN_INFO(HEXAGON.S4_or_andi,SI_ftype_SISISI,3)
//
def int_hexagon_S4_or_andi :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S4.or_andi">;
//
// BUILTIN_INFO(HEXAGON.S4_or_ori,SI_ftype_SISISI,3)
//
def int_hexagon_S4_or_ori :
Hexagon_si_sisisi_Intrinsic<"HEXAGON.S4.or_ori">;
//
// BUILTIN_INFO(HEXAGON.A4_modwrapu,SI_ftype_SISI,2)
//
def int_hexagon_A4_modwrapu :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.modwrapu">;
//
// BUILTIN_INFO(HEXAGON.A4_cround_ri,SI_ftype_SISI,2)
//
def int_hexagon_A4_cround_ri :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.cround_ri">;
//
// BUILTIN_INFO(HEXAGON.A4_cround_rr,SI_ftype_SISI,2)
//
def int_hexagon_A4_cround_rr :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.cround_rr">;
//
// BUILTIN_INFO(HEXAGON.A4_round_ri,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_ri :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.round_ri">;
//
// BUILTIN_INFO(HEXAGON.A4_round_rr,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_rr :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.round_rr">;
//
// BUILTIN_INFO(HEXAGON.A4_round_ri_sat,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_ri_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.round_ri_sat">;
//
// BUILTIN_INFO(HEXAGON.A4_round_rr_sat,SI_ftype_SISI,2)
//
def int_hexagon_A4_round_rr_sat :
Hexagon_si_sisi_Intrinsic<"HEXAGON.A4.round_rr_sat">;
