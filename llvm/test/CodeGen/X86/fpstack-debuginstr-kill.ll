; RUN: llc < %s -mcpu=generic -mtriple=i386-apple-darwin -no-integrated-as

@g1 = global double 0.000000e+00, align 8
@g2 = global i32 0, align 4

define void @_Z16fpuop_arithmeticjj(i32, i32) {
entry:
  switch i32 undef, label %sw.bb.i1921 [
  ]

sw.bb261:                                         ; preds = %entry, %entry
  unreachable

sw.bb.i1921:                                      ; preds = %if.end504
  switch i32 undef, label %if.end511 [
    i32 1, label %sw.bb27.i
  ]

sw.bb27.i:                                        ; preds = %sw.bb.i1921
  %conv.i.i1923 = fpext float undef to x86_fp80
  br label %if.end511

if.end511:                                        ; preds = %sw.bb27.i, %sw.bb13.i
  %src.sroa.0.0.src.sroa.0.0.2280 = phi x86_fp80 [ %conv.i.i1923, %sw.bb27.i ], [ undef, %sw.bb.i1921 ]
  switch i32 undef, label %sw.bb992 [
    i32 3, label %sw.bb735
    i32 18, label %if.end41.i2210
  ]

sw.bb735:                                         ; preds = %if.end511
  %2 = call x86_fp80 asm sideeffect "frndint", "={st},0,~{dirflag},~{fpsr},~{flags}"(x86_fp80 %src.sroa.0.0.src.sroa.0.0.2280)
  unreachable

if.end41.i2210:                                   ; preds = %if.end511
  call void @llvm.dbg.value(metadata !{x86_fp80 %src.sroa.0.0.src.sroa.0.0.2280}, i64 0, metadata !20)
  unreachable

sw.bb992:                                         ; preds = %if.end511
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24, !25}
!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.6.0 (http://llvm.org/git/clang 8444ae7cfeaefae031f8fedf0d1435ca3b14d90b) (http://llvm.org/git/llvm 886f0101a7d176543b831f5efb74c03427244a55)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !21, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [x87stackifier/fpu_ieee.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"fpu_ieee.cpp", metadata !"x87stackifier"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"fpuop_arithmetic", metadata !"fpuop_arithmetic", metadata !"_Z16fpuop_arithmeticjj", i32 11, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void (i32, i32)* @_Z16fpuop_arithmeticjj, null, null, metadata !10, i32 13} ; [ DW_TAG_subprogram ] [line 11] [def] [scope 13] [fpuop_arithmetic]
!5 = metadata !{metadata !"f1.cpp", metadata !"x87stackifier"}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [x87stackifier/f1.cpp]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9, metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"unsigned int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!10 = metadata !{metadata !11, metadata !12, metadata !13, metadata !18, metadata !20}
!11 = metadata !{i32 786689, metadata !4, metadata !"", metadata !6, i32 16777227, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [line 11]
!12 = metadata !{i32 786689, metadata !4, metadata !"", metadata !6, i32 33554443, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [line 11]
!13 = metadata !{i32 786688, metadata !4, metadata !"x", metadata !6, i32 14, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [x] [line 14]
!14 = metadata !{i32 786454, metadata !5, null, metadata !"fpu_extended", i32 3, i64 0, i64 0, i64 0, i32 0, metadata !15} ; [ DW_TAG_typedef ] [fpu_extended] [line 3, size 0, align 0, offset 0] [from fpu_register]
!15 = metadata !{i32 786454, metadata !5, null, metadata !"fpu_register", i32 2, i64 0, i64 0, i64 0, i32 0, metadata !16} ; [ DW_TAG_typedef ] [fpu_register] [line 2, size 0, align 0, offset 0] [from uae_f64]
!16 = metadata !{i32 786454, metadata !5, null, metadata !"uae_f64", i32 1, i64 0, i64 0, i64 0, i32 0, metadata !17} ; [ DW_TAG_typedef ] [uae_f64] [line 1, size 0, align 0, offset 0] [from double]
!17 = metadata !{i32 786468, null, null, metadata !"double", i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!18 = metadata !{i32 786688, metadata !4, metadata !"a", metadata !6, i32 15, metadata !19, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 15]
!19 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!20 = metadata !{i32 786688, metadata !4, metadata !"value", metadata !6, i32 16, metadata !14, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [value] [line 16]
!21 = metadata !{metadata !22, metadata !23}
!22 = metadata !{i32 786484, i32 0, null, metadata !"g1", metadata !"g1", metadata !"", metadata !6, i32 5, metadata !14, i32 0, i32 1, double* @g1, null} ; [ DW_TAG_variable ] [g1] [line 5] [def]
!23 = metadata !{i32 786484, i32 0, null, metadata !"g2", metadata !"g2", metadata !"", metadata !6, i32 6, metadata !19, i32 0, i32 1, i32* @g2, null} ; [ DW_TAG_variable ] [g2] [line 6] [def]
!24 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!25 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
