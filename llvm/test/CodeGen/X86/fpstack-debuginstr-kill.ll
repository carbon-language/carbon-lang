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
  call void @llvm.dbg.value(metadata x86_fp80 %src.sroa.0.0.src.sroa.0.0.2280, i64 0, metadata !20, metadata !{!"0x102"})
  unreachable

sw.bb992:                                         ; preds = %if.end511
  ret void
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24, !25}
!0 = !{!"0x11\004\00clang version 3.6.0 (http://llvm.org/git/clang 8444ae7cfeaefae031f8fedf0d1435ca3b14d90b) (http://llvm.org/git/llvm 886f0101a7d176543b831f5efb74c03427244a55)\001\00\000\00\001", !1, !2, !2, !3, !21, !2} ; [ DW_TAG_compile_unit ] [x87stackifier/fpu_ieee.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"fpu_ieee.cpp", !"x87stackifier"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00fpuop_arithmetic\00fpuop_arithmetic\00_Z16fpuop_arithmeticjj\0011\000\001\000\006\00256\001\0013", !5, !6, !7, null, void (i32, i32)* @_Z16fpuop_arithmeticjj, null, null, !10} ; [ DW_TAG_subprogram ] [line 11] [def] [scope 13] [fpuop_arithmetic]
!5 = !{!"f1.cpp", !"x87stackifier"}
!6 = !{!"0x29", !5}          ; [ DW_TAG_file_type ] [x87stackifier/f1.cpp]
!7 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = !{null, !9, !9}
!9 = !{!"0x24\00unsigned int\000\0032\0032\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!10 = !{!11, !12, !13, !18, !20}
!11 = !{!"0x101\00\0016777227\000", !4, !6, !9} ; [ DW_TAG_arg_variable ] [line 11]
!12 = !{!"0x101\00\0033554443\000", !4, !6, !9} ; [ DW_TAG_arg_variable ] [line 11]
!13 = !{!"0x100\00x\0014\000", !4, !6, !14} ; [ DW_TAG_auto_variable ] [x] [line 14]
!14 = !{!"0x16\00fpu_extended\003\000\000\000\000", !5, null, !15} ; [ DW_TAG_typedef ] [fpu_extended] [line 3, size 0, align 0, offset 0] [from fpu_register]
!15 = !{!"0x16\00fpu_register\002\000\000\000\000", !5, null, !16} ; [ DW_TAG_typedef ] [fpu_register] [line 2, size 0, align 0, offset 0] [from uae_f64]
!16 = !{!"0x16\00uae_f64\001\000\000\000\000", !5, null, !17} ; [ DW_TAG_typedef ] [uae_f64] [line 1, size 0, align 0, offset 0] [from double]
!17 = !{!"0x24\00double\000\0064\0064\000\000\004", null, null} ; [ DW_TAG_base_type ] [double] [line 0, size 64, align 64, offset 0, enc DW_ATE_float]
!18 = !{!"0x100\00a\0015\000", !4, !6, !19} ; [ DW_TAG_auto_variable ] [a] [line 15]
!19 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!20 = !{!"0x100\00value\0016\000", !4, !6, !14} ; [ DW_TAG_auto_variable ] [value] [line 16]
!21 = !{!22, !23}
!22 = !{!"0x34\00g1\00g1\00\005\000\001", null, !6, !14, double* @g1, null} ; [ DW_TAG_variable ] [g1] [line 5] [def]
!23 = !{!"0x34\00g2\00g2\00\006\000\001", null, !6, !19, i32* @g2, null} ; [ DW_TAG_variable ] [g2] [line 6] [def]
!24 = !{i32 2, !"Dwarf Version", i32 2}
!25 = !{i32 2, !"Debug Info Version", i32 2}
