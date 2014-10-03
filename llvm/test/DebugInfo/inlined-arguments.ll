; REQUIRES: object-emission

; RUN: %llc_dwarf -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated from clang -O -g with the following source
;
; void f1(int x, int y);
; void f3(int line);
; void f2() {
;   f1(1, 2);
; }
; void f1(int x, int y) {
;   f3(y);
; }

; CHECK: DW_AT_name{{.*}}"f1"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"x"
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"y"

; Function Attrs: uwtable
define void @_Z2f2v() #0 {
  tail call void @llvm.dbg.value(metadata !15, i64 0, metadata !16, metadata !{metadata !"0x102"}), !dbg !18
  tail call void @llvm.dbg.value(metadata !19, i64 0, metadata !20, metadata !{metadata !"0x102"}), !dbg !18
  tail call void @_Z2f3i(i32 2), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: uwtable
define void @_Z2f1ii(i32 %x, i32 %y) #0 {
  tail call void @llvm.dbg.value(metadata !{i32 %x}, i64 0, metadata !13, metadata !{metadata !"0x102"}), !dbg !23
  tail call void @llvm.dbg.value(metadata !{i32 %y}, i64 0, metadata !14, metadata !{metadata !"0x102"}), !dbg !23
  tail call void @_Z2f3i(i32 %y), !dbg !24
  ret void, !dbg !25
}

declare void @_Z2f3i(i32) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 \001\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/exp.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"exp.cpp", metadata !"/usr/local/google/home/blaikie/dev/scratch"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !8}
!4 = metadata !{metadata !"0x2e\00f2\00f2\00_Z2f2v\003\000\001\000\006\00256\001\003", metadata !1, metadata !5, metadata !6, null, void ()* @_Z2f2v, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 3] [def] [f2]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/exp.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !"0x2e\00f1\00f1\00_Z2f1ii\006\000\001\000\006\00256\001\006", metadata !1, metadata !5, metadata !9, null, void (i32, i32)* @_Z2f1ii, null, null, metadata !12} ; [ DW_TAG_subprogram ] [line 6] [def] [f1]
!9 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11, metadata !11}
!11 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13, metadata !14}
!13 = metadata !{metadata !"0x101\00x\0016777222\000", metadata !8, metadata !5, metadata !11} ; [ DW_TAG_arg_variable ] [x] [line 6]
!14 = metadata !{metadata !"0x101\00y\0033554438\000", metadata !8, metadata !5, metadata !11} ; [ DW_TAG_arg_variable ] [y] [line 6]
!15 = metadata !{i32 undef}
!16 = metadata !{metadata !"0x101\00x\0016777222\000", metadata !8, metadata !5, metadata !11, metadata !17} ; [ DW_TAG_arg_variable ] [x] [line 6]
!17 = metadata !{i32 4, i32 0, metadata !4, null}
!18 = metadata !{i32 6, i32 0, metadata !8, metadata !17}
!19 = metadata !{i32 2}
!20 = metadata !{metadata !"0x101\00y\0033554438\000", metadata !8, metadata !5, metadata !11, metadata !17} ; [ DW_TAG_arg_variable ] [y] [line 6]
!21 = metadata !{i32 7, i32 0, metadata !8, metadata !17}
!22 = metadata !{i32 5, i32 0, metadata !4, null}
!23 = metadata !{i32 6, i32 0, metadata !8, null}
!24 = metadata !{i32 7, i32 0, metadata !8, null}
!25 = metadata !{i32 8, i32 0, metadata !8, null}
!26 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
