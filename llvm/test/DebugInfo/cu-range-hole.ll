; REQUIRES: object-emission
; RUN: %llc_dwarf -O0 -filetype=obj %s -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; Check that we emit ranges for this CU since we have a function with and
; without debug info.
; Note: This depends upon the order of output in the .o file. Currently it's
; in order of the output to make sure that the CU has multiple ranges since
; there's a function in the middle. If they were together then it would have
; a single range and no DW_AT_ranges.
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_ranges
; CHECK: DW_TAG_subprogram
; CHECK: DW_TAG_subprogram

; Function Attrs: nounwind uwtable
define i32 @b(i32 %c) #0 {
entry:
  %c.addr = alloca i32, align 4
  store i32 %c, i32* %c.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %c.addr}, metadata !13), !dbg !14
  %0 = load i32* %c.addr, align 4, !dbg !14
  %add = add nsw i32 %0, 1, !dbg !14
  ret i32 %add, !dbg !14
}

; Function Attrs: nounwind uwtable
define i32 @a(i32 %b) #0 {
entry:
  %b.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  %0 = load i32* %b.addr, align 4
  %add = add nsw i32 %0, 1
  ret i32 %add
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @d(i32 %e) #0 {
entry:
  %e.addr = alloca i32, align 4
  store i32 %e, i32* %e.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %e.addr}, metadata !15), !dbg !16
  %0 = load i32* %e.addr, align 4, !dbg !16
  %add = add nsw i32 %0, 1, !dbg !16
  ret i32 %add, !dbg !16
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.ident = !{!0, !0}
!llvm.dbg.cu = !{!1}
!llvm.module.flags = !{!11, !12}

!0 = metadata !{metadata !"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)"}
!1 = metadata !{i32 786449, metadata !2, i32 12, metadata !"clang version 3.5.0 (trunk 204164) (llvm/trunk 204183)", i1 false, metadata !"", i32 0, metadata !3, metadata !3, metadata !4, metadata !3, metadata !3, metadata !"", i32 1}
!2 = metadata !{metadata !"b.c", metadata !"/usr/local/google/home/echristo"}
!3 = metadata !{}
!4 = metadata !{metadata !5, metadata !10}
!5 = metadata !{i32 786478, metadata !2, metadata !6, metadata !"b", metadata !"b", metadata !"", i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @b, null, null, metadata !3, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [b]
!6 = metadata !{i32 786473, metadata !2}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/b.c]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786478, metadata !2, metadata !6, metadata !"d", metadata !"d", metadata !"", i32 3, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @d, null, null, metadata !3, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [d]
!11 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!13 = metadata !{i32 786689, metadata !5, metadata !"c", metadata !6, i32 16777217, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [c] [line 1]
!14 = metadata !{i32 1, i32 0, metadata !5, null}
!15 = metadata !{i32 786689, metadata !10, metadata !"e", metadata !6, i32 16777219, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [e] [line 3]
!16 = metadata !{i32 3, i32 0, metadata !10, null}
