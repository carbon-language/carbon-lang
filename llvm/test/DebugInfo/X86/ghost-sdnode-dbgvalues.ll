; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-apple-macosx10.10.0 -o %t %s

; Testcase generated from:
; #include <stdint.h>
; int foo(int a) {
;     int b = (int16_t)a + 8;
;     int c = (int16_t)b + 8;
;     int d = (int16_t)c + 8;
;     int e = (int16_t)d + 8;
;     int f = (int16_t)e + 8;
;     return f;
; }
; by emitting the IR and then manually applying mem2reg to it.

; This testcase would trigger the assert commited along with it if the
; fix of r221709 isn't applied. There is no other check except the successful
; run of llc.
; What happened before r221709, is that SDDbgInfo (the data structure helping
; SelectionDAG to keep track of dbg.values) kept a map keyed by SDNode pointers.
; This map was never purged when the SDNodes were deallocated and thus if a new
; SDNode was allocated in the same memory, it would have an entry in the SDDbgInfo
; map upon creation (Reallocation in the same memory can happen easily as
; SelectionDAG uses a Recycling allocator). This behavior could turn into a
; pathological memory consumption explosion if the DAG combiner hit the 'right'
; allocation patterns as could be seen in PR20893.
; By nature, this test could bitrot quite easily. If it doesn't trigger an assert
; when run with r221709 reverted, then it really doesn't test anything anymore.

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i32 %a) #0 {
entry:
  call void @llvm.dbg.value(metadata !{i32 %a}, i64 0, metadata !16, metadata !17), !dbg !18
  %conv = trunc i32 %a to i16, !dbg !19
  %conv1 = sext i16 %conv to i32, !dbg !19
  %add = add nsw i32 %conv1, 8, !dbg !19
  call void @llvm.dbg.value(metadata !{i32 %add}, i64 0, metadata !20, metadata !17), !dbg !21
  %conv2 = trunc i32 %add to i16, !dbg !22
  %conv3 = sext i16 %conv2 to i32, !dbg !22
  %add4 = add nsw i32 %conv3, 8, !dbg !22
  call void @llvm.dbg.value(metadata !{i32 %add4}, i64 0, metadata !23, metadata !17), !dbg !24
  %conv5 = trunc i32 %add4 to i16, !dbg !25
  %conv6 = sext i16 %conv5 to i32, !dbg !25
  %add7 = add nsw i32 %conv6, 8, !dbg !25
  call void @llvm.dbg.value(metadata !{i32 %add7}, i64 0, metadata !26, metadata !17), !dbg !27
  %conv8 = trunc i32 %add7 to i16, !dbg !28
  %conv9 = sext i16 %conv8 to i32, !dbg !28
  %add10 = add nsw i32 %conv9, 8, !dbg !28
  call void @llvm.dbg.value(metadata !{i32 %add10}, i64 0, metadata !29, metadata !17), !dbg !30
  %conv11 = trunc i32 %add10 to i16, !dbg !31
  %conv12 = sext i16 %conv11 to i32, !dbg !31
  %add13 = add nsw i32 %conv12, 8, !dbg !31
  call void @llvm.dbg.value(metadata !{i32 %add13}, i64 0, metadata !32, metadata !17), !dbg !33
  ret i32 %add13, !dbg !34
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.6.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !3, metadata !7, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/tmp/ghost-sdnode-dbgvalues.c] [DW_LANG_C99]
!1 = metadata !{metadata !"ghost-sdnode-dbgvalues.c", metadata !"/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x16\00int16_t\0030\000\000\000\000", metadata !5, null, metadata !6} ; [ DW_TAG_typedef ] [int16_t] [line 30, size 0, align 0, offset 0] [from short]
!5 = metadata !{metadata !"/usr/include/sys/_types/_int16_t.h", metadata !"/tmp"}
!6 = metadata !{metadata !"0x24\00short\000\0016\0016\000\000\005", null, null} ; [ DW_TAG_base_type ] [short] [line 0, size 16, align 16, offset 0, enc DW_ATE_signed]
!7 = metadata !{metadata !8}
!8 = metadata !{metadata !"0x2e\00foo\00foo\00\003\000\001\000\000\00256\000\003", metadata !1, metadata !9, metadata !10, null, i32 (i32)* @foo, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 3] [def] [foo]
!9 = metadata !{metadata !"0x29", metadata !1}    ; [ DW_TAG_file_type ] [/tmp/ghost-sdnode-dbgvalues.c]
!10 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{metadata !12, metadata !12}
!12 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!13 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!14 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!15 = metadata !{metadata !"clang version 3.6.0 "}
!16 = metadata !{metadata !"0x101\00a\0016777219\000", metadata !8, metadata !9, metadata !12} ; [ DW_TAG_arg_variable ] [a] [line 3]
!17 = metadata !{metadata !"0x102"}               ; [ DW_TAG_expression ]
!18 = metadata !{i32 3, i32 13, metadata !8, null}
!19 = metadata !{i32 4, i32 5, metadata !8, null}
!20 = metadata !{metadata !"0x100\00b\004\000", metadata !8, metadata !9, metadata !12} ; [ DW_TAG_auto_variable ] [b] [line 4]
!21 = metadata !{i32 4, i32 9, metadata !8, null}
!22 = metadata !{i32 5, i32 5, metadata !8, null}
!23 = metadata !{metadata !"0x100\00c\005\000", metadata !8, metadata !9, metadata !12} ; [ DW_TAG_auto_variable ] [c] [line 5]
!24 = metadata !{i32 5, i32 9, metadata !8, null}
!25 = metadata !{i32 6, i32 5, metadata !8, null}
!26 = metadata !{metadata !"0x100\00d\006\000", metadata !8, metadata !9, metadata !12} ; [ DW_TAG_auto_variable ] [d] [line 6]
!27 = metadata !{i32 6, i32 9, metadata !8, null}
!28 = metadata !{i32 7, i32 5, metadata !8, null}
!29 = metadata !{metadata !"0x100\00e\007\000", metadata !8, metadata !9, metadata !12} ; [ DW_TAG_auto_variable ] [e] [line 7]
!30 = metadata !{i32 7, i32 9, metadata !8, null}
!31 = metadata !{i32 8, i32 5, metadata !8, null}
!32 = metadata !{metadata !"0x100\00f\008\000", metadata !8, metadata !9, metadata !12} ; [ DW_TAG_auto_variable ] [f] [line 8]
!33 = metadata !{i32 8, i32 9, metadata !8, null}
!34 = metadata !{i32 9, i32 5, metadata !8, null}
