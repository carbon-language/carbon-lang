; RUN: echo '!9 = metadata !{metadata !"%s", metadata !0}' > %t1
; RUN: cat %s %t1 > %t2
; RUN: opt -insert-gcov-profiling -disable-output < %t2
; RUN: head -c12 version.gcno | grep '^oncg\*204MVLL$'
; RUN: rm version.gcno
; RUN: not opt -insert-gcov-profiling -default-gcov-version=asdfasdf -disable-output < %t2
; RUN: opt -insert-gcov-profiling -default-gcov-version=407* -disable-output < %t2
; RUN: head -c12 version.gcno | grep '^oncg\*704MVLL$'
; RUN: rm version.gcno

define void @test() {
  ret void, !dbg !8
}

!llvm.gcov = !{!9}
!llvm.dbg.cu = !{!0}

!0 = metadata !{metadata !"%s/version", metadata !1}
!1 = metadata !{i32 786449, i32 0, i32 4, metadata !2, metadata !"clang version 3.3 (trunk 176994)", i1 false, metadata !"", i32 0, metadata !3, metadata !3, metadata !4, metadata !3, metadata !""} ; [ DW_TAG_compile_unit ] [./version] [DW_LANG_C_plus_plus]
!2 = metadata !{i32 786473, metadata !"version", metadata !"/usr/local/google/home/nlewycky"} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 0}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786478, i32 0, metadata !6, metadata !"test", metadata !"test", metadata !"", metadata !6, i32 1, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @test, null, null, metadata !3, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [test]
!6 = metadata !{i32 786473, metadata !"<stdin>", metadata !"."} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !3, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8= metadata !{i32 1, i32 0, metadata !5, null}
;; !9 is added through the echo line at the top.
