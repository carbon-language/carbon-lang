; RUN: echo '!9 = metadata !{metadata !"%T/linkagename.ll", metadata !0}' > %t1
; RUN: cat %s %t1 > %t2
; RUN: opt -insert-gcov-profiling -disable-output < %t2
; RUN: grep _Z3foov %T/linkagename.gcno
; RUN: rm %T/linkagename.gcno

; REQUIRES: shell

define void @_Z3foov() {
entry:
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}
!llvm.gcov = !{!9}

!0 = metadata !{metadata !"0x11\004\00clang version 3.3 (trunk 177323)\000\00\000\00\000", metadata !2, metadata !3, metadata !3, metadata !4, metadata !3,  metadata !3} ; [ DW_TAG_compile_unit ] [/home/nlewycky/hello.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"0x29", metadata !2}          ; [ DW_TAG_file_type ] [/home/nlewycky/hello.cc]
!2 = metadata !{metadata !"hello.cc", metadata !"/home/nlewycky"}
!3 = metadata !{i32 0}
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00foo\00foo\00_Z3foov\001\000\001\000\006\00256\000\001", metadata !1, metadata !1, metadata !6, null, void ()* @_Z3foov, null, null, metadata !3} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 1, i32 0, metadata !5, null}


!10 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
