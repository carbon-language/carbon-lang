; RUN: opt %s -sroa -verify -S -o - | FileCheck %s
; ModuleID = 'test.c'
; Test that SROA updates the debug info correctly if an alloca was rewritten but
; not partitioned into multiple allocas.
;
; CHECK: call void @llvm.dbg.value(metadata float %s.coerce, i64 0, metadata ![[VAR:[0-9]+]], metadata ![[EXPR:[0-9]+]])
; CHECK: ![[VAR]] = {{.*}} [ DW_TAG_arg_variable ] [s] [line 3]
; CHECK: ![[EXPR]] = {{.*}} [ DW_TAG_expression ]
; CHECK-NOT: DW_OP_bit_piece

;
; struct S { float f; };
;  
; float foo(struct S s) {
;   return s.f;
; }
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%struct.S = type { float }

; Function Attrs: nounwind ssp uwtable
define float @foo(float %s.coerce) #0 {
entry:
  %s = alloca %struct.S, align 4
  %coerce.dive = getelementptr %struct.S, %struct.S* %s, i32 0, i32 0
  store float %s.coerce, float* %coerce.dive, align 1
  call void @llvm.dbg.declare(metadata %struct.S* %s, metadata !16, metadata !17), !dbg !18
  %f = getelementptr inbounds %struct.S, %struct.S* %s, i32 0, i32 0, !dbg !19
  %0 = load float* %f, align 4, !dbg !19
  ret float %0, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!12, !13, !14}
!llvm.ident = !{!15}

!0 = !{!"0x11\0012\00clang version 3.6.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/Volumes/Data/llvm/_build.ninja.debug/test.c] [DW_LANG_C99]
!1 = !{!"test.c", !"/Volumes/Data/llvm/_build.ninja.debug"}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\003\000\001\000\000\00256\000\003", !1, !5, !6, null, float (float)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [def] [foo]
!5 = !{!"0x29", !1}                               ; [ DW_TAG_file_type ] [/Volumes/Data/llvm/_build.ninja.debug/test.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !9}
!8 = !{!"0x24\00float\000\0032\0032\000\000\004", null, null} ; [ DW_TAG_base_type ] [float] [line 0, size 32, align 32, offset 0, enc DW_ATE_float]
!9 = !{!"0x13\00S\001\0032\0032\000\000\000", !1, null, null, !10, null, null, null} ; [ DW_TAG_structure_type ] [S] [line 1, size 32, align 32, offset 0] [def] [from ]
!10 = !{!11}
!11 = !{!"0xd\00f\001\0032\0032\000\000", !1, !9, !8} ; [ DW_TAG_member ] [f] [line 1, size 32, align 32, offset 0] [from float]
!12 = !{i32 2, !"Dwarf Version", i32 2}
!13 = !{i32 2, !"Debug Info Version", i32 2}
!14 = !{i32 1, !"PIC Level", i32 2}
!15 = !{!"clang version 3.6.0 "}
!16 = !{!"0x101\00s\0016777219\000", !4, !5, !9}  ; [ DW_TAG_arg_variable ] [s] [line 3]
!17 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!18 = !MDLocation(line: 3, column: 20, scope: !4)
!19 = !MDLocation(line: 4, column: 2, scope: !4)
