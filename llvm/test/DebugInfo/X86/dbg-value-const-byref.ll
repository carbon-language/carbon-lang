; RUN: llc -O1 -filetype=obj -o - %s | llvm-dwarfdump -debug-dump=all - | FileCheck %s
; Generated with -O1 from:
; int f1();
; void f2(int*);
; int f3(int);
;
; int foo() {
;   int i = 3;
;   f3(i);
;   i = 7;
;   i = f1();
;   f2(&i);
;   return 0;
; }
;
; Test that we generate valid debug info for optimized code,
; particularly variables that are described as constants and passed
; by reference.
; rdar://problem/14874886
;
; CHECK: .debug_info contents:
; CHECK: DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_location [DW_FORM_data4]	([[LOC:.*]])
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"i"
; CHECK: .debug_loc contents:
; CHECK: [[LOC]]:
;        consts 0x00000003
; CHECK: Beginning address offset: 0x0000000000000{{.*}}
; CHECK:    Ending address offset: [[C1:.*]]
; CHECK:     Location description: 11 03
;        consts 0x00000007
; CHECK: Beginning address offset: [[C1]]
; CHECK:    Ending address offset: [[C2:.*]]
; CHECK:     Location description: 11 07
;        rax, piece 0x00000004
; CHECK: Beginning address offset: [[C2]]
; CHECK:    Ending address offset: [[R1:.*]]
; CHECK:     Location description: 50 93 04
;         rdi+0
; CHECK: Beginning address offset: [[R1]]
; CHECK:    Ending address offset: [[R2:.*]]
; CHECK:     Location description: 75 00
;
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define i32 @foo() #0 {
entry:
  %i = alloca i32, align 4
  call void @llvm.dbg.value(metadata !14, i64 0, metadata !10, metadata !{metadata !"0x102"}), !dbg !15
  %call = call i32 @f3(i32 3) #3, !dbg !16
  call void @llvm.dbg.value(metadata !17, i64 0, metadata !10, metadata !{metadata !"0x102"}), !dbg !18
  %call1 = call i32 (...)* @f1() #3, !dbg !19
  call void @llvm.dbg.value(metadata !{i32 %call1}, i64 0, metadata !10, metadata !{metadata !"0x102"}), !dbg !19
  store i32 %call1, i32* %i, align 4, !dbg !19, !tbaa !20
  call void @llvm.dbg.value(metadata !{i32* %i}, i64 0, metadata !10, metadata !{metadata !"0x102"}), !dbg !24
  call void @f2(i32* %i) #3, !dbg !24
  ret i32 0, !dbg !25
}

declare i32 @f3(i32)

declare i32 @f1(...)

declare void @f2(i32*)

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { nounwind ssp uwtable }
attributes #2 = { nounwind readnone }
attributes #3 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!11, !12}
!llvm.ident = !{!13}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5.0 \001\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [dbg-value-const-byref.c] [DW_LANG_C99]
!1 = metadata !{metadata !"dbg-value-const-byref.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00\005\000\001\000\006\000\001\005", metadata !1, metadata !5, metadata !6, null, i32 ()* @foo, null, null, metadata !9} ; [ DW_TAG_subprogram ] [line 5] [def] [foo]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [dbg-value-const-byref.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !"0x100\00i\006\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_auto_variable ] [i] [line 6]
!11 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!12 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!13 = metadata !{metadata !"clang version 3.5.0 "}
!14 = metadata !{i32 3}
!15 = metadata !{i32 6, i32 0, metadata !4, null}
!16 = metadata !{i32 7, i32 0, metadata !4, null}
!17 = metadata !{i32 7}
!18 = metadata !{i32 8, i32 0, metadata !4, null}
!19 = metadata !{i32 9, i32 0, metadata !4, null}
!20 = metadata !{metadata !21, metadata !21, i64 0}
!21 = metadata !{metadata !"int", metadata !22, i64 0}
!22 = metadata !{metadata !"omnipotent char", metadata !23, i64 0}
!23 = metadata !{metadata !"Simple C/C++ TBAA"}
!24 = metadata !{i32 10, i32 0, metadata !4, null}
!25 = metadata !{i32 11, i32 0, metadata !4, null}
