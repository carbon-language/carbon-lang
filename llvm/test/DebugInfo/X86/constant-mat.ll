; RUN: llc -disable-fp-elim -O0 %s -mtriple x86_64-unknown-linux-gnu -o - | FileCheck %s

; int callme(int);
;
; int isel_line_test(int arg)
; {
;   callme(100);
;   if (arg > 5000)
;     callme(200);
;   callme(300);
;   return 0;
; }

define i32 @isel_line_test(i32 %arg) nounwind uwtable {
; The start of each non-entry block (or sub-block) should get a .loc directive.
; CHECK:       isel_line_test:
; CHECK:       # BB#1:
; CHECK-NEXT:  .loc 1 7 5
; CHECK:       LBB0_2:
; CHECK-NEXT:  .loc 1 8 3
; CHECK:       callq callme
; CHECK-NEXT:  .loc 1 9 3

entry:
  %arg.addr = alloca i32, align 4
  store i32 %arg, i32* %arg.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %arg.addr}, metadata !10), !dbg !11
  %call = call i32 @callme(i32 100), !dbg !12
  %0 = load i32* %arg.addr, align 4, !dbg !14
  %cmp = icmp sgt i32 %0, 5000, !dbg !14
  br i1 %cmp, label %if.then, label %if.end, !dbg !14

if.then:                                          ; preds = %entry
  %call1 = call i32 @callme(i32 200), !dbg !15
  br label %if.end, !dbg !15

if.end:                                           ; preds = %if.then, %entry
  %call2 = call i32 @callme(i32 300), !dbg !16
  ret i32 0, !dbg !17
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare i32 @callme(i32)

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 12, metadata !"foo.c", metadata !"/usr/local/google/home/echristo/tmp", metadata !"clang version 3.2 (trunk 164952) (llvm/trunk 164949)", i1 true, i1 false, metadata !"", i32 0, metadata !1, metadata !1, metadata !3, metadata !1} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/foo.c] [DW_LANG_C99]
!1 = metadata !{metadata !2}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786478, i32 0, metadata !6, metadata !"isel_line_test", metadata !"isel_line_test", metadata !"", metadata !6, i32 3, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @isel_line_test, null, null, metadata !1, i32 4} ; [ DW_TAG_subprogram ] [line 3] [def] [scope 4] [isel_line_test]
!6 = metadata !{i32 786473, metadata !"foo.c", metadata !"/usr/local/google/home/echristo/tmp", null} ; [ DW_TAG_file_type ]
!7 = metadata !{i32 786453, i32 0, metadata !"", i32 0, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9}
!9 = metadata !{i32 786468, null, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786689, metadata !5, metadata !"arg", metadata !6, i32 16777219, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [arg] [line 3]
!11 = metadata !{i32 3, i32 24, metadata !5, null}
!12 = metadata !{i32 5, i32 3, metadata !13, null}
!13 = metadata !{i32 786443, metadata !5, i32 4, i32 1, metadata !6, i32 0} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/foo.c]
!14 = metadata !{i32 6, i32 3, metadata !13, null}
!15 = metadata !{i32 7, i32 5, metadata !13, null}
!16 = metadata !{i32 8, i32 3, metadata !13, null}
!17 = metadata !{i32 9, i32 3, metadata !13, null}
