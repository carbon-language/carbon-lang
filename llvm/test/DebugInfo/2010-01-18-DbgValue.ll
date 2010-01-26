; RUN: llc -O0 < %s | FileCheck %s
; ModuleID = 'try.c'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin9.8"
; Currently, dbg.declare generates a DEBUG_VALUE comment.  Eventually it will
; generate DWARF and this test will need to be modified or removed.

@Y = common global i32 0                          ; <i32*> [#uses=1]

define i32 @test() nounwind {
entry:
; CHECK: DEBUG_VALUE:
  %retval = alloca i32                            ; <i32*> [#uses=2]
  %X = alloca i32                                 ; <i32*> [#uses=5]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{i32* %X}, metadata !3), !dbg !7
  store i32 4, i32* %X, align 4, !dbg !8
  %1 = load i32* %X, align 4, !dbg !9             ; <i32> [#uses=1]
  call void @use(i32 %1) nounwind, !dbg !9
  %2 = load i32* @Y, align 4, !dbg !10            ; <i32> [#uses=1]
  %3 = add nsw i32 %2, 2, !dbg !10                ; <i32> [#uses=1]
  store i32 %3, i32* %X, align 4, !dbg !10
  %4 = load i32* %X, align 4, !dbg !11            ; <i32> [#uses=1]
  call void @use(i32 %4) nounwind, !dbg !11
  %5 = load i32* %X, align 4, !dbg !12            ; <i32> [#uses=1]
  store i32 %5, i32* %0, align 4, !dbg !12
  %6 = load i32* %0, align 4, !dbg !12            ; <i32> [#uses=1]
  store i32 %6, i32* %retval, align 4, !dbg !12
  br label %return, !dbg !12

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !12          ; <i32> [#uses=1]
  ret i32 %retval1, !dbg !12
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare void @use(i32)

!llvm.dbg.gv = !{!0}

!0 = metadata !{i32 458804, i32 0, metadata !1, metadata !"Y", metadata !"Y", metadata !"Y", metadata !1, i32 2, metadata !2, i1 false, i1 true, i32* @Y} ; [ DW_TAG_variable ]
!1 = metadata !{i32 458769, i32 0, i32 1, metadata !"try.c", metadata !"/Volumes/MacOS9/tests/", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!2 = metadata !{i32 458788, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!3 = metadata !{i32 459008, metadata !4, metadata !"X", metadata !1, i32 4, metadata !2} ; [ DW_TAG_auto_variable ]
!4 = metadata !{i32 458798, i32 0, metadata !1, metadata !"", metadata !"", metadata !"test", metadata !1, i32 3, metadata !5, i1 false, i1 true, i32 0, i32 0, null} ; [ DW_TAG_subprogram ]
!5 = metadata !{i32 458773, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !6, i32 0} ; [ DW_TAG_subroutine_type ]
!6 = metadata !{metadata !2}
!7 = metadata !{i32 3, i32 0, metadata !4, null}
!8 = metadata !{i32 4, i32 0, metadata !4, null}
!9 = metadata !{i32 5, i32 0, metadata !4, null}
!10 = metadata !{i32 6, i32 0, metadata !4, null}
!11 = metadata !{i32 7, i32 0, metadata !4, null}
!12 = metadata !{i32 8, i32 0, metadata !4, null}
