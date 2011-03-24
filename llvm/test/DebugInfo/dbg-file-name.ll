; RUN: llc -O0 < %s | FileCheck %s
; Radar 8884898
; CHECK: file	1 "/Users/manav/one/two/simple.c"

@.str = private unnamed_addr constant [8 x i8] c"i = %d\0A\00", align 4
@.str1 = private unnamed_addr constant [12 x i8] c"i + 1 = %d\0A\00", align 4

define void @foo(i32 %i) nounwind {
entry:
  %i_addr = alloca i32, align 4
  %"alloca point" = bitcast i32 0 to i32
  call void @llvm.dbg.declare(metadata !{i32* %i_addr}, metadata !9), !dbg !10
  store i32 %i, i32* %i_addr
  %0 = load i32* %i_addr, align 4, !dbg !11
  %1 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([8 x i8]* @.str, i32 0, i32 0), i32 %0) nounwind, !dbg !11
  %2 = load i32* %i_addr, align 4, !dbg !13
  %3 = add nsw i32 %2, 1, !dbg !13
  %4 = call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([12 x i8]* @.str1, i32 0, i32 0), i32 %3) nounwind, !dbg !13
  br label %return, !dbg !14

return:                                           ; preds = %entry
  ret void, !dbg !14
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

declare i32 @printf(i8*, ...) nounwind

define i32 @main() nounwind {
entry:
  %retval = alloca i32
  %0 = alloca i32
  %"alloca point" = bitcast i32 0 to i32
  call void @foo(i32 2) nounwind, !dbg !15
  call void @foo(i32 4) nounwind, !dbg !17
  store i32 0, i32* %0, align 4, !dbg !18
  %1 = load i32* %0, align 4, !dbg !18
  store i32 %1, i32* %retval, align 4, !dbg !18
  br label %return, !dbg !18

return:                                           ; preds = %entry
  %retval1 = load i32* %retval, !dbg !18
  ret i32 %retval1, !dbg !18
}

!llvm.dbg.sp = !{!0, !6}

!0 = metadata !{i32 589870, i32 0, metadata !1, metadata !"foo", metadata !"foo", metadata !"foo", metadata !1, i32 4, metadata !3, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (i32)* @foo} ; [ DW_TAG_subprogram ]
!1 = metadata !{i32 589865, metadata !"simple.c", metadata !"/Users/manav/one/two", metadata !2} ; [ DW_TAG_file_type ]
!2 = metadata !{i32 589841, i32 0, i32 1, metadata !"simple.c", metadata !"/Users/manav/one/two", metadata !"LLVM build 00", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!3 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !4, i32 0, null} ; [ DW_TAG_subroutine_type ]
!4 = metadata !{null, metadata !5}
!5 = metadata !{i32 589860, metadata !1, metadata !"int", metadata !1, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!6 = metadata !{i32 589870, i32 0, metadata !1, metadata !"main", metadata !"main", metadata !"main", metadata !1, i32 9, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main} ; [ DW_TAG_subprogram ]
!7 = metadata !{i32 589845, metadata !1, metadata !"", metadata !1, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{metadata !5}
!9 = metadata !{i32 590081, metadata !0, metadata !"i", metadata !1, i32 4, metadata !5, i32 0} ; [ DW_TAG_arg_variable ]
!10 = metadata !{i32 4, i32 0, metadata !0, null}
!11 = metadata !{i32 5, i32 0, metadata !12, null}
!12 = metadata !{i32 589835, metadata !0, i32 4, i32 0, metadata !1, i32 0} ; [ DW_TAG_lexical_block ]
!13 = metadata !{i32 6, i32 0, metadata !12, null}
!14 = metadata !{i32 7, i32 0, metadata !12, null}
!15 = metadata !{i32 10, i32 0, metadata !16, null}
!16 = metadata !{i32 589835, metadata !6, i32 9, i32 0, metadata !1, i32 1} ; [ DW_TAG_lexical_block ]
!17 = metadata !{i32 11, i32 0, metadata !16, null}
!18 = metadata !{i32 12, i32 0, metadata !16, null}
