; RUN: llc < %s | FileCheck %s
; Test to check separate label for inlined function argument.
define void @bar(double %x) nounwind {
entry:
  %__x_addr.i = alloca double                     ; <double*> [#uses=2]
  %retval.i = alloca i32                          ; <i32*> [#uses=2]
  %0 = alloca i32                                 ; <i32*> [#uses=2]
  %x_addr = alloca double                         ; <double*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{double* %x_addr}, metadata !0), !dbg !7
  store double %x, double* %x_addr
  %1 = load double* %x_addr, align 8, !dbg !8     ; <double> [#uses=1]
  call void @llvm.dbg.declare(metadata !{double* %__x_addr.i}, metadata !10), !dbg !15
  store double %1, double* %__x_addr.i
  %2 = load double* %__x_addr.i, align 8, !dbg !15 ; <double> [#uses=1]
  %3 = fptosi double %2 to i32, !dbg !15          ; <i32> [#uses=1]
  store i32 %3, i32* %0, align 4, !dbg !15
  %4 = load i32* %0, align 4, !dbg !15            ; <i32> [#uses=1]
  store i32 %4, i32* %retval.i, align 4, !dbg !15
  %retval1.i = load i32* %retval.i, !dbg !15      ; <i32> [#uses=0]
  br label %return, !dbg !16

return:                                           ; preds = %entry
  ret void, !dbg !16
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

!0 = metadata !{i32 524545, metadata !1, metadata !"x", metadata !2, i32 7, metadata !6} ; [ DW_TAG_arg_variable ]
!1 = metadata !{i32 524334, i32 0, metadata !2, metadata !"bar", metadata !"bar", metadata !"bar", metadata !2, i32 7, metadata !4, i1 false, i1 true, i32 0, i32 0, null, i1 false, i1 false} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 524329, metadata !"2010-01-18-Inlined-Debug.c", metadata !"/Users/buildslave/zorg/buildbot/smooshlab/slave/build.llvm-gcc-powerpc-darwin9/llvm.src/test/FrontendC", metadata !3} ; [ DW_TAG_file_type ]
!3 = metadata !{i32 524305, i32 0, i32 1, metadata !"2010-01-18-Inlined-Debug.c", metadata !"/Users/buildslave/zorg/buildbot/smooshlab/slave/build.llvm-gcc-powerpc-darwin9/llvm.src/test/FrontendC", metadata !"4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", i1 true, i1 false, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!4 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !5, i32 0, null} ; [ DW_TAG_subroutine_type ]
!5 = metadata !{null, metadata !6}
!6 = metadata !{i32 524324, metadata !2, metadata !"double", metadata !2, i32 0, i64 64, i64 64, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 7, i32 0, metadata !1, null}
!8 = metadata !{i32 8, i32 0, metadata !9, null}
!9 = metadata !{i32 524299, metadata !1, i32 7, i32 0} ; [ DW_TAG_lexical_block ]
!10 = metadata !{i32 524545, metadata !11, metadata !"__x", metadata !2, i32 5, metadata !6} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 524334, i32 0, metadata !2, metadata !"foo", metadata !"foo", metadata !"foo", metadata !2, i32 5, metadata !12, i1 true, i1 true, i32 0, i32 0, null, i1 false, i1 false} ; [ DW_TAG_subprogram ]
!12 = metadata !{i32 524309, metadata !2, metadata !"", metadata !2, i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !13, i32 0, null} ; [ DW_TAG_subroutine_type ]
!13 = metadata !{metadata !14, metadata !6}
!14 = metadata !{i32 524324, metadata !2, metadata !"int", metadata !2, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!15 = metadata !{i32 5, i32 0, metadata !11, metadata !8}
!16 = metadata !{i32 9, i32 0, metadata !9, null}

;CHECK:	        ##DEBUG_VALUE: bar:x 
;CHECK-NEXT:Ltmp
;CHECK-NEXT	##DEBUG_VALUE: foo:__x 
