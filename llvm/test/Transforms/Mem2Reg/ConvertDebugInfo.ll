; RUN: opt < %s -mem2reg -S | FileCheck %s

define double @testfunc(i32 %i, double %j) nounwind ssp {
entry:
  %i_addr = alloca i32                            ; <i32*> [#uses=2]
  %j_addr = alloca double                         ; <double*> [#uses=2]
  %retval = alloca double                         ; <double*> [#uses=2]
  %0 = alloca double                              ; <double*> [#uses=2]
  %"alloca point" = bitcast i32 0 to i32          ; <i32> [#uses=0]
  call void @llvm.dbg.declare(metadata !{i32* %i_addr}, metadata !0, metadata !{}), !dbg !8
; CHECK: call void @llvm.dbg.value(metadata !{i32 %i}, i64 0, metadata ![[IVAR:[0-9]*]], metadata {{.*}})
; CHECK: call void @llvm.dbg.value(metadata !{double %j}, i64 0, metadata ![[JVAR:[0-9]*]], metadata {{.*}})
; CHECK: ![[IVAR]] = {{.*}} ; [ DW_TAG_arg_variable ] [i]
; CHECK: ![[JVAR]] = {{.*}} ; [ DW_TAG_arg_variable ] [j]
  store i32 %i, i32* %i_addr
  call void @llvm.dbg.declare(metadata !{double* %j_addr}, metadata !9, metadata !{}), !dbg !8
  store double %j, double* %j_addr
  %1 = load i32* %i_addr, align 4, !dbg !10       ; <i32> [#uses=1]
  %2 = add nsw i32 %1, 1, !dbg !10                ; <i32> [#uses=1]
  %3 = sitofp i32 %2 to double, !dbg !10          ; <double> [#uses=1]
  %4 = load double* %j_addr, align 8, !dbg !10    ; <double> [#uses=1]
  %5 = fadd double %3, %4, !dbg !10               ; <double> [#uses=1]
  store double %5, double* %0, align 8, !dbg !10
  %6 = load double* %0, align 8, !dbg !10         ; <double> [#uses=1]
  store double %6, double* %retval, align 8, !dbg !10
  br label %return, !dbg !10

return:                                           ; preds = %entry
  %retval1 = load double* %retval, !dbg !10       ; <double> [#uses=1]
  ret double %retval1, !dbg !10
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!14}

!0 = metadata !{metadata !"0x101\00i\002\000", metadata !1, metadata !2, metadata !7} ; [ DW_TAG_arg_variable ]
!1 = metadata !{metadata !"0x2e\00testfunc\00testfunc\00testfunc\002\000\001\000\006\000\000\002", metadata !12, metadata !2, metadata !4, null, double (i32, double)* @testfunc, null, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{metadata !"0x29", metadata !12} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\000", metadata !12, metadata !13, metadata !13, null, null, null} ; [ DW_TAG_compile_unit ]
!4 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !12, metadata !2, null, metadata !5, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!5 = metadata !{metadata !6, metadata !7, metadata !6}
!6 = metadata !{metadata !"0x24\00double\000\0064\0064\000\000\004", metadata !12, metadata !2} ; [ DW_TAG_base_type ]
!7 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !12, metadata !2} ; [ DW_TAG_base_type ]
!8 = metadata !{i32 2, i32 0, metadata !1, null}
!9 = metadata !{metadata !"0x101\00j\002\000", metadata !1, metadata !2, metadata !6} ; [ DW_TAG_arg_variable ]
!10 = metadata !{i32 3, i32 0, metadata !11, null}
!11 = metadata !{metadata !"0xb\002\000\000", metadata !12, metadata !1} ; [ DW_TAG_lexical_block ]
!12 = metadata !{metadata !"testfunc.c", metadata !"/tmp"}
!13 = metadata !{i32 0}
!14 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
