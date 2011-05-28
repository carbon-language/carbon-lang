; This test checks debug info of unused, zero extended argument.
; RUN: %clang -arch x86_64 -mllvm -fast-isel=false  %s -c -o %t.o
; RUN: %clang -arch x86_64 %t.o -o %t.out
; RUN: %test_debuginfo %s %t.out
; XFAIL: *
; XTARGET: darwin
; Radar 9422775

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"

%class.aClass = type { float }

; DEBUGGER: break aClass::setValues
; DEBUGGER: r
; DEBUGGER: p Filter
; CHECK: true

define void @_ZN6aClass9setValuesEibf(%class.aClass* nocapture %this, i32 %ch, i1 zeroext %Filter, float %a1) nounwind noinline ssp align 2 {
entry:
  tail call void @llvm.dbg.value(metadata !{%class.aClass* %this}, i64 0, metadata !19), !dbg !25
  tail call void @llvm.dbg.value(metadata !{i32 %ch}, i64 0, metadata !20), !dbg !26
  tail call void @llvm.dbg.value(metadata !{i1 %Filter}, i64 0, metadata !21), !dbg !27
  tail call void @llvm.dbg.value(metadata !{float %a1}, i64 0, metadata !22), !dbg !28
  %m = getelementptr inbounds %class.aClass* %this, i64 0, i32 0, !dbg !29
  store float %a1, float* %m, align 4, !dbg !29, !tbaa !31
  ret void, !dbg !34
}

declare void @llvm.dbg.declare(metadata, metadata) nounwind readnone

define i32 @main() nounwind ssp {
entry:
  %a = alloca %class.aClass, align 4
  call void @llvm.dbg.declare(metadata !{%class.aClass* %a}, metadata !23), !dbg !35
  call void @_ZN6aClass9setValuesEibf(%class.aClass* %a, i32 undef, i1 zeroext 1, float 1.000000e+00), !dbg !36
  ret i32 0, !dbg !37
}

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.dbg.sp = !{!1, !12, !16}
!llvm.dbg.lv._ZN6aClass9setValuesEibf = !{!19, !20, !21, !22}
!llvm.dbg.lv.main = !{!23}

!0 = metadata !{i32 589841, i32 0, i32 4, metadata !"two.cpp", metadata !"/private/tmp/inc", metadata !"clang version 3.0 (trunk 131411)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 589870, i32 0, metadata !2, metadata !"setValues", metadata !"setValues", metadata !"_ZN6aClass9setValuesEibf", metadata !3, i32 6, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 589826, metadata !0, metadata !"aClass", metadata !3, i32 2, i64 32, i64 32, i32 0, i32 0, null, metadata !4, i32 0, null, null} ; [ DW_TAG_class_type ]
!3 = metadata !{i32 589865, metadata !"./one.h", metadata !"/private/tmp/inc", metadata !0} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !5, metadata !1}
!5 = metadata !{i32 589837, metadata !3, metadata !"m", metadata !3, i32 4, i64 32, i64 32, i64 0, i32 1, metadata !6} ; [ DW_TAG_member ]
!6 = metadata !{i32 589860, metadata !0, metadata !"float", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 4} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 589845, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!8 = metadata !{null, metadata !9, metadata !10, metadata !11, metadata !6}
!9 = metadata !{i32 589839, metadata !0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !2} ; [ DW_TAG_pointer_type ]
!10 = metadata !{i32 589860, metadata !0, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!11 = metadata !{i32 589860, metadata !0, metadata !"bool", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ]
!12 = metadata !{i32 589870, i32 0, metadata !13, metadata !"setValues", metadata !"setValues", metadata !"_ZN6aClass9setValuesEibf", metadata !13, i32 4, metadata !14, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, void (%class.aClass*, i32, i1, float)* @_ZN6aClass9setValuesEibf, null, metadata !1} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 589865, metadata !"two.cpp", metadata !"/private/tmp/inc", metadata !0} ; [ DW_TAG_file_type ]
!14 = metadata !{i32 589845, metadata !13, metadata !"", metadata !13, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !15, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!15 = metadata !{null}
!16 = metadata !{i32 589870, i32 0, metadata !13, metadata !"main", metadata !"main", metadata !"", metadata !13, i32 9, metadata !17, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i32 ()* @main, null, null} ; [ DW_TAG_subprogram ]
!17 = metadata !{i32 589845, metadata !13, metadata !"", metadata !13, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !18, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!18 = metadata !{metadata !10}
!19 = metadata !{i32 590081, metadata !12, metadata !"this", metadata !13, i32 16777219, metadata !9, i32 64} ; [ DW_TAG_arg_variable ]
!20 = metadata !{i32 590081, metadata !12, metadata !"ch", metadata !13, i32 33554435, metadata !10, i32 0} ; [ DW_TAG_arg_variable ]
!21 = metadata !{i32 590081, metadata !12, metadata !"Filter", metadata !13, i32 50331651, metadata !11, i32 0} ; [ DW_TAG_arg_variable ]
!22 = metadata !{i32 590081, metadata !12, metadata !"a1", metadata !13, i32 67108867, metadata !6, i32 0} ; [ DW_TAG_arg_variable ]
!23 = metadata !{i32 590080, metadata !24, metadata !"a", metadata !13, i32 10, metadata !2, i32 0} ; [ DW_TAG_auto_variable ]
!24 = metadata !{i32 589835, metadata !16, i32 9, i32 1, metadata !13, i32 1} ; [ DW_TAG_lexical_block ]
!25 = metadata !{i32 3, i32 40, metadata !12, null}
!26 = metadata !{i32 3, i32 54, metadata !12, null}
!27 = metadata !{i32 3, i32 63, metadata !12, null}
!28 = metadata !{i32 3, i32 77, metadata !12, null}
!29 = metadata !{i32 5, i32 2, metadata !30, null}
!30 = metadata !{i32 589835, metadata !12, i32 4, i32 1, metadata !13, i32 0} ; [ DW_TAG_lexical_block ]
!31 = metadata !{metadata !"float", metadata !32}
!32 = metadata !{metadata !"omnipotent char", metadata !33}
!33 = metadata !{metadata !"Simple C/C++ TBAA", null}
!34 = metadata !{i32 6, i32 1, metadata !30, null}
!35 = metadata !{i32 10, i32 11, metadata !24, null}
!36 = metadata !{i32 11, i32 4, metadata !24, null}
!37 = metadata !{i32 12, i32 4, metadata !24, null}
