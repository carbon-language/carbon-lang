; RUN: llc < %s | FileCheck %s
; Radar 7881628, 9747970
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.7.0"

%class.APFloat = type { i32 }

define i32 @_ZNK7APFloat9partCountEv(%class.APFloat* nocapture %this) nounwind uwtable readonly optsize ssp align 2 {
entry:
  tail call void @llvm.dbg.value(metadata !{%class.APFloat* %this}, i64 0, metadata !28), !dbg !41
  %prec = getelementptr inbounds %class.APFloat* %this, i64 0, i32 0, !dbg !42
  %tmp = load i32* %prec, align 4, !dbg !42, !tbaa !44
  tail call void @llvm.dbg.value(metadata !{i32 %tmp}, i64 0, metadata !47), !dbg !48
  %add.i = add i32 %tmp, 42, !dbg !49
  ret i32 %add.i, !dbg !42
}

define zeroext i1 @_ZNK7APFloat14bitwiseIsEqualERKS_(%class.APFloat* %this, %class.APFloat* %rhs) uwtable optsize ssp align 2 {
entry:
  tail call void @llvm.dbg.value(metadata !{%class.APFloat* %this}, i64 0, metadata !29), !dbg !51
  tail call void @llvm.dbg.value(metadata !{%class.APFloat* %rhs}, i64 0, metadata !30), !dbg !52
  tail call void @llvm.dbg.value(metadata !{%class.APFloat* %this}, i64 0, metadata !53), !dbg !55
  %prec.i = getelementptr inbounds %class.APFloat* %this, i64 0, i32 0, !dbg !56
;CHECK: DW_TAG_inlined_subroutine
;CHECK: DW_AT_abstract_origin
;CHECK: DW_AT_ranges
  %tmp.i = load i32* %prec.i, align 4, !dbg !56, !tbaa !44
  tail call void @llvm.dbg.value(metadata !{i32 %tmp.i}, i64 0, metadata !57), !dbg !58
  %add.i.i = add i32 %tmp.i, 42, !dbg !59
  tail call void @llvm.dbg.value(metadata !{i32 %add.i.i}, i64 0, metadata !31), !dbg !54
  %call2 = tail call i64* @_ZNK7APFloat16significandPartsEv(%class.APFloat* %this) optsize, !dbg !60
  tail call void @llvm.dbg.value(metadata !{i64* %call2}, i64 0, metadata !34), !dbg !60
  %call3 = tail call i64* @_ZNK7APFloat16significandPartsEv(%class.APFloat* %rhs) optsize, !dbg !61
  tail call void @llvm.dbg.value(metadata !{i64* %call3}, i64 0, metadata !37), !dbg !61
  %tmp = zext i32 %add.i.i to i64
  br label %for.cond, !dbg !62

for.cond:                                         ; preds = %for.inc, %entry
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %entry ]
  %tmp13 = sub i64 %tmp, %indvar, !dbg !62
  %i.0 = trunc i64 %tmp13 to i32, !dbg !62
  %cmp = icmp sgt i32 %i.0, 0, !dbg !62
  br i1 %cmp, label %for.body, label %return, !dbg !62

for.body:                                         ; preds = %for.cond
  %p.0 = getelementptr i64* %call2, i64 %indvar, !dbg !63
  %tmp6 = load i64* %p.0, align 8, !dbg !63, !tbaa !66
  %tmp8 = load i64* %call3, align 8, !dbg !63, !tbaa !66
  %cmp9 = icmp eq i64 %tmp6, %tmp8, !dbg !63
  br i1 %cmp9, label %for.inc, label %return, !dbg !63

for.inc:                                          ; preds = %for.body
  %indvar.next = add i64 %indvar, 1, !dbg !67
  br label %for.cond, !dbg !67

return:                                           ; preds = %for.cond, %for.body
  %retval.0 = phi i1 [ false, %for.body ], [ true, %for.cond ]
  ret i1 %retval.0, !dbg !68
}

declare i64* @_ZNK7APFloat16significandPartsEv(%class.APFloat*) optsize

declare void @llvm.dbg.value(metadata, i64, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.dbg.sp = !{!1, !7, !12, !23, !24, !25}
!llvm.dbg.lv._ZNK7APFloat9partCountEv = !{!28}
!llvm.dbg.lv._ZNK7APFloat14bitwiseIsEqualERKS_ = !{!29, !30, !31, !34, !37}
!llvm.dbg.lv._ZL16partCountForBitsj = !{!38}
!llvm.dbg.gv = !{!39}

!0 = metadata !{i32 655377, i32 0, i32 4, metadata !"/Volumes/Athwagate/R9747970/apf.cc", metadata !"/private/tmp", metadata !"clang version 3.0 (trunk 136149)", i1 true, i1 true, metadata !"", i32 0} ; [ DW_TAG_compile_unit ]
!1 = metadata !{i32 655406, i32 0, metadata !2, metadata !"bitwiseIsEqual", metadata !"bitwiseIsEqual", metadata !"_ZNK7APFloat14bitwiseIsEqualERKS_", metadata !3, i32 8, metadata !19, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null} ; [ DW_TAG_subprogram ]
!2 = metadata !{i32 655362, metadata !0, metadata !"APFloat", metadata !3, i32 6, i64 32, i64 32, i32 0, i32 0, null, metadata !4, i32 0, null, null} ; [ DW_TAG_class_type ]
!3 = metadata !{i32 655401, metadata !"/Volumes/Athwagate/R9747970/apf.cc", metadata !"/private/tmp", metadata !0} ; [ DW_TAG_file_type ]
!4 = metadata !{metadata !5, metadata !1, metadata !7, metadata !12}
!5 = metadata !{i32 655373, metadata !2, metadata !"prec", metadata !3, i32 13, i64 32, i64 32, i64 0, i32 0, metadata !6} ; [ DW_TAG_member ]
!6 = metadata !{i32 655396, metadata !0, metadata !"unsigned int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!7 = metadata !{i32 655406, i32 0, metadata !2, metadata !"partCount", metadata !"partCount", metadata !"_ZNK7APFloat9partCountEv", metadata !3, i32 9, metadata !8, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null} ; [ DW_TAG_subprogram ]
!8 = metadata !{i32 655381, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !9, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!9 = metadata !{metadata !6, metadata !10}
!10 = metadata !{i32 655375, metadata !0, metadata !"", i32 0, i32 0, i64 64, i64 64, i64 0, i32 64, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{i32 655398, metadata !0, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !2} ; [ DW_TAG_const_type ]
!12 = metadata !{i32 655406, i32 0, metadata !2, metadata !"significandParts", metadata !"significandParts", metadata !"_ZNK7APFloat16significandPartsEv", metadata !3, i32 11, metadata !13, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null} ; [ DW_TAG_subprogram ]
!13 = metadata !{i32 655381, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !14, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!14 = metadata !{metadata !15, metadata !10}
!15 = metadata !{i32 655375, metadata !0, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !16} ; [ DW_TAG_pointer_type ]
!16 = metadata !{i32 655382, metadata !0, metadata !"integerPart", metadata !3, i32 2, i64 0, i64 0, i64 0, i32 0, metadata !17} ; [ DW_TAG_typedef ]
!17 = metadata !{i32 655382, metadata !0, metadata !"uint64_t", metadata !3, i32 1, i64 0, i64 0, i64 0, i32 0, metadata !18} ; [ DW_TAG_typedef ]
!18 = metadata !{i32 655396, metadata !0, metadata !"long long unsigned int", null, i32 0, i64 64, i64 64, i64 0, i32 0, i32 7} ; [ DW_TAG_base_type ]
!19 = metadata !{i32 655381, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !20, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!20 = metadata !{metadata !21, metadata !10, metadata !22}
!21 = metadata !{i32 655396, metadata !0, metadata !"bool", null, i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ]
!22 = metadata !{i32 655376, metadata !0, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !11} ; [ DW_TAG_reference_type ]
!23 = metadata !{i32 655406, i32 0, metadata !0, metadata !"partCount", metadata !"partCount", metadata !"_ZNK7APFloat9partCountEv", metadata !3, i32 23, metadata !8, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i32 (%class.APFloat*)* @_ZNK7APFloat9partCountEv, null, metadata !7} ; [ DW_TAG_subprogram ]
!24 = metadata !{i32 655406, i32 0, metadata !0, metadata !"bitwiseIsEqual", metadata !"bitwiseIsEqual", metadata !"_ZNK7APFloat14bitwiseIsEqualERKS_", metadata !3, i32 28, metadata !19, i1 false, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, i1 (%class.APFloat*, %class.APFloat*)* @_ZNK7APFloat14bitwiseIsEqualERKS_, null, metadata !1} ; [ DW_TAG_subprogram ]
!25 = metadata !{i32 655406, i32 0, metadata !3, metadata !"partCountForBits", metadata !"partCountForBits", metadata !"", metadata !3, i32 17, metadata !26, i1 true, i1 true, i32 0, i32 0, i32 0, i32 256, i1 true, null, null, null} ; [ DW_TAG_subprogram ]
!26 = metadata !{i32 655381, metadata !3, metadata !"", metadata !3, i32 0, i64 0, i64 0, i32 0, i32 0, i32 0, metadata !27, i32 0, i32 0} ; [ DW_TAG_subroutine_type ]
!27 = metadata !{metadata !6}
!28 = metadata !{i32 655617, metadata !23, metadata !"this", metadata !3, i32 16777238, metadata !10, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!29 = metadata !{i32 655617, metadata !24, metadata !"this", metadata !3, i32 16777244, metadata !10, i32 64, i32 0} ; [ DW_TAG_arg_variable ]
!30 = metadata !{i32 655617, metadata !24, metadata !"rhs", metadata !3, i32 33554460, metadata !22, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!31 = metadata !{i32 655616, metadata !32, metadata !"i", metadata !3, i32 29, metadata !33, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!32 = metadata !{i32 655371, metadata !24, i32 28, i32 56, metadata !3, i32 1} ; [ DW_TAG_lexical_block ]
!33 = metadata !{i32 655396, metadata !0, metadata !"int", null, i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ]
!34 = metadata !{i32 655616, metadata !32, metadata !"p", metadata !3, i32 30, metadata !35, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!35 = metadata !{i32 655375, metadata !0, metadata !"", null, i32 0, i64 64, i64 64, i64 0, i32 0, metadata !36} ; [ DW_TAG_pointer_type ]
!36 = metadata !{i32 655398, metadata !0, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !16} ; [ DW_TAG_const_type ]
!37 = metadata !{i32 655616, metadata !32, metadata !"q", metadata !3, i32 31, metadata !35, i32 0, i32 0} ; [ DW_TAG_auto_variable ]
!38 = metadata !{i32 655617, metadata !25, metadata !"bits", metadata !3, i32 16777232, metadata !6, i32 0, i32 0} ; [ DW_TAG_arg_variable ]
!39 = metadata !{i32 655412, i32 0, metadata !3, metadata !"integerPartWidth", metadata !"integerPartWidth", metadata !"integerPartWidth", metadata !3, i32 3, metadata !40, i32 1, i32 1, i32 42} ; [ DW_TAG_variable ]
!40 = metadata !{i32 655398, metadata !0, metadata !"", null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !6} ; [ DW_TAG_const_type ]
!41 = metadata !{i32 22, i32 23, metadata !23, null}
!42 = metadata !{i32 24, i32 10, metadata !43, null}
!43 = metadata !{i32 655371, metadata !23, i32 23, i32 1, metadata !3, i32 0} ; [ DW_TAG_lexical_block ]
!44 = metadata !{metadata !"int", metadata !45}
!45 = metadata !{metadata !"omnipotent char", metadata !46}
!46 = metadata !{metadata !"Simple C/C++ TBAA", null}
!47 = metadata !{i32 655617, metadata !25, metadata !"bits", metadata !3, i32 16777232, metadata !6, i32 0, metadata !42} ; [ DW_TAG_arg_variable ]
!48 = metadata !{i32 16, i32 58, metadata !25, metadata !42}
!49 = metadata !{i32 18, i32 3, metadata !50, metadata !42}
!50 = metadata !{i32 655371, metadata !25, i32 17, i32 1, metadata !3, i32 4} ; [ DW_TAG_lexical_block ]
!51 = metadata !{i32 28, i32 15, metadata !24, null}
!52 = metadata !{i32 28, i32 45, metadata !24, null}
!53 = metadata !{i32 655617, metadata !23, metadata !"this", metadata !3, i32 16777238, metadata !10, i32 64, metadata !54} ; [ DW_TAG_arg_variable ]
!54 = metadata !{i32 29, i32 10, metadata !32, null}
!55 = metadata !{i32 22, i32 23, metadata !23, metadata !54}
!56 = metadata !{i32 24, i32 10, metadata !43, metadata !54}
!57 = metadata !{i32 655617, metadata !25, metadata !"bits", metadata !3, i32 16777232, metadata !6, i32 0, metadata !56} ; [ DW_TAG_arg_variable ]
!58 = metadata !{i32 16, i32 58, metadata !25, metadata !56}
!59 = metadata !{i32 18, i32 3, metadata !50, metadata !56}
!60 = metadata !{i32 30, i32 24, metadata !32, null}
!61 = metadata !{i32 31, i32 24, metadata !32, null}
!62 = metadata !{i32 32, i32 3, metadata !32, null}
!63 = metadata !{i32 33, i32 5, metadata !64, null}
!64 = metadata !{i32 655371, metadata !65, i32 32, i32 25, metadata !3, i32 3} ; [ DW_TAG_lexical_block ]
!65 = metadata !{i32 655371, metadata !32, i32 32, i32 3, metadata !3, i32 2} ; [ DW_TAG_lexical_block ]
!66 = metadata !{metadata !"long long", metadata !45}
!67 = metadata !{i32 32, i32 15, metadata !65, null}
!68 = metadata !{i32 37, i32 1, metadata !32, null}
