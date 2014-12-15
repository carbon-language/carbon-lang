; RUN: llc < %s - | FileCheck %s
; Radar 9309221
; Test dwarf reg no for s16
;CHECK: super-register
;CHECK-NEXT: DW_OP_regx
;CHECK-NEXT: ascii
;CHECK-NEXT: DW_OP_piece
;CHECK-NEXT: 4

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

@.str = private unnamed_addr constant [11 x i8] c"%p %lf %c\0A\00"
@.str1 = private unnamed_addr constant [6 x i8] c"point\00"

define i32 @inlineprinter(i8* %ptr, float %val, i8 zeroext %c) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata i8* %ptr, i64 0, metadata !8, metadata !{!"0x102"}), !dbg !24
  tail call void @llvm.dbg.value(metadata float %val, i64 0, metadata !10, metadata !{!"0x102"}), !dbg !25
  tail call void @llvm.dbg.value(metadata i8 %c, i64 0, metadata !12, metadata !{!"0x102"}), !dbg !26
  %conv = fpext float %val to double, !dbg !27
  %conv3 = zext i8 %c to i32, !dbg !27
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %conv, i32 %conv3) nounwind optsize, !dbg !27
  ret i32 0, !dbg !29
}

declare i32 @printf(i8* nocapture, ...) nounwind optsize

define i32 @printer(i8* %ptr, float %val, i8 zeroext %c) nounwind optsize noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata i8* %ptr, i64 0, metadata !14, metadata !{!"0x102"}), !dbg !30
  tail call void @llvm.dbg.value(metadata float %val, i64 0, metadata !15, metadata !{!"0x102"}), !dbg !31
  tail call void @llvm.dbg.value(metadata i8 %c, i64 0, metadata !16, metadata !{!"0x102"}), !dbg !32
  %conv = fpext float %val to double, !dbg !33
  %conv3 = zext i8 %c to i32, !dbg !33
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %conv, i32 %conv3) nounwind optsize, !dbg !33
  ret i32 0, !dbg !35
}

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata i32 %argc, i64 0, metadata !17, metadata !{!"0x102"}), !dbg !36
  tail call void @llvm.dbg.value(metadata i8** %argv, i64 0, metadata !18, metadata !{!"0x102"}), !dbg !37
  %conv = sitofp i32 %argc to double, !dbg !38
  %add = fadd double %conv, 5.555552e+05, !dbg !38
  %conv1 = fptrunc double %add to float, !dbg !38
  tail call void @llvm.dbg.value(metadata float %conv1, i64 0, metadata !22, metadata !{!"0x102"}), !dbg !38
  %call = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @.str1, i32 0, i32 0)) nounwind optsize, !dbg !39
  %add.ptr = getelementptr i8* bitcast (i32 (i32, i8**)* @main to i8*), i32 %argc, !dbg !40
  %add5 = add nsw i32 %argc, 97, !dbg !40
  %conv6 = trunc i32 %add5 to i8, !dbg !40
  tail call void @llvm.dbg.value(metadata i8* %add.ptr, i64 0, metadata !8, metadata !{!"0x102"}) nounwind, !dbg !41
  tail call void @llvm.dbg.value(metadata float %conv1, i64 0, metadata !10, metadata !{!"0x102"}) nounwind, !dbg !42
  tail call void @llvm.dbg.value(metadata i8 %conv6, i64 0, metadata !12, metadata !{!"0x102"}) nounwind, !dbg !43
  %conv.i = fpext float %conv1 to double, !dbg !44
  %conv3.i = and i32 %add5, 255, !dbg !44
  %call.i = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %add.ptr, double %conv.i, i32 %conv3.i) nounwind optsize, !dbg !44
  %call14 = tail call i32 @printer(i8* %add.ptr, float %conv1, i8 zeroext %conv6) optsize, !dbg !45
  ret i32 0, !dbg !46
}

declare i32 @puts(i8* nocapture) nounwind optsize

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!53}

!0 = !{!"0x2e\00inlineprinter\00inlineprinter\00\005\000\001\000\006\00256\001\005", !51, !1, !3, null, i32 (i8*, float, i8)* @inlineprinter, null, null, !48} ; [ DW_TAG_subprogram ] [line 5] [def] [inlineprinter]
!1 = !{!"0x29", !51} ; [ DW_TAG_file_type ]
!2 = !{!"0x11\0012\00clang version 3.0 (trunk 129915)\001\00\000\00\001", !51, !52, !52, !47, null,  null} ; [ DW_TAG_compile_unit ]
!3 = !{!"0x15\00\000\000\000\000\000\000", !51, !1, null, !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = !{!5}
!5 = !{!"0x24\00int\000\0032\0032\000\000\005", null, !2} ; [ DW_TAG_base_type ]
!6 = !{!"0x2e\00printer\00printer\00\0012\000\001\000\006\00256\001\0012", !51, !1, !3, null, i32 (i8*, float, i8)* @printer, null, null, !49} ; [ DW_TAG_subprogram ] [line 12] [def] [printer]
!7 = !{!"0x2e\00main\00main\00\0018\000\001\000\006\00256\001\0018", !51, !1, !3, null, i32 (i32, i8**)* @main, null, null, !50} ; [ DW_TAG_subprogram ] [line 18] [def] [main]
!8 = !{!"0x101\00ptr\0016777220\000", !0, !1, !9} ; [ DW_TAG_arg_variable ]
!9 = !{!"0xf\00\000\0032\0032\000\000", null, !2, null} ; [ DW_TAG_pointer_type ]
!10 = !{!"0x101\00val\0033554436\000", !0, !1, !11} ; [ DW_TAG_arg_variable ]
!11 = !{!"0x24\00float\000\0032\0032\000\000\004", null, !2} ; [ DW_TAG_base_type ]
!12 = !{!"0x101\00c\0050331652\000", !0, !1, !13} ; [ DW_TAG_arg_variable ]
!13 = !{!"0x24\00unsigned char\000\008\008\000\000\008", null, !2} ; [ DW_TAG_base_type ]
!14 = !{!"0x101\00ptr\0016777227\000", !6, !1, !9} ; [ DW_TAG_arg_variable ]
!15 = !{!"0x101\00val\0033554443\000", !6, !1, !11} ; [ DW_TAG_arg_variable ]
!16 = !{!"0x101\00c\0050331659\000", !6, !1, !13} ; [ DW_TAG_arg_variable ]
!17 = !{!"0x101\00argc\0016777233\000", !7, !1, !5} ; [ DW_TAG_arg_variable ]
!18 = !{!"0x101\00argv\0033554449\000", !7, !1, !19} ; [ DW_TAG_arg_variable ]
!19 = !{!"0xf\00\000\0032\0032\000\000", null, !2, !20} ; [ DW_TAG_pointer_type ]
!20 = !{!"0xf\00\000\0032\0032\000\000", null, !2, !21} ; [ DW_TAG_pointer_type ]
!21 = !{!"0x24\00char\000\008\008\000\000\006", null, !2} ; [ DW_TAG_base_type ]
!22 = !{!"0x100\00dval\0019\000", !23, !1, !11} ; [ DW_TAG_auto_variable ]
!23 = !{!"0xb\0018\001\002", !51, !7} ; [ DW_TAG_lexical_block ]
!24 = !{i32 4, i32 22, !0, null}
!25 = !{i32 4, i32 33, !0, null}
!26 = !{i32 4, i32 52, !0, null}
!27 = !{i32 6, i32 3, !28, null}
!28 = !{!"0xb\005\001\000", !51, !0} ; [ DW_TAG_lexical_block ]
!29 = !{i32 7, i32 3, !28, null}
!30 = !{i32 11, i32 42, !6, null}
!31 = !{i32 11, i32 53, !6, null}
!32 = !{i32 11, i32 72, !6, null}
!33 = !{i32 13, i32 3, !34, null}
!34 = !{!"0xb\0012\001\001", !51, !6} ; [ DW_TAG_lexical_block ]
!35 = !{i32 14, i32 3, !34, null}
!36 = !{i32 17, i32 15, !7, null}
!37 = !{i32 17, i32 28, !7, null}
!38 = !{i32 19, i32 31, !23, null}
!39 = !{i32 20, i32 3, !23, null}
!40 = !{i32 21, i32 3, !23, null}
!41 = !{i32 4, i32 22, !0, !40}
!42 = !{i32 4, i32 33, !0, !40}
!43 = !{i32 4, i32 52, !0, !40}
!44 = !{i32 6, i32 3, !28, !40}
!45 = !{i32 22, i32 3, !23, null}
!46 = !{i32 23, i32 1, !23, null}
!47 = !{!0, !6, !7}
!48 = !{!8, !10, !12}
!49 = !{!14, !15, !16}
!50 = !{!17, !18, !22}
!51 = !{!"a.c", !"/private/tmp"}
!52 = !{i32 0}
!53 = !{i32 1, !"Debug Info Version", i32 2}
