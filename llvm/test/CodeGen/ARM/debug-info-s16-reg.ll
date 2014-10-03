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
  tail call void @llvm.dbg.value(metadata !{i8* %ptr}, i64 0, metadata !8, metadata !{metadata !"0x102"}), !dbg !24
  tail call void @llvm.dbg.value(metadata !{float %val}, i64 0, metadata !10, metadata !{metadata !"0x102"}), !dbg !25
  tail call void @llvm.dbg.value(metadata !{i8 %c}, i64 0, metadata !12, metadata !{metadata !"0x102"}), !dbg !26
  %conv = fpext float %val to double, !dbg !27
  %conv3 = zext i8 %c to i32, !dbg !27
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %conv, i32 %conv3) nounwind optsize, !dbg !27
  ret i32 0, !dbg !29
}

declare i32 @printf(i8* nocapture, ...) nounwind optsize

define i32 @printer(i8* %ptr, float %val, i8 zeroext %c) nounwind optsize noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i8* %ptr}, i64 0, metadata !14, metadata !{metadata !"0x102"}), !dbg !30
  tail call void @llvm.dbg.value(metadata !{float %val}, i64 0, metadata !15, metadata !{metadata !"0x102"}), !dbg !31
  tail call void @llvm.dbg.value(metadata !{i8 %c}, i64 0, metadata !16, metadata !{metadata !"0x102"}), !dbg !32
  %conv = fpext float %val to double, !dbg !33
  %conv3 = zext i8 %c to i32, !dbg !33
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %conv, i32 %conv3) nounwind optsize, !dbg !33
  ret i32 0, !dbg !35
}

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !17, metadata !{metadata !"0x102"}), !dbg !36
  tail call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !18, metadata !{metadata !"0x102"}), !dbg !37
  %conv = sitofp i32 %argc to double, !dbg !38
  %add = fadd double %conv, 5.555552e+05, !dbg !38
  %conv1 = fptrunc double %add to float, !dbg !38
  tail call void @llvm.dbg.value(metadata !{float %conv1}, i64 0, metadata !22, metadata !{metadata !"0x102"}), !dbg !38
  %call = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @.str1, i32 0, i32 0)) nounwind optsize, !dbg !39
  %add.ptr = getelementptr i8* bitcast (i32 (i32, i8**)* @main to i8*), i32 %argc, !dbg !40
  %add5 = add nsw i32 %argc, 97, !dbg !40
  %conv6 = trunc i32 %add5 to i8, !dbg !40
  tail call void @llvm.dbg.value(metadata !{i8* %add.ptr}, i64 0, metadata !8, metadata !{metadata !"0x102"}) nounwind, !dbg !41
  tail call void @llvm.dbg.value(metadata !{float %conv1}, i64 0, metadata !10, metadata !{metadata !"0x102"}) nounwind, !dbg !42
  tail call void @llvm.dbg.value(metadata !{i8 %conv6}, i64 0, metadata !12, metadata !{metadata !"0x102"}) nounwind, !dbg !43
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

!0 = metadata !{metadata !"0x2e\00inlineprinter\00inlineprinter\00\005\000\001\000\006\00256\001\005", metadata !51, metadata !1, metadata !3, null, i32 (i8*, float, i8)* @inlineprinter, null, null, metadata !48} ; [ DW_TAG_subprogram ] [line 5] [def] [inlineprinter]
!1 = metadata !{metadata !"0x29", metadata !51} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 3.0 (trunk 129915)\001\00\000\00\001", metadata !51, metadata !52, metadata !52, metadata !47, null,  null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !51, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !2} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x2e\00printer\00printer\00\0012\000\001\000\006\00256\001\0012", metadata !51, metadata !1, metadata !3, null, i32 (i8*, float, i8)* @printer, null, null, metadata !49} ; [ DW_TAG_subprogram ] [line 12] [def] [printer]
!7 = metadata !{metadata !"0x2e\00main\00main\00\0018\000\001\000\006\00256\001\0018", metadata !51, metadata !1, metadata !3, null, i32 (i32, i8**)* @main, null, null, metadata !50} ; [ DW_TAG_subprogram ] [line 18] [def] [main]
!8 = metadata !{metadata !"0x101\00ptr\0016777220\000", metadata !0, metadata !1, metadata !9} ; [ DW_TAG_arg_variable ]
!9 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !2, null} ; [ DW_TAG_pointer_type ]
!10 = metadata !{metadata !"0x101\00val\0033554436\000", metadata !0, metadata !1, metadata !11} ; [ DW_TAG_arg_variable ]
!11 = metadata !{metadata !"0x24\00float\000\0032\0032\000\000\004", null, metadata !2} ; [ DW_TAG_base_type ]
!12 = metadata !{metadata !"0x101\00c\0050331652\000", metadata !0, metadata !1, metadata !13} ; [ DW_TAG_arg_variable ]
!13 = metadata !{metadata !"0x24\00unsigned char\000\008\008\000\000\008", null, metadata !2} ; [ DW_TAG_base_type ]
!14 = metadata !{metadata !"0x101\00ptr\0016777227\000", metadata !6, metadata !1, metadata !9} ; [ DW_TAG_arg_variable ]
!15 = metadata !{metadata !"0x101\00val\0033554443\000", metadata !6, metadata !1, metadata !11} ; [ DW_TAG_arg_variable ]
!16 = metadata !{metadata !"0x101\00c\0050331659\000", metadata !6, metadata !1, metadata !13} ; [ DW_TAG_arg_variable ]
!17 = metadata !{metadata !"0x101\00argc\0016777233\000", metadata !7, metadata !1, metadata !5} ; [ DW_TAG_arg_variable ]
!18 = metadata !{metadata !"0x101\00argv\0033554449\000", metadata !7, metadata !1, metadata !19} ; [ DW_TAG_arg_variable ]
!19 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !2, metadata !20} ; [ DW_TAG_pointer_type ]
!20 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", null, metadata !2, metadata !21} ; [ DW_TAG_pointer_type ]
!21 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", null, metadata !2} ; [ DW_TAG_base_type ]
!22 = metadata !{metadata !"0x100\00dval\0019\000", metadata !23, metadata !1, metadata !11} ; [ DW_TAG_auto_variable ]
!23 = metadata !{metadata !"0xb\0018\001\002", metadata !51, metadata !7} ; [ DW_TAG_lexical_block ]
!24 = metadata !{i32 4, i32 22, metadata !0, null}
!25 = metadata !{i32 4, i32 33, metadata !0, null}
!26 = metadata !{i32 4, i32 52, metadata !0, null}
!27 = metadata !{i32 6, i32 3, metadata !28, null}
!28 = metadata !{metadata !"0xb\005\001\000", metadata !51, metadata !0} ; [ DW_TAG_lexical_block ]
!29 = metadata !{i32 7, i32 3, metadata !28, null}
!30 = metadata !{i32 11, i32 42, metadata !6, null}
!31 = metadata !{i32 11, i32 53, metadata !6, null}
!32 = metadata !{i32 11, i32 72, metadata !6, null}
!33 = metadata !{i32 13, i32 3, metadata !34, null}
!34 = metadata !{metadata !"0xb\0012\001\001", metadata !51, metadata !6} ; [ DW_TAG_lexical_block ]
!35 = metadata !{i32 14, i32 3, metadata !34, null}
!36 = metadata !{i32 17, i32 15, metadata !7, null}
!37 = metadata !{i32 17, i32 28, metadata !7, null}
!38 = metadata !{i32 19, i32 31, metadata !23, null}
!39 = metadata !{i32 20, i32 3, metadata !23, null}
!40 = metadata !{i32 21, i32 3, metadata !23, null}
!41 = metadata !{i32 4, i32 22, metadata !0, metadata !40}
!42 = metadata !{i32 4, i32 33, metadata !0, metadata !40}
!43 = metadata !{i32 4, i32 52, metadata !0, metadata !40}
!44 = metadata !{i32 6, i32 3, metadata !28, metadata !40}
!45 = metadata !{i32 22, i32 3, metadata !23, null}
!46 = metadata !{i32 23, i32 1, metadata !23, null}
!47 = metadata !{metadata !0, metadata !6, metadata !7}
!48 = metadata !{metadata !8, metadata !10, metadata !12}
!49 = metadata !{metadata !14, metadata !15, metadata !16}
!50 = metadata !{metadata !17, metadata !18, metadata !22}
!51 = metadata !{metadata !"a.c", metadata !"/private/tmp"}
!52 = metadata !{i32 0}
!53 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
