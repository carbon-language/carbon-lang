; RUN: llc < %s | FileCheck %s
; Radar 9309221
; Test dwarf reg no for d16
;CHECK: DW_OP_regx
;CHECK-NEXT: 272

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-darwin10"

@.str = private unnamed_addr constant [11 x i8] c"%p %lf %c\0A\00", align 4
@.str1 = private unnamed_addr constant [6 x i8] c"point\00", align 4

define i32 @inlineprinter(i8* %ptr, double %val, i8 zeroext %c) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata !{i8* %ptr}, i64 0, metadata !19, metadata !{metadata !"0x102"}), !dbg !26
  tail call void @llvm.dbg.value(metadata !{double %val}, i64 0, metadata !20, metadata !{metadata !"0x102"}), !dbg !26
  tail call void @llvm.dbg.value(metadata !{i8 %c}, i64 0, metadata !21, metadata !{metadata !"0x102"}), !dbg !26
  %0 = zext i8 %c to i32, !dbg !27
  %1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %val, i32 %0) nounwind, !dbg !27
  ret i32 0, !dbg !29
}

define i32 @printer(i8* %ptr, double %val, i8 zeroext %c) nounwind optsize noinline {
entry:
  tail call void @llvm.dbg.value(metadata !{i8* %ptr}, i64 0, metadata !16, metadata !{metadata !"0x102"}), !dbg !30
  tail call void @llvm.dbg.value(metadata !{double %val}, i64 0, metadata !17, metadata !{metadata !"0x102"}), !dbg !30
  tail call void @llvm.dbg.value(metadata !{i8 %c}, i64 0, metadata !18, metadata !{metadata !"0x102"}), !dbg !30
  %0 = zext i8 %c to i32, !dbg !31
  %1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %ptr, double %val, i32 %0) nounwind, !dbg !31
  ret i32 0, !dbg !33
}

declare i32 @printf(i8* nocapture, ...) nounwind

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @main(i32 %argc, i8** nocapture %argv) nounwind optsize {
entry:
  tail call void @llvm.dbg.value(metadata !{i32 %argc}, i64 0, metadata !22, metadata !{metadata !"0x102"}), !dbg !34
  tail call void @llvm.dbg.value(metadata !{i8** %argv}, i64 0, metadata !23, metadata !{metadata !"0x102"}), !dbg !34
  %0 = sitofp i32 %argc to double, !dbg !35
  %1 = fadd double %0, 5.555552e+05, !dbg !35
  tail call void @llvm.dbg.value(metadata !{double %1}, i64 0, metadata !24, metadata !{metadata !"0x102"}), !dbg !35
  %2 = tail call i32 @puts(i8* getelementptr inbounds ([6 x i8]* @.str1, i32 0, i32 0)) nounwind, !dbg !36
  %3 = getelementptr inbounds i8* bitcast (i32 (i32, i8**)* @main to i8*), i32 %argc, !dbg !37
  %4 = trunc i32 %argc to i8, !dbg !37
  %5 = add i8 %4, 97, !dbg !37
  tail call void @llvm.dbg.value(metadata !{i8* %3}, i64 0, metadata !19, metadata !{metadata !"0x102"}) nounwind, !dbg !38
  tail call void @llvm.dbg.value(metadata !{double %1}, i64 0, metadata !20, metadata !{metadata !"0x102"}) nounwind, !dbg !38
  tail call void @llvm.dbg.value(metadata !{i8 %5}, i64 0, metadata !21, metadata !{metadata !"0x102"}) nounwind, !dbg !38
  %6 = zext i8 %5 to i32, !dbg !39
  %7 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([11 x i8]* @.str, i32 0, i32 0), i8* %3, double %1, i32 %6) nounwind, !dbg !39
  %8 = tail call i32 @printer(i8* %3, double %1, i8 zeroext %5) nounwind, !dbg !40
  ret i32 0, !dbg !41
}

declare i32 @puts(i8* nocapture) nounwind

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!48}

!0 = metadata !{metadata !"0x2e\00printer\00printer\00printer\0012\000\001\000\006\00256\001\0012", metadata !46, metadata !1, metadata !3, null, i32 (i8*, double, i8)* @printer, null, null, metadata !43} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !46} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\001\00(LLVM build 00)\001\00\000\00\001", metadata !46, metadata !47, metadata !47, metadata !42, null,  null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !46, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5, metadata !6, metadata !7, metadata !8}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !46, metadata !1} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !46, metadata !1, null} ; [ DW_TAG_pointer_type ]
!7 = metadata !{metadata !"0x24\00double\000\0064\0032\000\000\004", metadata !46, metadata !1} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !"0x24\00unsigned char\000\008\008\000\000\008", metadata !46, metadata !1} ; [ DW_TAG_base_type ]
!9 = metadata !{metadata !"0x2e\00inlineprinter\00inlineprinter\00inlineprinter\005\000\001\000\006\00256\001\005", metadata !46, metadata !1, metadata !3, null, i32 (i8*, double, i8)* @inlineprinter, null, null, metadata !44} ; [ DW_TAG_subprogram ]
!10 = metadata !{metadata !"0x2e\00main\00main\00main\0018\000\001\000\006\00256\001\0018", metadata !46, metadata !1, metadata !11, null, i32 (i32, i8**)* @main, null, null, metadata !45} ; [ DW_TAG_subprogram ]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !46, metadata !1, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !5, metadata !5, metadata !13}
!13 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !46, metadata !1, metadata !14} ; [ DW_TAG_pointer_type ]
!14 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !46, metadata !1, metadata !15} ; [ DW_TAG_pointer_type ]
!15 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", metadata !46, metadata !1} ; [ DW_TAG_base_type ]
!16 = metadata !{metadata !"0x101\00ptr\0011\000", metadata !0, metadata !1, metadata !6} ; [ DW_TAG_arg_variable ]
!17 = metadata !{metadata !"0x101\00val\0011\000", metadata !0, metadata !1, metadata !7} ; [ DW_TAG_arg_variable ]
!18 = metadata !{metadata !"0x101\00c\0011\000", metadata !0, metadata !1, metadata !8} ; [ DW_TAG_arg_variable ]
!19 = metadata !{metadata !"0x101\00ptr\004\000", metadata !9, metadata !1, metadata !6} ; [ DW_TAG_arg_variable ]
!20 = metadata !{metadata !"0x101\00val\004\000", metadata !9, metadata !1, metadata !7} ; [ DW_TAG_arg_variable ]
!21 = metadata !{metadata !"0x101\00c\004\000", metadata !9, metadata !1, metadata !8} ; [ DW_TAG_arg_variable ]
!22 = metadata !{metadata !"0x101\00argc\0017\000", metadata !10, metadata !1, metadata !5} ; [ DW_TAG_arg_variable ]
!23 = metadata !{metadata !"0x101\00argv\0017\000", metadata !10, metadata !1, metadata !13} ; [ DW_TAG_arg_variable ]
!24 = metadata !{metadata !"0x100\00dval\0019\000", metadata !25, metadata !1, metadata !7} ; [ DW_TAG_auto_variable ]
!25 = metadata !{metadata !"0xb\0018\000\002", metadata !46, metadata !10} ; [ DW_TAG_lexical_block ]
!26 = metadata !{i32 4, i32 0, metadata !9, null}
!27 = metadata !{i32 6, i32 0, metadata !28, null}
!28 = metadata !{metadata !"0xb\005\000\001", metadata !46, metadata !9} ; [ DW_TAG_lexical_block ]
!29 = metadata !{i32 7, i32 0, metadata !28, null}
!30 = metadata !{i32 11, i32 0, metadata !0, null}
!31 = metadata !{i32 13, i32 0, metadata !32, null}
!32 = metadata !{metadata !"0xb\0012\000\000", metadata !46, metadata !0} ; [ DW_TAG_lexical_block ]
!33 = metadata !{i32 14, i32 0, metadata !32, null}
!34 = metadata !{i32 17, i32 0, metadata !10, null}
!35 = metadata !{i32 19, i32 0, metadata !25, null}
!36 = metadata !{i32 20, i32 0, metadata !25, null}
!37 = metadata !{i32 21, i32 0, metadata !25, null}
!38 = metadata !{i32 4, i32 0, metadata !9, metadata !37}
!39 = metadata !{i32 6, i32 0, metadata !28, metadata !37}
!40 = metadata !{i32 22, i32 0, metadata !25, null}
!41 = metadata !{i32 23, i32 0, metadata !25, null}
!42 = metadata !{metadata !0, metadata !9, metadata !10}
!43 = metadata !{metadata !16, metadata !17, metadata !18}
!44 = metadata !{metadata !19, metadata !20, metadata !21}
!45 = metadata !{metadata !22, metadata !23, metadata !24}
!46 = metadata !{metadata !"a.c", metadata !"/tmp/"}
!47 = metadata !{i32 0}
!48 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
