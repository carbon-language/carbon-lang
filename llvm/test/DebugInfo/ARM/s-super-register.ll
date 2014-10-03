; RUN: llc < %s - -filetype=obj | llvm-dwarfdump -debug-dump=loc - | FileCheck %s
target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "thumbv7-apple-macosx10.6.7"

; The S registers on ARM are expressed as pieces of their super-registers in DWARF.
;
; 0x90   DW_OP_regx of super-register
; 0x93   DW_OP_piece
; 0x9d   DW_OP_bit_piece
; CHECK:            Location description: 90 {{.. .. ((93 ..)|(9d .. ..)) $}}

define void @_Z3foov() optsize ssp {
entry:
  %call = tail call float @_Z3barv() optsize, !dbg !11
  tail call void @llvm.dbg.value(metadata !{float %call}, i64 0, metadata !5, metadata !{metadata !"0x102"}), !dbg !11
  %call16 = tail call float @_Z2f2v() optsize, !dbg !12
  %cmp7 = fcmp olt float %call, %call16, !dbg !12
  br i1 %cmp7, label %for.body, label %for.end, !dbg !12

for.body:                                         ; preds = %entry, %for.body
  %k.08 = phi float [ %inc, %for.body ], [ %call, %entry ]
  %call4 = tail call float @_Z2f3f(float %k.08) optsize, !dbg !13
  %inc = fadd float %k.08, 1.000000e+00, !dbg !14
  %call1 = tail call float @_Z2f2v() optsize, !dbg !12
  %cmp = fcmp olt float %inc, %call1, !dbg !12
  br i1 %cmp, label %for.body, label %for.end, !dbg !12

for.end:                                          ; preds = %for.body, %entry
  ret void, !dbg !15
}

declare float @_Z3barv() optsize

declare float @_Z2f2v() optsize

declare float @_Z2f3f(float) optsize

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20}

!0 = metadata !{metadata !"0x11\004\00clang version 3.0 (trunk 130845)\001\00\000\00\001", metadata !18, metadata !19, metadata !19, metadata !16, null,  null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00_Z3foov\005\000\001\000\006\00256\001\005", metadata !18, metadata !2, metadata !3, null, void ()* @_Z3foov, null, null, metadata !17} ; [ DW_TAG_subprogram ] [line 5] [def] [foo]
!2 = metadata !{metadata !"0x29", metadata !18} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !18, metadata !2, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{null}
!5 = metadata !{metadata !"0x100\00k\006\000", metadata !6, metadata !2, metadata !7} ; [ DW_TAG_auto_variable ]
!6 = metadata !{metadata !"0xb\005\0012\000", metadata !18, metadata !1} ; [ DW_TAG_lexical_block ]
!7 = metadata !{metadata !"0x24\00float\000\0032\0032\000\000\004", null, metadata !0} ; [ DW_TAG_base_type ]
!8 = metadata !{metadata !"0x100\00y\008\000", metadata !9, metadata !2, metadata !7} ; [ DW_TAG_auto_variable ]
!9 = metadata !{metadata !"0xb\007\0025\002", metadata !18, metadata !10} ; [ DW_TAG_lexical_block ]
!10 = metadata !{metadata !"0xb\007\003\001", metadata !18, metadata !6} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 6, i32 18, metadata !6, null}
!12 = metadata !{i32 7, i32 3, metadata !6, null}
!13 = metadata !{i32 8, i32 20, metadata !9, null}
!14 = metadata !{i32 7, i32 20, metadata !10, null}
!15 = metadata !{i32 10, i32 1, metadata !6, null}
!16 = metadata !{metadata !1}
!17 = metadata !{metadata !5, metadata !8}
!18 = metadata !{metadata !"k.cc", metadata !"/private/tmp"}
!19 = metadata !{i32 0}
!20 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
