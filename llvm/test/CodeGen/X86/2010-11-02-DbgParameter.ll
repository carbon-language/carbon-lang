; RUN: llc -O2 -asm-verbose < %s | FileCheck %s
; Radar 8616981

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128-n8:16:32"
target triple = "i386-apple-darwin11.0.0"

%struct.bar = type { i32, i32 }

define i32 @foo(%struct.bar* nocapture %i) nounwind readnone optsize noinline ssp {
; CHECK: TAG_formal_parameter
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.bar* %i}, i64 0, metadata !6, metadata !{metadata !"0x102"}), !dbg !12
  ret i32 1, !dbg !13
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!19}

!0 = metadata !{metadata !"0x2e\00foo\00foo\00\003\000\001\000\006\00256\001\003", metadata !17, metadata !1, metadata !3, null, i32 (%struct.bar*)* @foo, null, null, metadata !16} ; [ DW_TAG_subprogram ]
!1 = metadata !{metadata !"0x29", metadata !17} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 2.9 (trunk 117922)\001\00\000\00\000", metadata !17, metadata !18, metadata !18, metadata !15, null,  null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !17, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !17, metadata !2} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x101\00i\003\000", metadata !0, metadata !1, metadata !7} ; [ DW_TAG_arg_variable ]
!7 = metadata !{metadata !"0xf\00\000\0032\0032\000\000", metadata !17, metadata !1, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{metadata !"0x13\00bar\002\0064\0032\000\000\000", metadata !17, metadata !1, null, metadata !9, null, null, null} ; [ DW_TAG_structure_type ] [bar] [line 2, size 64, align 32, offset 0] [def] [from ]
!9 = metadata !{metadata !10, metadata !11}
!10 = metadata !{metadata !"0xd\00x\002\0032\0032\000\000", metadata !17,  metadata !1, metadata !5} ; [ DW_TAG_member ]
!11 = metadata !{metadata !"0xd\00y\002\0032\0032\0032\000", metadata !17, metadata !1, metadata !5} ; [ DW_TAG_member ]
!12 = metadata !{i32 3, i32 47, metadata !0, null}
!13 = metadata !{i32 4, i32 2, metadata !14, null}
!14 = metadata !{metadata !"0xb\003\0050\000", metadata !17, metadata !0} ; [ DW_TAG_lexical_block ]
!15 = metadata !{metadata !0}
!16 = metadata !{metadata !6}
!17 = metadata !{metadata !"one.c", metadata !"/private/tmp"}
!18 = metadata !{i32 0}
!19 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
