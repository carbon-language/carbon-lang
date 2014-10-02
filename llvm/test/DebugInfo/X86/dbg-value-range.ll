; RUN: llc -mtriple=x86_64-apple-darwin10 < %s | FileCheck %s

%struct.a = type { i32 }

define i32 @bar(%struct.a* nocapture %b) nounwind ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.a* %b}, i64 0, metadata !6, metadata !{metadata !"0x102"}), !dbg !13
  %tmp1 = getelementptr inbounds %struct.a* %b, i64 0, i32 0, !dbg !14
  %tmp2 = load i32* %tmp1, align 4, !dbg !14
  tail call void @llvm.dbg.value(metadata !{i32 %tmp2}, i64 0, metadata !11, metadata !{metadata !"0x102"}), !dbg !14
  %call = tail call i32 (...)* @foo(i32 %tmp2) nounwind , !dbg !18
  %add = add nsw i32 %tmp2, 1, !dbg !19
  ret i32 %add, !dbg !19
}

declare i32 @foo(...) 

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!24}

!0 = metadata !{metadata !"0x2e\00bar\00bar\00\005\000\001\000\006\00256\001\000", metadata !22, metadata !1, metadata !3, null, i32 (%struct.a*)* @bar, null, null, metadata !21} ; [ DW_TAG_subprogram ] [line 5] [def] [scope 0] [bar]
!1 = metadata !{metadata !"0x29", metadata !22} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 2.9 (trunk 122997)\001\00\000\00\001", metadata !22, metadata !23, metadata !23, metadata !20, null,  null} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !22, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !2} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x101\00b\005\000", metadata !0, metadata !1, metadata !7} ; [ DW_TAG_arg_variable ]
!7 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, metadata !2, metadata !8} ; [ DW_TAG_pointer_type ]
!8 = metadata !{metadata !"0x13\00a\001\0032\0032\000\000\000", metadata !22, metadata !2, null, metadata !9, null, null, null} ; [ DW_TAG_structure_type ] [a] [line 1, size 32, align 32, offset 0] [def] [from ]
!9 = metadata !{metadata !10}
!10 = metadata !{metadata !"0xd\00c\002\0032\0032\000\000", metadata !22, metadata !1, metadata !5} ; [ DW_TAG_member ]
!11 = metadata !{metadata !"0x100\00x\006\000", metadata !12, metadata !1, metadata !5} ; [ DW_TAG_auto_variable ]
!12 = metadata !{metadata !"0xb\005\0022\000", metadata !22, metadata !0} ; [ DW_TAG_lexical_block ]
!13 = metadata !{i32 5, i32 19, metadata !0, null}
!14 = metadata !{i32 6, i32 14, metadata !12, null}
!18 = metadata !{i32 7, i32 2, metadata !12, null}
!19 = metadata !{i32 8, i32 2, metadata !12, null}
!20 = metadata !{metadata !0}
!21 = metadata !{metadata !6, metadata !11}
!22 = metadata !{metadata !"bar.c", metadata !"/private/tmp"}
!23 = metadata !{i32 0}

; Check that variable bar:b value range is appropriately truncated in debug info.
; The variable is in %rdi which is clobbered by 'movl %ebx, %edi'
; Here Ltmp7 is the end of the location range.

;CHECK: .loc	1 7 2
;CHECK: movl
;CHECK-NEXT: [[CLOBBER:Ltmp[0-9]*]]

;CHECK:Ldebug_loc0:
;CHECK-NEXT: Lset{{.*}} =
;CHECK-NEXT:	.quad
;CHECK-NEXT: [[CLOBBER_OFF:Lset.*]] = [[CLOBBER]]-{{.*}}
;CHECK-NEXT:	.quad	[[CLOBBER_OFF]]
;CHECK-NEXT: Lset{{.*}} = Ltmp{{.*}}-Ltmp{{.*}}
;CHECK-NEXT:    .short  Lset
;CHECK-NEXT: Ltmp
;CHECK-NEXT:	.byte	85 ## DW_OP_reg
;CHECK-NEXT: Ltmp
;CHECK-NEXT:	.quad	0
;CHECK-NEXT:	.quad	0
!24 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
