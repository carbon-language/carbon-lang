; RUN: llc -O2 < %s | FileCheck %s
; RUN: llc -O2 -regalloc=basic < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10"

%struct.a = type { i32, %struct.a* }

@llvm.used = appending global [1 x i8*] [i8* bitcast (i8* (%struct.a*)* @bar to i8*)], section "llvm.metadata" ; <[1 x i8*]*> [#uses=0]

define i8* @bar(%struct.a* %myvar) nounwind optsize noinline ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.a* %myvar}, i64 0, metadata !8, metadata !{metadata !"0x102"})
  %0 = getelementptr inbounds %struct.a* %myvar, i64 0, i32 0, !dbg !28 ; <i32*> [#uses=1]
  %1 = load i32* %0, align 8, !dbg !28            ; <i32> [#uses=1]
  tail call void @foo(i32 %1) nounwind optsize noinline ssp, !dbg !28
  %2 = bitcast %struct.a* %myvar to i8*, !dbg !30 ; <i8*> [#uses=1]
  ret i8* %2, !dbg !30
}

declare void @foo(i32) nounwind optsize noinline ssp

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!38}

!0 = metadata !{metadata !"0x34\00ret\00ret\00\007\000\001", metadata !1, metadata !1, metadata !3, null, null} ; [ DW_TAG_variable ]
!1 = metadata !{metadata !"0x29", metadata !36} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\001\004.2.1 (Based on Apple Inc. build 5658) (LLVM build)\001\00\000\00\001", metadata !36, metadata !37, metadata !37, metadata !32, metadata !31,  metadata !37} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", metadata !36, metadata !1} ; [ DW_TAG_base_type ]
!4 = metadata !{metadata !"0x101\00x\0012\000", metadata !5, metadata !1, metadata !3} ; [ DW_TAG_arg_variable ]
!5 = metadata !{metadata !"0x2e\00foo\00foo\00foo\0013\000\001\000\006\000\001\0013", metadata !36, metadata !1, metadata !6, null, void (i32)* @foo, null, null, metadata !33} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !36, metadata !1, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !3}
!8 = metadata !{metadata !"0x101\00myvar\0017\000", metadata !9, metadata !1, metadata !13} ; [ DW_TAG_arg_variable ]
!9 = metadata !{metadata !"0x2e\00bar\00bar\00bar\0017\000\001\000\006\000\001\0017", metadata !36, metadata !1, metadata !10, null, i8* (%struct.a*)* @bar, null, null, metadata !34} ; [ DW_TAG_subprogram ]
!10 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !36, metadata !1, null, metadata !11, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{metadata !12, metadata !13}
!12 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !36, metadata !1, null} ; [ DW_TAG_pointer_type ]
!13 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !36, metadata !1, metadata !14} ; [ DW_TAG_pointer_type ]
!14 = metadata !{metadata !"0x13\00a\002\00128\0064\000\000\000", metadata !36, metadata !1, null, metadata !15, null, null, null} ; [ DW_TAG_structure_type ] [a] [line 2, size 128, align 64, offset 0] [def] [from ]
!15 = metadata !{metadata !16, metadata !17}
!16 = metadata !{metadata !"0xd\00c\003\0032\0032\000\000", metadata !36, metadata !14, metadata !3} ; [ DW_TAG_member ]
!17 = metadata !{metadata !"0xd\00d\004\0064\0064\0064\000", metadata !36, metadata !14, metadata !13} ; [ DW_TAG_member ]
!18 = metadata !{metadata !"0x101\00argc\0022\000", metadata !19, metadata !1, metadata !3} ; [ DW_TAG_arg_variable ]
!19 = metadata !{metadata !"0x2e\00main\00main\00main\0022\000\001\000\006\000\001\0022", metadata !36, metadata !1, metadata !20, null, null, null, null, metadata !35} ; [ DW_TAG_subprogram ]
!20 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !36, metadata !1, null, metadata !21, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{metadata !3, metadata !3, metadata !22}
!22 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !36, metadata !1, metadata !23} ; [ DW_TAG_pointer_type ]
!23 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", metadata !36, metadata !1, metadata !24} ; [ DW_TAG_pointer_type ]
!24 = metadata !{metadata !"0x24\00char\000\008\008\000\000\006", metadata !36, metadata !1} ; [ DW_TAG_base_type ]
!25 = metadata !{metadata !"0x101\00argv\0022\000", metadata !19, metadata !1, metadata !22} ; [ DW_TAG_arg_variable ]
!26 = metadata !{metadata !"0x100\00e\0023\000", metadata !27, metadata !1, metadata !14} ; [ DW_TAG_auto_variable ]
!27 = metadata !{metadata !"0xb\0022\000\000", metadata !36, metadata !19} ; [ DW_TAG_lexical_block ]
!28 = metadata !{i32 18, i32 0, metadata !29, null}
!29 = metadata !{metadata !"0xb\0017\000\001", metadata !36, metadata !9} ; [ DW_TAG_lexical_block ]
!30 = metadata !{i32 19, i32 0, metadata !29, null}
!31 = metadata !{metadata !0}
!32 = metadata !{metadata !5, metadata !9, metadata !19}
!33 = metadata !{metadata !4}
!34 = metadata !{metadata !8}
!35 = metadata !{metadata !18, metadata !25, metadata !26}
!36 = metadata !{metadata !"foo.c", metadata !"/tmp/"}
!37 = metadata !{}

; The variable bar:myvar changes registers after the first movq.
; It is cobbered by popq %rbx
; CHECK: movq
; CHECK-NEXT: [[LABEL:Ltmp[0-9]*]]
; CHECK: .loc	1 19 0
; CHECK: popq
; CHECK-NEXT: [[CLOBBER:Ltmp[0-9]*]]


; CHECK: Ldebug_loc0:
; CHECK-NEXT: [[SET1:.*]] = Lfunc_begin0-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET1]]
; CHECK-NEXT: [[SET2:.*]] = [[LABEL]]-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET2]]
; CHECK-NEXT: Lset{{.*}} = Ltmp{{.*}}-Ltmp{{.*}}               ## Loc expr size
; CHECK-NEXT: .short  Lset{{.*}}
; CHECK-NEXT: Ltmp{{.*}}:
; CHECK-NEXT: .byte   85
; CHECK-NEXT: Ltmp{{.*}}:
; CHECK-NEXT: [[SET3:.*]] = [[LABEL]]-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET3]]
; CHECK-NEXT: [[SET4:.*]] = [[CLOBBER]]-Lfunc_begin0
; CHECK-NEXT: .quad   [[SET4]]
; CHECK-NEXT: Lset{{.*}} = Ltmp{{.*}}-Ltmp{{.*}}               ## Loc expr size
; CHECK-NEXT: .short  Lset{{.*}}
; CHECK-NEXT: Ltmp{{.*}}:
; CHECK-NEXT: .byte   83
; CHECK-NEXT: Ltmp{{.*}}:
!38 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
