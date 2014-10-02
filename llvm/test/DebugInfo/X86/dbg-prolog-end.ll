; RUN: llc -O0 < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-macosx10.6.7"

;CHECK: .loc	1 2 11 prologue_end
define i32 @foo(i32 %i) nounwind ssp {
entry:
  %i.addr = alloca i32, align 4
  %j = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !7, metadata !{metadata !"0x102"}), !dbg !8
  call void @llvm.dbg.declare(metadata !{i32* %j}, metadata !9, metadata !{metadata !"0x102"}), !dbg !11
  store i32 2, i32* %j, align 4, !dbg !12
  %tmp = load i32* %j, align 4, !dbg !13
  %inc = add nsw i32 %tmp, 1, !dbg !13
  store i32 %inc, i32* %j, align 4, !dbg !13
  %tmp1 = load i32* %j, align 4, !dbg !14
  %tmp2 = load i32* %i.addr, align 4, !dbg !14
  %add = add nsw i32 %tmp1, %tmp2, !dbg !14
  store i32 %add, i32* %j, align 4, !dbg !14
  %tmp3 = load i32* %j, align 4, !dbg !15
  ret i32 %tmp3, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

define i32 @main() nounwind ssp {
entry:
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 @foo(i32 21), !dbg !16
  ret i32 %call, !dbg !16
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21}
!18 = metadata !{metadata !1, metadata !6}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.0 (trunk 131100)\000\00\000\00\000", metadata !19, metadata !20, metadata !20, metadata !18, null,  null} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"0x2e\00foo\00foo\00\001\000\001\000\006\00256\000\001", metadata !19, metadata !2, metadata !3, null, i32 (i32)* @foo, null, null, null} ; [ DW_TAG_subprogram ] [line 1] [def] [foo]
!2 = metadata !{metadata !"0x29", metadata !19} ; [ DW_TAG_file_type ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !19, metadata !2, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !0} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x2e\00main\00main\00\007\000\001\000\006\000\000\007", metadata !19, metadata !2, metadata !3, null, i32 ()* @main, null, null, null} ; [ DW_TAG_subprogram ] [line 7] [def] [main]
!7 = metadata !{metadata !"0x101\00i\0016777217\000", metadata !1, metadata !2, metadata !5} ; [ DW_TAG_arg_variable ]
!8 = metadata !{i32 1, i32 13, metadata !1, null}
!9 = metadata !{metadata !"0x100\00j\002\000", metadata !10, metadata !2, metadata !5} ; [ DW_TAG_auto_variable ]
!10 = metadata !{metadata !"0xb\001\0016\000", metadata !19, metadata !1} ; [ DW_TAG_lexical_block ]
!11 = metadata !{i32 2, i32 6, metadata !10, null}
!12 = metadata !{i32 2, i32 11, metadata !10, null}
!13 = metadata !{i32 3, i32 2, metadata !10, null}
!14 = metadata !{i32 4, i32 2, metadata !10, null}
!15 = metadata !{i32 5, i32 2, metadata !10, null}
!16 = metadata !{i32 8, i32 2, metadata !17, null}
!17 = metadata !{metadata !"0xb\007\0012\001", metadata !19, metadata !6} ; [ DW_TAG_lexical_block ]
!19 = metadata !{metadata !"/tmp/a.c", metadata !"/private/tmp"}
!20 = metadata !{i32 0}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
