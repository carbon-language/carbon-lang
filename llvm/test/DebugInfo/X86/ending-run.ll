; RUN: llc -mtriple=x86_64-apple-darwin -filetype=obj %s -o %t
; RUN: llvm-dwarfdump -debug-dump=line %t | FileCheck %s

; Check that the line table starts at 7, not 4, but that the first
; statement isn't until line 8.

; CHECK-NOT: 0x0000000000000000      7      0      1   0  0  is_stmt
; CHECK: 0x0000000000000000      7      0      1   0
; CHECK: 0x0000000000000004      8     18      1   0  0  is_stmt prologue_end

define i32 @callee(i32 %x) nounwind uwtable ssp {
entry:
  %x.addr = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr}, metadata !12, metadata !{metadata !"0x102"}), !dbg !13
  call void @llvm.dbg.declare(metadata !{i32* %y}, metadata !14, metadata !{metadata !"0x102"}), !dbg !16
  %0 = load i32* %x.addr, align 4, !dbg !17
  %1 = load i32* %x.addr, align 4, !dbg !17
  %mul = mul nsw i32 %0, %1, !dbg !17
  store i32 %mul, i32* %y, align 4, !dbg !17
  %2 = load i32* %y, align 4, !dbg !18
  %sub = sub nsw i32 %2, 2, !dbg !18
  ret i32 %sub, !dbg !18
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.1 (trunk 153921) (llvm/trunk 153916)\000\00\000\00\000", metadata !19, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00callee\00callee\00\004\000\001\000\006\000\000\007", metadata !19, metadata !6, metadata !7, null, i32 (i32)* @callee, null, null, null} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x29", metadata !19} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!12 = metadata !{metadata !"0x101\00x\0016777221\000", metadata !5, metadata !6, metadata !9} ; [ DW_TAG_arg_variable ]
!13 = metadata !{i32 5, i32 5, metadata !5, null}
!14 = metadata !{metadata !"0x100\00y\008\000", metadata !15, metadata !6, metadata !9} ; [ DW_TAG_auto_variable ]
!15 = metadata !{metadata !"0xb\007\001\000", metadata !19, metadata !5} ; [ DW_TAG_lexical_block ]
!16 = metadata !{i32 8, i32 9, metadata !15, null}
!17 = metadata !{i32 8, i32 18, metadata !15, null}
!18 = metadata !{i32 9, i32 5, metadata !15, null}
!19 = metadata !{metadata !"ending-run.c", metadata !"/Users/echristo/tmp"}
!20 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
