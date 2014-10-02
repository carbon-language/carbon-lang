; RUN: llc -O0 -mtriple=x86_64-apple-darwin < %s -filetype=obj \
; RUN:     | llvm-dwarfdump -debug-dump=info - \
; RUN:     | FileCheck %s -check-prefix=CHECK -check-prefix=DWARF4
; RUN: llc -O0 -mtriple=x86_64-apple-darwin < %s -filetype=obj -dwarf-version=3 \
; RUN:     | llvm-dwarfdump -debug-dump=info - \
; RUN:     | FileCheck %s -check-prefix=CHECK -check-prefix=DWARF3

; FIXME: The location here needs to be fixed, but llvm-dwarfdump doesn't handle
; DW_AT_location lists yet.
; DWARF4: DW_AT_location [DW_FORM_sec_offset]                      (0x00000000)

; FIXME: The location here needs to be fixed, but llvm-dwarfdump doesn't handle
; DW_AT_location lists yet.
; DWARF3: DW_AT_location [DW_FORM_data4]                      (0x00000000)

; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]  ( .debug_str[0x00000067] = "vla")

; Unfortunately llvm-dwarfdump can't unparse a list of DW_AT_locations
; right now, so we check the asm output:
; RUN: llc -O0 -mtriple=x86_64-apple-darwin %s -o - -filetype=asm | FileCheck %s -check-prefix=ASM-CHECK
; vla should have a register-indirect address at one point.
; ASM-CHECK: DEBUG_VALUE: vla <- RCX
; ASM-CHECK: DW_OP_breg2

define void @testVLAwithSize(i32 %s) nounwind uwtable ssp {
entry:
  %s.addr = alloca i32, align 4
  %saved_stack = alloca i8*
  %i = alloca i32, align 4
  store i32 %s, i32* %s.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %s.addr}, metadata !10, metadata !{metadata !"0x102"}), !dbg !11
  %0 = load i32* %s.addr, align 4, !dbg !12
  %1 = zext i32 %0 to i64, !dbg !12
  %2 = call i8* @llvm.stacksave(), !dbg !12
  store i8* %2, i8** %saved_stack, !dbg !12
  %vla = alloca i32, i64 %1, align 16, !dbg !12
  call void @llvm.dbg.declare(metadata !{i32* %vla}, metadata !14, metadata !30), !dbg !18
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !19, metadata !{metadata !"0x102"}), !dbg !20
  store i32 0, i32* %i, align 4, !dbg !21
  br label %for.cond, !dbg !21

for.cond:                                         ; preds = %for.inc, %entry
  %3 = load i32* %i, align 4, !dbg !21
  %4 = load i32* %s.addr, align 4, !dbg !21
  %cmp = icmp slt i32 %3, %4, !dbg !21
  br i1 %cmp, label %for.body, label %for.end, !dbg !21

for.body:                                         ; preds = %for.cond
  %5 = load i32* %i, align 4, !dbg !23
  %6 = load i32* %i, align 4, !dbg !23
  %mul = mul nsw i32 %5, %6, !dbg !23
  %7 = load i32* %i, align 4, !dbg !23
  %idxprom = sext i32 %7 to i64, !dbg !23
  %arrayidx = getelementptr inbounds i32* %vla, i64 %idxprom, !dbg !23
  store i32 %mul, i32* %arrayidx, align 4, !dbg !23
  br label %for.inc, !dbg !25

for.inc:                                          ; preds = %for.body
  %8 = load i32* %i, align 4, !dbg !26
  %inc = add nsw i32 %8, 1, !dbg !26
  store i32 %inc, i32* %i, align 4, !dbg !26
  br label %for.cond, !dbg !26

for.end:                                          ; preds = %for.cond
  %9 = load i8** %saved_stack, !dbg !27
  call void @llvm.stackrestore(i8* %9), !dbg !27
  ret void, !dbg !27
}

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare i8* @llvm.stacksave() nounwind

declare void @llvm.stackrestore(i8*) nounwind

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!29}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.2 (trunk 156005) (llvm/trunk 156000)\000\00\000\00\001", metadata !28, metadata !1, metadata !1, metadata !3, metadata !1,  metadata !1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{}
!3 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x2e\00testVLAwithSize\00testVLAwithSize\00\001\000\001\000\006\00256\000\002", metadata !28, metadata !6, metadata !7, null, void (i32)* @testVLAwithSize, null, null, metadata !1} ; [ DW_TAG_subprogram ]
!6 = metadata !{metadata !"0x29", metadata !28} ; [ DW_TAG_file_type ]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null, metadata !9}
!9 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ]
!10 = metadata !{metadata !"0x101\00s\0016777217\000", metadata !5, metadata !6, metadata !9} ; [ DW_TAG_arg_variable ]
!11 = metadata !{i32 1, i32 26, metadata !5, null}
!12 = metadata !{i32 3, i32 13, metadata !13, null}
!13 = metadata !{metadata !"0xb\002\001\000", metadata !28, metadata !5} ; [ DW_TAG_lexical_block ]
!14 = metadata !{metadata !"0x100\00vla\003\008192", metadata !13, metadata !6, metadata !15} ; [ DW_TAG_auto_variable ]
!15 = metadata !{metadata !"0x1\00\000\000\0032\000\000", null, null, metadata !9, metadata !16, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 0, align 32, offset 0] [from int]
!16 = metadata !{metadata !17}
!17 = metadata !{metadata !"0x21\000\00-1"}        ; [ DW_TAG_subrange_type ]
!18 = metadata !{i32 3, i32 7, metadata !13, null}
!19 = metadata !{metadata !"0x100\00i\004\000", metadata !13, metadata !6, metadata !9} ; [ DW_TAG_auto_variable ]
!20 = metadata !{i32 4, i32 7, metadata !13, null}
!21 = metadata !{i32 5, i32 8, metadata !22, null}
!22 = metadata !{metadata !"0xb\005\003\001", metadata !28, metadata !13} ; [ DW_TAG_lexical_block ]
!23 = metadata !{i32 6, i32 5, metadata !24, null}
!24 = metadata !{metadata !"0xb\005\0027\002", metadata !28, metadata !22} ; [ DW_TAG_lexical_block ]
!25 = metadata !{i32 7, i32 3, metadata !24, null}
!26 = metadata !{i32 5, i32 22, metadata !22, null}
!27 = metadata !{i32 8, i32 1, metadata !13, null}
!28 = metadata !{metadata !"bar.c", metadata !"/Users/echristo/tmp"}
!29 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!30 = metadata !{metadata !"0x102\006"} ; [ DW_TAG_expression ] [DW_OP_deref]
