; RUN: not llvm-as -disable-output -verify-debug-info < %s 2>&1 | FileCheck %s
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i64 %s.coerce0, i32 %s.coerce1) #0 {
entry:
  call void @llvm.dbg.value(metadata i64 %s.coerce0, i64 0, metadata !20, metadata !24), !dbg !21
  call void @llvm.dbg.value(metadata i32 %s.coerce1, i64 0, metadata !22, metadata !27), !dbg !21
  ret i32 %s.coerce1, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!0 = !{!"0x11\0012\00clang version 3.5 \001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ]
!1 = !{!"pieces.c", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\003\000\001\000\006\00256\001\003", !1, !5, !6, null, i32 (i64, i32)* @foo, null, null, !15} ; [ DW_TAG_subprogram ] [line 3] [def] [foo]
!5 = !{!"0x29", !1}          ; [ DW_TAG_file_type ] [/pieces.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !9}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0x16\00S\001\000\000\000\000", !1, null, !10} ; [ DW_TAG_typedef ] [S] [line 1, size 0, align 0, offset 0] [from ]
!10 = !{!"0x13\00\001\00128\0064\000\000\000", !1, null, null, !11, null, null, null} ; [ DW_TAG_structure_type ] [line 1, size 128, align 64, offset 0] [def] [from ]
!11 = !{!12, !14}
!12 = !{!"0xd\00a\001\0064\0064\000\000", !1, !10, !13} ; [ DW_TAG_member ] [a] [line 1, size 64, align 64, offset 0] [from long int]
!13 = !{!"0x24\00long int\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!14 = !{!"0xd\00b\001\0032\0032\0064\000", !1, !10, !8} ; [ DW_TAG_member ] [b] [line 1, size 32, align 32, offset 64] [from int]
!15 = !{!16}
!16 = !{!"0x101\00s\0016777219\000", !4, !5, !9} ; [ DW_TAG_arg_variable ] [s] [line 3]
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 1, !"Debug Info Version", i32 2}
!19 = !{!"clang version 3.5 "}
!20 = !{!"0x101\00s\0016777219\000", !4, !5, !9} ; [ DW_TAG_arg_variable ] [s] [line 3]
!21 = !MDLocation(line: 3, scope: !4)
!22 = !{!"0x101\00s\0016777219\000", !4, !5, !9} ; [ DW_TAG_arg_variable ] [s] [line 3]
!23 = !MDLocation(line: 4, scope: !4)
!24 = !{!"0x102\006\00157\000\0064"} ; [ DW_TAG_expression ] [DW_OP_bit_piece 0 64] [piece, size 64, offset 0]
!25 = !{}
; This expression has elements after DW_OP_bit_piece.
; CHECK: DIExpression does not Verify
!27 = !{!"0x102\00157\0064\0032\006"} ; [ DW_TAG_expression ] [DW_OP_bit_piece 64 32] [piece, size 32, offset 64]
