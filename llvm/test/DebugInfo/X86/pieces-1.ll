; RUN: llc -O0 %s -filetype=obj -o %t.o
; RUN: llvm-dwarfdump -debug-dump=loc %t.o | FileCheck %s
;
; rdar://problem/15928306
;
; Test that we can emit debug info for aggregate values that are split
; up across multiple registers by SROA.
;
;    // Compile with -O1.
;    typedef struct { long int a; int b;} S;
;
;    int foo(S s) {
;            return s.b;
;    }
;
;
; CHECK: .debug_loc contents:
;

; 0x0000000000000000 - 0x0000000000000006: rdi, piece 0x00000008, rsi, piece 0x00000004
; CHECK:            Beginning address offset: 0x0000000000000000
; CHECK:               Ending address offset: [[LTMP3:.*]]
; CHECK:                Location description: 55 93 08 54 93 04
; 0x0000000000000006 - 0x0000000000000008: rbp-8, piece 0x00000008, rax, piece 0x00000004 )
; CHECK:            Beginning address offset: [[LTMP3]]
; CHECK:               Ending address offset: [[END:.*]]
; CHECK:                Location description: 76 78 93 08 54 93 04

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i64 %s.coerce0, i32 %s.coerce1) #0 {
entry:
  call void @llvm.dbg.value(metadata !{i64 %s.coerce0}, i64 0, metadata !20), !dbg !21
  call void @llvm.dbg.value(metadata !{i32 %s.coerce1}, i64 0, metadata !22), !dbg !21
  ret i32 %s.coerce1, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind ssp uwtable "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5 ", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ]
!1 = metadata !{metadata !"pieces.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"", i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (i64, i32)* @foo, null, null, metadata !15, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/pieces.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !9}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786454, metadata !1, null, metadata !"S", i32 1, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_typedef ] [S] [line 1, size 0, align 0, offset 0] [from ]
!10 = metadata !{i32 786451, metadata !1, null, metadata !"", i32 1, i64 128, i64 64, i32 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [line 1, size 128, align 64, offset 0] [def] [from ]
!11 = metadata !{metadata !12, metadata !14}
!12 = metadata !{i32 786445, metadata !1, metadata !10, metadata !"a", i32 1, i64 64, i64 64, i64 0, i32 0, metadata !13} ; [ DW_TAG_member ] [a] [line 1, size 64, align 64, offset 0] [from long int]
!13 = metadata !{i32 786468, null, null, metadata !"long int", i32 0, i64 64, i64 64, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!14 = metadata !{i32 786445, metadata !1, metadata !10, metadata !"b", i32 1, i64 32, i64 32, i64 64, i32 0, metadata !8} ; [ DW_TAG_member ] [b] [line 1, size 32, align 32, offset 64] [from int]
!15 = metadata !{metadata !16}
!16 = metadata !{i32 786689, metadata !4, metadata !"s", metadata !5, i32 16777219, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [s] [line 3]
!17 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!18 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!19 = metadata !{metadata !"clang version 3.5 "}
!20 = metadata !{i32 786689, metadata !4, metadata !"s", metadata !5, i32 16777219, metadata !9, i32 0, i32 0, metadata !24} ; [ DW_TAG_arg_variable ] [s] [line 3] [piece, size 8, offset 0]
!21 = metadata !{i32 3, i32 0, metadata !4, null}
!22 = metadata !{i32 786689, metadata !4, metadata !"s", metadata !5, i32 16777219, metadata !9, i32 0, i32 0, metadata !27} ; [ DW_TAG_arg_variable ] [s] [line 3] [piece, size 4, offset 8]
!23 = metadata !{i32 4, i32 0, metadata !4, null}
!24 = metadata !{i64 3, i64 0, i64 8}
!25 = metadata !{}
!27 = metadata !{i64 3, i64 8, i64 4}
