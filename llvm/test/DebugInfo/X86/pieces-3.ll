; RUN: llc %s -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s
;
;    // Compile with -O1
;    typedef struct {
;      int a;
;      int b;
;    } Inner;
;
;    typedef struct {
;      Inner inner[2];
;    } Outer;
;
;    int foo(Outer outer) {
;      Inner i1 = outer.inner[1];
;      return i1.a;
;    }
;
; CHECK: DW_TAG_formal_parameter [3]
; CHECK-NEXT:   DW_AT_location [DW_FORM_data4]        ([[LOC:.*]])
; CHECK-NEXT:   DW_AT_name {{.*}}"outer"
; CHECK: DW_TAG_variable
;                                                 rsi, piece 0x00000004
; CHECK-NEXT:   DW_AT_location [DW_FORM_block1]       {{.*}} 54 93 04
; CHECK-NEXT:   "i1"
;
; CHECK: .debug_loc
; CHECK: [[LOC]]:
; CHECK: Beginning address offset: 0x0000000000000000
; CHECK:    Ending address offset: 0x0000000000000004
; rdi, piece 0x00000008, piece 0x00000004, rsi, piece 0x00000004
; CHECK: Location description: 55 93 08 93 04 54 93 04 
;
; ModuleID = '/Volumes/Data/llvm/test/DebugInfo/X86/sroasplit-2.ll'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i64 %outer.coerce0, i64 %outer.coerce1) #0 {
  call void @llvm.dbg.value(metadata !{i64 %outer.coerce0}, i64 0, metadata !24, metadata !25), !dbg !26
  call void @llvm.dbg.declare(metadata !{null}, metadata !27, metadata !28), !dbg !26
  call void @llvm.dbg.value(metadata !{i64 %outer.coerce1}, i64 0, metadata !29, metadata !30), !dbg !26
  call void @llvm.dbg.declare(metadata !{null}, metadata !31, metadata !32), !dbg !26
  %outer.sroa.1.8.extract.trunc = trunc i64 %outer.coerce1 to i32, !dbg !33
  call void @llvm.dbg.value(metadata !{i32 %outer.sroa.1.8.extract.trunc}, i64 0, metadata !34, metadata !35), !dbg !33
  %outer.sroa.1.12.extract.shift = lshr i64 %outer.coerce1, 32, !dbg !33
  %outer.sroa.1.12.extract.trunc = trunc i64 %outer.sroa.1.12.extract.shift to i32, !dbg !33
  call void @llvm.dbg.value(metadata !{i64 %outer.sroa.1.12.extract.shift}, i64 0, metadata !34, metadata !35), !dbg !33
  call void @llvm.dbg.value(metadata !{i32 %outer.sroa.1.12.extract.trunc}, i64 0, metadata !34, metadata !35), !dbg !33
  call void @llvm.dbg.declare(metadata !{null}, metadata !34, metadata !35), !dbg !33
  ret i32 %outer.sroa.1.8.extract.trunc, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "no-frame-pointer-elim"="true" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/sroasplit-2.c] [DW_LANG_C99]
!1 = metadata !{metadata !"sroasplit-2.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00foo\00foo\00\0010\000\001\000\006\00256\000\0010", metadata !1, metadata !5, metadata !6, null, i32 (i64, i64)* @foo, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 10] [def] [foo]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/sroasplit-2.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !9}
!8 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !"0x16\00Outer\008\000\000\000\000", metadata !1, null, metadata !10} ; [ DW_TAG_typedef ] [Outer] [line 8, size 0, align 0, offset 0] [from ]
!10 = metadata !{metadata !"0x13\00\006\00128\0032\000\000\000", metadata !1, null, null, metadata !11, null, null, null} ; [ DW_TAG_structure_type ] [line 6, size 128, align 32, offset 0] [def] [from ]
!11 = metadata !{metadata !12}
!12 = metadata !{metadata !"0xd\00inner\007\00128\0032\000\000", metadata !1, metadata !10, metadata !13} ; [ DW_TAG_member ] [inner] [line 7, size 128, align 32, offset 0] [from ]
!13 = metadata !{metadata !"0x1\00\000\00128\0032\000\000", null, null, metadata !14, metadata !19, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 32, offset 0] [from Inner]
!14 = metadata !{metadata !"0x16\00Inner\004\000\000\000\000", metadata !1, null, metadata !15} ; [ DW_TAG_typedef ] [Inner] [line 4, size 0, align 0, offset 0] [from ]
!15 = metadata !{metadata !"0x13\00\001\0064\0032\000\000\000", metadata !1, null, null, metadata !16, null, null, null} ; [ DW_TAG_structure_type ] [line 1, size 64, align 32, offset 0] [def] [from ]
!16 = metadata !{metadata !17, metadata !18}
!17 = metadata !{metadata !"0xd\00a\002\0032\0032\000\000", metadata !1, metadata !15, metadata !8} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!18 = metadata !{metadata !"0xd\00b\003\0032\0032\0032\000", metadata !1, metadata !15, metadata !8} ; [ DW_TAG_member ] [b] [line 3, size 32, align 32, offset 32] [from int]
!19 = metadata !{metadata !20}
!20 = metadata !{metadata !"0x21\000\002"}        ; [ DW_TAG_subrange_type ] [0, 1]
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!23 = metadata !{metadata !"clang version 3.5.0 "}
!24 = metadata !{metadata !"0x101\00outer\0016777226\000", metadata !4, metadata !5, metadata !9} ; [ DW_TAG_arg_variable ] [outer] [line 10]
!25 = metadata !{metadata !"0x102\00147\000\008"} ; [ DW_TAG_expression ] [DW_OP_piece 0 8] [piece, size 8, offset 0]
!26 = metadata !{i32 10, i32 0, metadata !4, null}
!27 = metadata !{metadata !"0x101\00outer\0016777226\000", metadata !4, metadata !5, metadata !9} ; [ DW_TAG_arg_variable ] [outer] [line 10]
!28 = metadata !{metadata !"0x102\00147\008\008"} ; [ DW_TAG_expression ] [DW_OP_piece 8 8] [piece, size 8, offset 8]
!29 = metadata !{metadata !"0x101\00outer\0016777226\000", metadata !4, metadata !5, metadata !9} ; [ DW_TAG_arg_variable ] [outer] [line 10]
!30 = metadata !{metadata !"0x102\00147\0012\004"} ; [ DW_TAG_expression ] [DW_OP_piece 12 4] [piece, size 4, offset 12]
!31 = metadata !{metadata !"0x101\00outer\0016777226\000", metadata !4, metadata !5, metadata !9} ; [ DW_TAG_arg_variable ] [outer] [line 10]
!32 = metadata !{metadata !"0x102\00147\008\004"} ; [ DW_TAG_expression ] [DW_OP_piece 8 4] [piece, size 4, offset 8]
!33 = metadata !{i32 11, i32 0, metadata !4, null}
!34 = metadata !{metadata !"0x100\00i1\0011\000", metadata !4, metadata !5, metadata !14} ; [ DW_TAG_auto_variable ] [i1] [line 11]
!35 = metadata !{metadata !"0x102\00147\000\004"} ; [ DW_TAG_expression ] [DW_OP_piece 0 4] [piece, size 4, offset 0]
!36 = metadata !{i32 12, i32 0, metadata !4, null}
