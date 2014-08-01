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
;                                                 rsi, piece 0x00000004, bit-piece 32 0
; CHECK-NEXT:   DW_AT_location [DW_FORM_block1]       (<0x06> 54 93 04 9d 20 00 )
; CHECK-NEXT:   "i1"

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
  call void @llvm.dbg.value(metadata !{i64 %outer.coerce0}, i64 0, metadata !24), !dbg !26
  call void @llvm.dbg.declare(metadata !{null}, metadata !27), !dbg !26
  call void @llvm.dbg.value(metadata !{i64 %outer.coerce1}, i64 0, metadata !29), !dbg !26
  call void @llvm.dbg.declare(metadata !{null}, metadata !31), !dbg !26
  %outer.sroa.1.8.extract.trunc = trunc i64 %outer.coerce1 to i32, !dbg !33
  call void @llvm.dbg.value(metadata !{i32 %outer.sroa.1.8.extract.trunc}, i64 0, metadata !34), !dbg !33
  %outer.sroa.1.12.extract.shift = lshr i64 %outer.coerce1, 32, !dbg !33
  %outer.sroa.1.12.extract.trunc = trunc i64 %outer.sroa.1.12.extract.shift to i32, !dbg !33
  call void @llvm.dbg.value(metadata !{i32 %outer.sroa.1.12.extract.trunc}, i64 0, metadata !34), !dbg !33
  call void @llvm.dbg.declare(metadata !{null}, metadata !34), !dbg !33
  ret i32 %outer.sroa.1.8.extract.trunc, !dbg !36
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { nounwind ssp uwtable "no-frame-pointer-elim"="true" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/sroasplit-2.c] [DW_LANG_C99]
!1 = metadata !{metadata !"sroasplit-2.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"", i32 10, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i64, i64)* @foo, null, null, metadata !2, i32 10} ; [ DW_TAG_subprogram ] [line 10] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/sroasplit-2.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !9}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786454, metadata !1, null, metadata !"Outer", i32 8, i64 0, i64 0, i64 0, i32 0, metadata !10} ; [ DW_TAG_typedef ] [Outer] [line 8, size 0, align 0, offset 0] [from ]
!10 = metadata !{i32 786451, metadata !1, null, metadata !"", i32 6, i64 128, i64 32, i32 0, i32 0, null, metadata !11, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [line 6, size 128, align 32, offset 0] [def] [from ]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786445, metadata !1, metadata !10, metadata !"inner", i32 7, i64 128, i64 32, i64 0, i32 0, metadata !13} ; [ DW_TAG_member ] [inner] [line 7, size 128, align 32, offset 0] [from ]
!13 = metadata !{i32 786433, null, null, metadata !"", i32 0, i64 128, i64 32, i32 0, i32 0, metadata !14, metadata !19, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 32, offset 0] [from Inner]
!14 = metadata !{i32 786454, metadata !1, null, metadata !"Inner", i32 4, i64 0, i64 0, i64 0, i32 0, metadata !15} ; [ DW_TAG_typedef ] [Inner] [line 4, size 0, align 0, offset 0] [from ]
!15 = metadata !{i32 786451, metadata !1, null, metadata !"", i32 1, i64 64, i64 32, i32 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [line 1, size 64, align 32, offset 0] [def] [from ]
!16 = metadata !{metadata !17, metadata !18}
!17 = metadata !{i32 786445, metadata !1, metadata !15, metadata !"a", i32 2, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!18 = metadata !{i32 786445, metadata !1, metadata !15, metadata !"b", i32 3, i64 32, i64 32, i64 32, i32 0, metadata !8} ; [ DW_TAG_member ] [b] [line 3, size 32, align 32, offset 32] [from int]
!19 = metadata !{metadata !20}
!20 = metadata !{i32 786465, i64 0, i64 2}        ; [ DW_TAG_subrange_type ] [0, 1]
!21 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!22 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!23 = metadata !{metadata !"clang version 3.5.0 "}
!24 = metadata !{i32 786689, metadata !4, metadata !"outer", metadata !5, i32 16777226, metadata !9, i32 0, i32 0, metadata !25} ; [ DW_TAG_arg_variable ] [outer] [line 10] [piece, size 8, offset 0]
!25 = metadata !{i32 3, i32 0, i32 8}
!26 = metadata !{i32 10, i32 0, metadata !4, null}
!27 = metadata !{i32 786689, metadata !4, metadata !"outer", metadata !5, i32 16777226, metadata !9, i32 0, i32 0, metadata !28} ; [ DW_TAG_arg_variable ] [outer] [line 10] [piece, size 8, offset 8]
!28 = metadata !{i32 3, i32 8, i32 8}
!29 = metadata !{i32 786689, metadata !4, metadata !"outer", metadata !5, i32 16777226, metadata !9, i32 0, i32 0, metadata !30} ; [ DW_TAG_arg_variable ] [outer] [line 10] [piece, size 4, offset 12]
!30 = metadata !{i32 3, i32 12, i32 4}
!31 = metadata !{i32 786689, metadata !4, metadata !"outer", metadata !5, i32 16777226, metadata !9, i32 0, i32 0, metadata !32} ; [ DW_TAG_arg_variable ] [outer] [line 10] [piece, size 4, offset 8]
!32 = metadata !{i32 3, i32 8, i32 4}
!33 = metadata !{i32 11, i32 0, metadata !4, null}
!34 = metadata !{i32 786688, metadata !4, metadata !"i1", metadata !5, i32 11, metadata !14, i32 0, i32 0, metadata !35} ; [ DW_TAG_auto_variable ] [i1] [line 11] [piece, size 4, offset 0]
!35 = metadata !{i32 3, i32 0, i32 4}
!36 = metadata !{i32 12, i32 0, metadata !4, null}
