; RUN: opt %s -sroa -verify -S -o - | FileCheck %s
;
; Test that we can partial emit debug info for aggregates repeatedly
; split up by SROA.
;
;    // Compile with -O1
;    typedef struct {
;      int a;
;      long int b;
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

; Verify that SROA creates a variable piece when splitting i1.
; CHECK: %[[I1:.*]] = alloca [12 x i8], align 4
; CHECK: call void @llvm.dbg.declare(metadata [12 x i8]* %[[I1]], metadata ![[VAR:[0-9]+]], metadata ![[PIECE1:[0-9]+]])
; CHECK: call void @llvm.dbg.value(metadata i32 %[[A:.*]], i64 0, metadata ![[VAR]], metadata ![[PIECE2:[0-9]+]])
; CHECK: ret i32 %[[A]]
; Read Var and Piece:
; CHECK: ![[VAR]] = {{.*}} ; [ DW_TAG_auto_variable ] [i1] [line 11]
; CHECK: ![[PIECE1]] = {{.*}} ; [ DW_TAG_expression ] [DW_OP_bit_piece offset=32, size=96]
; CHECK: ![[PIECE2]] = {{.*}} ; [ DW_TAG_expression ] [DW_OP_bit_piece offset=0, size=32]

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.Outer = type { [2 x %struct.Inner] }
%struct.Inner = type { i32, i64 }

; Function Attrs: nounwind ssp uwtable
define i32 @foo(%struct.Outer* byval align 8 %outer) #0 {
entry:
  %i1 = alloca %struct.Inner, align 8
  call void @llvm.dbg.declare(metadata %struct.Outer* %outer, metadata !25, metadata !2), !dbg !26
  call void @llvm.dbg.declare(metadata %struct.Inner* %i1, metadata !27, metadata !2), !dbg !28
  %inner = getelementptr inbounds %struct.Outer, %struct.Outer* %outer, i32 0, i32 0, !dbg !28
  %arrayidx = getelementptr inbounds [2 x %struct.Inner], [2 x %struct.Inner]* %inner, i32 0, i64 1, !dbg !28
  %0 = bitcast %struct.Inner* %i1 to i8*, !dbg !28
  %1 = bitcast %struct.Inner* %arrayidx to i8*, !dbg !28
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %0, i8* %1, i64 16, i32 8, i1 false), !dbg !28
  %a = getelementptr inbounds %struct.Inner, %struct.Inner* %i1, i32 0, i32 0, !dbg !29
  %2 = load i32* %a, align 4, !dbg !29
  ret i32 %2, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!22, !23}
!llvm.ident = !{!24}

!0 = !{!"0x11\0012\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [ DW_TAG_compile_unit ] [sroasplit-1.c] [DW_LANG_C99]
!1 = !{!"sroasplit-1.c", !""}
!2 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\0010\000\001\000\006\00256\000\0010", !1, !5, !6, null, i32 (%struct.Outer*)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [ DW_TAG_subprogram ] [line 10] [def] [foo]
!5 = !{!"0x29", !1} ; [ DW_TAG_file_type ] [ DW_TAG_file_type ] [sroasplit-1.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !9}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0x16\00Outer\008\000\000\000\000", !1, null, !10} ; [ DW_TAG_typedef ] [ DW_TAG_typedef ] [Outer] [line 8, size 0, align 0, offset 0] [from ]
!10 = !{!"0x13\00\006\00256\0064\000\000\000", !1, null, null, !11, null, null, null} ; [ DW_TAG_structure_type ] [line 6, size 256, align 64, offset 0] [def] [from ]
!11 = !{!12}
!12 = !{!"0xd\00inner\007\00256\0064\000\000", !1, !10, !13} ; [ DW_TAG_member ] [inner] [line 7, size 256, align 64, offset 0] [from ]
!13 = !{!"0x1\00\000\00256\0064\000\000", null, null, !14, !20, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 256, align 64, offset 0] [from Inner]
!14 = !{!"0x16\00Inner\004\000\000\000\000", !1, null, !15} ; [ DW_TAG_typedef ] [ DW_TAG_typedef ] [Inner] [line 4, size 0, align 0, offset 0] [from ]
!15 = !{!"0x13\00\001\00128\0064\000\000\000", !1, null, null, !16, null, null, null} ; [ DW_TAG_structure_type ] [line 1, size 128, align 64, offset 0] [def] [from ]
!16 = !{!17, !18}
!17 = !{!"0xd\00a\002\0032\0032\000\000", !1, !15, !8} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!18 = !{!"0xd\00b\003\0064\0064\0064\000", !1, !15, !19} ; [ DW_TAG_member ] [b] [line 3, size 64, align 64, offset 64] [from long int]
!19 = !{!"0x24\00long int\000\0064\0064\000\000\005", null, null} ; [ DW_TAG_base_type ] [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!20 = !{!21}
!21 = !{!"0x21\000\002"}        ; [ DW_TAG_subrange_type ] [0, 1]
!22 = !{i32 2, !"Dwarf Version", i32 2}
!23 = !{i32 1, !"Debug Info Version", i32 2}
!24 = !{!"clang version 3.5.0 "}
!25 = !{!"0x101\00outer\0016777226\000", !4, !5, !9} ; [ DW_TAG_arg_variable ] [ DW_TAG_arg_variable ] [outer] [line 10]
!26 = !MDLocation(line: 10, scope: !4)
!27 = !{!"0x100\00i1\0011\000", !4, !5, !14} ; [ DW_TAG_auto_variable ] [ DW_TAG_auto_variable ] [i1] [line 11]
!28 = !MDLocation(line: 11, scope: !4)
!29 = !MDLocation(line: 12, scope: !4)
