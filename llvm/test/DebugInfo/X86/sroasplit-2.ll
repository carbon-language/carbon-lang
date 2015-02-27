; RUN: opt %s -sroa -verify -S -o - | FileCheck %s
;
; Test that we can partial emit debug info for aggregates repeatedly
; split up by SROA.
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

; Verify that SROA creates a variable piece when splitting i1.
; CHECK:  call void @llvm.dbg.value(metadata i64 %outer.coerce0, i64 0, metadata ![[O:[0-9]+]], metadata ![[PIECE1:[0-9]+]]),
; CHECK:  call void @llvm.dbg.value(metadata i64 %outer.coerce1, i64 0, metadata ![[O]], metadata ![[PIECE2:[0-9]+]]),
; CHECK:  call void @llvm.dbg.value({{.*}}, i64 0, metadata ![[I1:[0-9]+]], metadata ![[PIECE3:[0-9]+]]),
; CHECK-DAG: ![[O]] = {{.*}} [ DW_TAG_arg_variable ] [outer] [line 10]
; CHECK-DAG: ![[PIECE1]] = {{.*}} [ DW_TAG_expression ] [DW_OP_bit_piece offset=0, size=64]
; CHECK-DAG: ![[PIECE2]] = {{.*}} [ DW_TAG_expression ] [DW_OP_bit_piece offset=64, size=64]
; CHECK-DAG: ![[I1]] = {{.*}} [ DW_TAG_auto_variable ] [i1] [line 11]
; CHECK-DAG: ![[PIECE3]] = {{.*}} [ DW_TAG_expression ] [DW_OP_bit_piece offset=0, size=32]

; ModuleID = 'sroasplit-2.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.Outer = type { [2 x %struct.Inner] }
%struct.Inner = type { i32, i32 }

; Function Attrs: nounwind ssp uwtable
define i32 @foo(i64 %outer.coerce0, i64 %outer.coerce1) #0 {
  %outer = alloca %struct.Outer, align 8
  %i1 = alloca %struct.Inner, align 4
  %1 = bitcast %struct.Outer* %outer to { i64, i64 }*
  %2 = getelementptr { i64, i64 }, { i64, i64 }* %1, i32 0, i32 0
  store i64 %outer.coerce0, i64* %2
  %3 = getelementptr { i64, i64 }, { i64, i64 }* %1, i32 0, i32 1
  store i64 %outer.coerce1, i64* %3
  call void @llvm.dbg.declare(metadata %struct.Outer* %outer, metadata !24, metadata !2), !dbg !25
  call void @llvm.dbg.declare(metadata %struct.Inner* %i1, metadata !26, metadata !2), !dbg !27
  %4 = getelementptr inbounds %struct.Outer, %struct.Outer* %outer, i32 0, i32 0, !dbg !27
  %5 = getelementptr inbounds [2 x %struct.Inner], [2 x %struct.Inner]* %4, i32 0, i64 1, !dbg !27
  %6 = bitcast %struct.Inner* %i1 to i8*, !dbg !27
  %7 = bitcast %struct.Inner* %5 to i8*, !dbg !27
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %6, i8* %7, i64 8, i32 4, i1 false), !dbg !27
  %8 = getelementptr inbounds %struct.Inner, %struct.Inner* %i1, i32 0, i32 0, !dbg !28
  %9 = load i32* %8, align 4, !dbg !28
  ret i32 %9, !dbg !28
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture, i8* nocapture readonly, i64, i32, i1) #2

attributes #0 = { nounwind ssp uwtable "no-frame-pointer-elim"="true" }
attributes #1 = { nounwind readnone }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!21, !22}
!llvm.ident = !{!23}

!0 = !{!"0x11\0012\00clang version 3.5.0 \000\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [ DW_TAG_compile_unit ] [sroasplit-2.c] [DW_LANG_C99]
!1 = !{!"sroasplit-2.c", !""}
!2 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!3 = !{!4}
!4 = !{!"0x2e\00foo\00foo\00\0010\000\001\000\006\00256\000\0010", !1, !5, !6, null, i32 (i64, i64)* @foo, null, null, !2} ; [ DW_TAG_subprogram ] [ DW_TAG_subprogram ] [line 10] [def] [foo]
!5 = !{!"0x29", !1} ; [ DW_TAG_file_type ] [ DW_TAG_file_type ] [sroasplit-2.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8, !9}
!8 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = !{!"0x16\00Outer\008\000\000\000\000", !1, null, !10} ; [ DW_TAG_typedef ] [ DW_TAG_typedef ] [Outer] [line 8, size 0, align 0, offset 0] [from ]
!10 = !{!"0x13\00\006\00128\0032\000\000\000", !1, null, null, !11, null, null, null} ; [ DW_TAG_structure_type ] [line 6, size 128, align 32, offset 0] [def] [from ]
!11 = !{!12}
!12 = !{!"0xd\00inner\007\00128\0032\000\000", !1, !10, !13} ; [ DW_TAG_member ] [inner] [line 7, size 128, align 32, offset 0] [from ]
!13 = !{!"0x1\00\000\00128\0032\000\000", null, null, !14, !19, i32 0, null, null, null} ; [ DW_TAG_array_type ] [line 0, size 128, align 32, offset 0] [from Inner]
!14 = !{!"0x16\00Inner\004\000\000\000\000", !1, null, !15} ; [ DW_TAG_typedef ] [ DW_TAG_typedef ] [Inner] [line 4, size 0, align 0, offset 0] [from ]
!15 = !{!"0x13\00\001\0064\0032\000\000\000", !1, null, null, !16, null, null, null} ; [ DW_TAG_structure_type ] [line 1, size 64, align 32, offset 0] [def] [from ]
!16 = !{!17, !18}
!17 = !{!"0xd\00a\002\0032\0032\000\000", !1, !15, !8} ; [ DW_TAG_member ] [a] [line 2, size 32, align 32, offset 0] [from int]
!18 = !{!"0xd\00b\003\0032\0032\0032\000", !1, !15, !8} ; [ DW_TAG_member ] [b] [line 3, size 32, align 32, offset 32] [from int]
!19 = !{!20}
!20 = !{!"0x21\000\002"}        ; [ DW_TAG_subrange_type ] [0, 1]
!21 = !{i32 2, !"Dwarf Version", i32 2}
!22 = !{i32 1, !"Debug Info Version", i32 2}
!23 = !{!"clang version 3.5.0 "}
!24 = !{!"0x101\00outer\0016777226\000", !4, !5, !9} ; [ DW_TAG_arg_variable ] [ DW_TAG_arg_variable ] [outer] [line 10]
!25 = !MDLocation(line: 10, scope: !4)
!26 = !{!"0x100\00i1\0011\000", !4, !5, !14} ; [ DW_TAG_auto_variable ] [ DW_TAG_auto_variable ] [i1] [line 11]
!27 = !MDLocation(line: 11, scope: !4)
!28 = !MDLocation(line: 12, scope: !4)
