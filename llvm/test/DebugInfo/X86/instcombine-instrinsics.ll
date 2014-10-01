; RUN: opt %s -O2 -S -o - | FileCheck %s
; Verify that we emit the same intrinsic at most once.
; rdar://problem/13056109
;
; CHECK: call void @llvm.dbg.value(metadata !{%struct.i14** %p}
; CHECK-NOT: call void @llvm.dbg.value(metadata !{%struct.i14** %p}
; CHECK-NEXT: call i32 @foo
; CHECK: ret
;
;
; typedef struct {
;   long i;
; } i14;
;
; int foo(i14**);
;
;   void init() {
;     i14* p = 0;
;     foo(&p);
;     p->i |= 4;
;     foo(&p);
;   }
;
; ModuleID = 'instcombine_intrinsics.c'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%struct.i14 = type { i64 }

; Function Attrs: nounwind ssp uwtable
define void @init() #0 {
  %p = alloca %struct.i14*, align 8
  call void @llvm.dbg.declare(metadata !{%struct.i14** %p}, metadata !11), !dbg !18
  store %struct.i14* null, %struct.i14** %p, align 8, !dbg !18
  %1 = call i32 @foo(%struct.i14** %p), !dbg !19
  %2 = load %struct.i14** %p, align 8, !dbg !20
  %3 = getelementptr inbounds %struct.i14* %2, i32 0, i32 0, !dbg !20
  %4 = load i64* %3, align 8, !dbg !20
  %5 = or i64 %4, 4, !dbg !20
  store i64 %5, i64* %3, align 8, !dbg !20
  %6 = call i32 @foo(%struct.i14** %p), !dbg !21
  ret void, !dbg !22
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

declare i32 @foo(%struct.i14**)

attributes #0 = { nounwind ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{i32 786449, metadata !1, i32 12, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [instcombine_intrinsics.c] [DW_LANG_C99]
!1 = metadata !{metadata !"instcombine_intrinsics.c", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"init", metadata !"init", metadata !"", i32 7, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 0, i1 false, void ()* @init, null, null, metadata !2, i32 7} ; [ DW_TAG_subprogram ] [line 7] [def] [init]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [instcombine_intrinsics.c]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!10 = metadata !{metadata !"clang version 3.5.0 "}
!11 = metadata !{i32 786688, metadata !4, metadata !"p", metadata !5, i32 8, metadata !12, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [p] [line 8]
!12 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !13} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from i14]
!13 = metadata !{i32 786454, metadata !1, null, metadata !"i14", i32 3, i64 0, i64 0, i64 0, i32 0, metadata !14} ; [ DW_TAG_typedef ] [i14] [line 3, size 0, align 0, offset 0] [from ]
!14 = metadata !{i32 786451, metadata !1, null, metadata !"", i32 1, i64 64, i64 64, i32 0, i32 0, null, metadata !15, i32 0, null, null, null} ; [ DW_TAG_structure_type ] [line 1, size 64, align 64, offset 0] [def] [from ]
!15 = metadata !{metadata !16}
!16 = metadata !{i32 786445, metadata !1, metadata !14, metadata !"i", i32 2, i64 64, i64 64, i64 0, i32 0, metadata !17} ; [ DW_TAG_member ] [i] [line 2, size 64, align 64, offset 0] [from long int]
!17 = metadata !{i32 786468, null, null, metadata !"long int", i32 0, i64 64, i64 64, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [long int] [line 0, size 64, align 64, offset 0, enc DW_ATE_signed]
!18 = metadata !{i32 8, i32 0, metadata !4, null} ; [ DW_TAG_imported_declaration ]
!19 = metadata !{i32 9, i32 0, metadata !4, null}
!20 = metadata !{i32 10, i32 0, metadata !4, null}
!21 = metadata !{i32 11, i32 0, metadata !4, null}
!22 = metadata !{i32 12, i32 0, metadata !4, null}
