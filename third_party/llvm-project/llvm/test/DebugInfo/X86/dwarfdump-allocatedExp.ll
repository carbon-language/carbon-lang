; RUN: llc %s -O2 -filetype=obj -o %t.o
; RUN: llvm-dwarfdump  %t.o | FileCheck %s

; Test whether DW_AT_data_location is generated.
; CHECK-LABEL:  DW_TAG_array_type
; CHECK:        DW_AT_allocated (DW_OP_push_object_address, DW_OP_deref)
; CHECK-NOT: DW_TAG
; CHECK:       DW_TAG_subrange_type

; Test case is hand written with the help of below testcase
;------------------------------
;program main
;integer, allocatable :: arr(:)
;allocate(arr(2:20))
;arr(2)=99
;print *, arr
;end program main
;------------------------------

; ModuleID = 'allocated.ll'
source_filename = "allocated.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @MAIN_() !dbg !5 {
L.entry:
  %.Z0640_333 = alloca i32*, align 8
  %"arr$sd1_349" = alloca [16 x i64], align 8
  call void @llvm.dbg.declare(metadata [16 x i64]* %"arr$sd1_349", metadata !8, metadata !DIExpression()), !dbg !13
  call void @llvm.dbg.declare(metadata i32** %.Z0640_333, metadata !14, metadata !DIExpression()), !dbg !13
  ret void, !dbg !16
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "allocated.f90", directory: "/dir")
!4 = !{}
!5 = distinct !DISubprogram(name: "main", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagMainSubprogram, unit: !2)
!6 = !DISubroutineType(cc: DW_CC_program, types: !7)
!7 = !{null}
!8 = !DILocalVariable(name: "arr", scope: !5, file: !3, type: !9)
!9 = !DICompositeType(tag: DW_TAG_array_type, baseType: !10, size: 32, align: 32, elements: !11, dataLocation: !DIExpression(DW_OP_push_object_address, DW_OP_deref), allocated: !DIExpression(DW_OP_push_object_address, DW_OP_deref))
!10 = !DIBasicType(name: "integer", size: 32, align: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DISubrange(count: 19, lowerBound: 2)
!13 = !DILocation(line: 0, scope: !5)
!14 = distinct !DILocalVariable(scope: !5, file: !3, type: !15, flags: DIFlagArtificial)
!15 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !10, size: 32, align: 32)
!16 = !DILocation(line: 6, column: 1, scope: !5)
