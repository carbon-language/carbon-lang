;; This test checks whether DW_AT_rank attribute accepts constants.
;; constants are interally stored as DIExpression.

; RUN: llc %s -mtriple=x86_64 -filetype=obj -o - | llvm-dwarfdump - | FileCheck %s

;; Test whether DW_AT_data_location is generated.
; CHECK-LABEL:  DW_TAG_array_type

; CHECK:  DW_AT_rank (DW_OP_push_object_address, DW_OP_plus_uconst 0x8, DW_OP_deref)
; CHECK:  DW_TAG_generic_subrange
; CHECK:  DW_AT_lower_bound     (-20)
; CHECK:  DW_AT_upper_bound     (0)
; CHECK:  DW_AT_byte_stride     (4)

;; Test case is hand written with the help of below testcase
;;------------------------------
;;subroutine sub(arank)
;;  real :: arank(..)
;;  print *, RANK(arank)
;;end
;;------------------------------

; ModuleID = 'dwarfdump-subrange_const.ll'
source_filename = "dwarfdump-subrange_const.ll"

define void @sub_(i64* noalias %arank, i64* noalias %"arank$sd") !dbg !5 {
L.entry:
  call void @llvm.dbg.value(metadata i64* %arank, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64* %"arank$sd", metadata !19, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata i64* %"arank$sd", metadata !29, metadata !DIExpression()), !dbg !18
  ret void, !dbg !18
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata)

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4)
!3 = !DIFile(filename: "generic_subrange_const.f90", directory: "/dir")
!4 = !{}
!5 = distinct !DISubprogram(name: "sub", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8, !14}
!8 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !10)
!9 = !DIBasicType(name: "real", size: 32, align: 32, encoding: DW_ATE_float)
!10 = !{!11}
!11 = !DISubrange(lowerBound: 1, upperBound: !12)
!12 = distinct !DILocalVariable(scope: !5, file: !3, type: !13, flags: DIFlagArtificial)
!13 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!14 = !DICompositeType(tag: DW_TAG_array_type, baseType: !13, size: 1024, align: 64, elements: !15)
!15 = !{!16}
!16 = !DISubrange(lowerBound: 1, upperBound: 16)
!17 = distinct !DILocalVariable(scope: !5, file: !3, type: !13, flags: DIFlagArtificial)
!18 = !DILocation(line: 0, scope: !5)
!19 = !DILocalVariable(name: "arank", arg: 1, scope: !5, file: !3, line: 1, type: !20)
!20 = !DICompositeType(tag: DW_TAG_array_type, baseType: !9, size: 32, align: 32, elements: !21, dataLocation: !17, rank: !DIExpression(DW_OP_push_object_address, DW_OP_plus_uconst, 8, DW_OP_deref))
!21 = !{!22}
!22 = !DIGenericSubrange(lowerBound: -20, upperBound: 0, stride: 4)
!29 = !DILocalVariable(arg: 2, scope: !5, file: !3, line: 1, type: !14, flags: DIFlagArtificial)
