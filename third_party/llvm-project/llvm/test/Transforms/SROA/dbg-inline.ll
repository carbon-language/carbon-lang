; Test that SROA can deal with allocas that have more than one
; dbg.declare hanging off of it.

; RUN: opt < %s -passes=sroa -S | FileCheck %s
source_filename = "/tmp/inlinesplit.cpp"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.15.0"

%struct.pair = type { i64, i64 }

; Function Attrs: noinline optnone ssp uwtable
define i64 @_Z1g4pair(i64 %p.coerce0, i64 %p.coerce1) #0 !dbg !8 {
entry:
  %p = alloca %struct.pair, align 8
  %0 = getelementptr inbounds %struct.pair, %struct.pair* %p, i32 0, i32 0
  store i64 %p.coerce0, i64* %0, align 8
  %1 = getelementptr inbounds %struct.pair, %struct.pair* %p, i32 0, i32 1
  store i64 %p.coerce1, i64* %1, align 8
  ; CHECK-DAG: call void @llvm.dbg.value(metadata i64 %p.coerce0, metadata ![[VAR:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg ![[LOC:[0-9]+]]
  ; CHECK-DAG: call void @llvm.dbg.value(metadata i64 %p.coerce1, metadata ![[VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg ![[LOC]]
  ; CHECK-DAG: call void @llvm.dbg.value(metadata i64 %p.coerce0, metadata ![[INLINED_VAR:[0-9]+]], metadata !DIExpression(DW_OP_LLVM_fragment, 0, 64)), !dbg ![[INLINED_LOC:[0-9]+]]
  ; CHECK-DAG: call void @llvm.dbg.value(metadata i64 %p.coerce1, metadata ![[INLINED_VAR]], metadata !DIExpression(DW_OP_LLVM_fragment, 64, 64)), !dbg ![[INLINED_LOC]]
  call void @llvm.dbg.declare(metadata %struct.pair* %p, metadata !17, metadata !DIExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata %struct.pair* %p, metadata !21, metadata !DIExpression()), !dbg !23
  %a.i = getelementptr inbounds %struct.pair, %struct.pair* %p, i32 0, i32 0, !dbg !25
  %x2 = load i64, i64* %0, align 8, !dbg !25
  ret i64 %x2, !dbg !26
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #2

attributes #0 = { noinline ssp uwtable }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { argmemonly nounwind willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 12.0.0 (git@github.com:llvm/llvm-project 5110fd0343c2d06c8ae538741fbef13ece5e68de)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None, sysroot: "/")
!1 = !DIFile(filename: "/tmp/inlinesplit.cpp", directory: "/Volumes/Data/llvm-project")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!8 = distinct !DISubprogram(name: "g", linkageName: "_Z1g4pair", scope: !9, file: !9, line: 9, type: !10, scopeLine: 9, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "/tmp/inlinesplit.cpp", directory: "")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !13}
!12 = !DIBasicType(name: "long long unsigned int", size: 64, encoding: DW_ATE_unsigned)
!13 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "pair", file: !9, line: 1, size: 128, flags: DIFlagTypePassByValue, elements: !14, identifier: "_ZTS4pair")
!14 = !{!15, !16}
!15 = !DIDerivedType(tag: DW_TAG_member, name: "a", scope: !13, file: !9, line: 1, baseType: !12, size: 64)
!16 = !DIDerivedType(tag: DW_TAG_member, name: "b", scope: !13, file: !9, line: 1, baseType: !12, size: 64, offset: 64)
!17 = !DILocalVariable(name: "p", arg: 1, scope: !8, file: !9, line: 9, type: !13)
; CHECK: ![[LOC]] = !DILocation
; CHECK-NOT: inlinedAt
; CHECK: =
!18 = !DILocation(line: 9, column: 27, scope: !8)
!19 = !DILocation(line: 10, column: 12, scope: !8)
!20 = !DILocation(line: 10, column: 10, scope: !8)
!21 = !DILocalVariable(name: "p", arg: 1, scope: !22, file: !9, line: 5, type: !13)
!22 = distinct !DISubprogram(name: "f", linkageName: "_ZL1f4pair", scope: !9, file: !9, line: 5, type: !10, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !2)
; CHECK: ![[INLINED_LOC]] = !DILocation({{.*}}inlinedAt
!23 = !DILocation(line: 5, column: 27, scope: !22, inlinedAt: !24)
!24 = distinct !DILocation(line: 10, column: 10, scope: !8)
!25 = !DILocation(line: 6, column: 12, scope: !22, inlinedAt: !24)
!26 = !DILocation(line: 10, column: 3, scope: !8)
