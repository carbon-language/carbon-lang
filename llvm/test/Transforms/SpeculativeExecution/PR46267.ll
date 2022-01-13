; RUN: opt < %s -S -passes='speculative-execution' | FileCheck %s

%class.B = type { i32 (...)** }

; Testing that two bitcasts are not hoisted to the first BB
define i8* @foo(%class.B* readonly %b) {
; CHECK-LABEL: foo
; CHECK-LABEL: entry
; CHECK-NEXT: %i = icmp eq %class.B* %b, null
; CHECK-NEXT: br i1 %i, label %end, label %notnull
entry:
  %i = icmp eq %class.B* %b, null
  br i1 %i, label %end, label %notnull

; CHECK-LABEL: notnull:
; CHECK-NEXT: %i1 = bitcast %class.B* %b to i32**
; CHECK: %i3 = bitcast %class.B* %b to i8*
notnull:                             ; preds = %entry
  %i1 = bitcast %class.B* %b to i32**
  %vtable = load i32*, i32** %i1, align 8
  %i2 = getelementptr inbounds i32, i32* %vtable, i64 -2
  %offset.to.top = load i32, i32* %i2, align 4
  %i3 = bitcast %class.B* %b to i8*
  %i4 = sext i32 %offset.to.top to i64
  %i5 = getelementptr inbounds i8, i8* %i3, i64 %i4
  br label %end

end:                                 ; preds = %notnull, %entry
  %i6 = phi i8* [ %i5, %notnull ], [ null, %entry ]
  ret i8* %i6
}

define void @f(i32 %i) {
entry:
; CHECK-LABEL: @f(
; CHECK:  %a2 = add i32 %i, 0
; CHECK-NEXT:  call void @llvm.dbg.value(metadata i32 %a2
  br i1 undef, label %land.rhs, label %land.end

land.rhs:                                         ; preds = %entry
; CHECK: land.rhs:
; CHECK-NEXT: call void @llvm.dbg.label
; CHECK-NEXT: %x = alloca i32, align 4
; CHECK-NEXT: call void @llvm.dbg.addr(metadata i32* %x
; CHECK-NEXT: %y = alloca i32, align 4
; CHECK-NEXT: call void @llvm.dbg.declare(metadata i32* %y
; CHECK-NEXT: %a0 = load i32, i32* undef, align 1
; CHECK-NEXT: call void @llvm.dbg.value(metadata i32 %a0
  call void @llvm.dbg.label(metadata !11), !dbg !10
  %x = alloca i32, align 4
  call void @llvm.dbg.addr(metadata i32* %x, metadata !12, metadata !DIExpression()), !dbg !10
  %y = alloca i32, align 4
  call void @llvm.dbg.declare(metadata i32* %y, metadata !14, metadata !DIExpression()), !dbg !10

  %a0 = load i32, i32* undef, align 1
  call void @llvm.dbg.value(metadata i32 %a0, metadata !9, metadata !DIExpression()), !dbg !10

  %a2 = add i32 %i, 0
  call void @llvm.dbg.value(metadata i32 %a2, metadata !13, metadata !DIExpression()), !dbg !10

  br label %land.end

land.end:                                         ; preds = %land.rhs, %entry
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare void @llvm.dbg.value(metadata, metadata, metadata) #1
declare void @llvm.dbg.label(metadata)
declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.addr(metadata, metadata, metadata)

attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 11.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !3, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "foo.c", directory: "/bar")
!2 = !{}
!3 = !{!4}
!4 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !7, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !2)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !DILocalVariable(name: "a0", scope: !6, file: !1, line: 3, type: !4)
!10 = !DILocation(line: 0, scope: !6)
!11 = !DILabel(scope: !6, name: "label", file: !1, line: 1)
!12 = !DILocalVariable(name: "x", scope: !6, file: !1, line: 3, type: !4)
!13 = !DILocalVariable(name: "a2", scope: !6, file: !1, line: 3, type: !4)
!14 = !DILocalVariable(name: "y", scope: !6, file: !1, line: 3, type: !4)
