; RUN: opt -disable-output -S -passes=print-ir-similarity < %s 2>&1 | FileCheck %s

; When a debug instruction is the first instruction in a block, when that block
; has not been given a canonical numbering, since debug instructions are not
; counted in similarity matching they must be ignored when creating canonical
; relations from one region to another.  This checks that this is enforced.

; CHECK: 2 candidates of length 3.  Found in: 
; CHECK-NEXT:   Function: main, Basic Block: entry
; CHECK-NEXT:     Start Instruction:   br label %for.body169
; CHECK-NEXT:       End Instruction:   %1 = sub i32 1, 4
; CHECK-NEXT:   Function: main, Basic Block: for.body169
; CHECK-NEXT:     Start Instruction:   br label %for.end122
; CHECK-NEXT:       End Instruction:   %3 = sub i32 1, 4
; CHECK-NEXT: 2 candidates of length 2.  Found in: 
; CHECK-NEXT:   Function: main, Basic Block: for.end122
; CHECK-NEXT:     Start Instruction:   store i32 30, ptr undef, align 1
; CHECK-NEXT:       End Instruction:   %1 = sub i32 1, 4
; CHECK-NEXT:   Function: main, Basic Block: for.end246
; CHECK-NEXT:     Start Instruction:   store i32 0, ptr undef, align 1
; CHECK-NEXT:       End Instruction:   %3 = sub i32 1, 4
; CHECK-NEXT: 2 candidates of length 4.  Found in: 
; CHECK-NEXT:   Function: main, Basic Block: entry
; CHECK-NEXT:     Start Instruction:   %0 = add i32 1, 4
; CHECK-NEXT:       End Instruction:   %1 = sub i32 1, 4
; CHECK-NEXT:   Function: main, Basic Block: for.body169
; CHECK-NEXT:     Start Instruction:   %2 = add i32 1, 4
; CHECK-NEXT:       End Instruction:   %3 = sub i32 1, 4

source_filename = "irsimilarity_crash.ll"

@v_13 = external dso_local global ptr, align 1

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #0

define dso_local i16 @main() {
entry:
  %0 = add i32 1, 4
  br label %for.body169

for.end122:                                       ; preds = %for.cond108
  store i32 30, ptr undef, align 1
  %1 = sub i32 1, 4
  ret i16 1

for.body169:                                      ; preds = %for.cond167
  %2 = add i32 1, 4
  br label %for.end122

for.end246:                                     ; preds = %for.cond167
  call void @llvm.dbg.declare(metadata ptr undef, metadata !1, metadata !DIExpression()), !dbg !11
  store i32 0, ptr undef, align 1
  %3 = sub i32 1, 4
  unreachable
}

attributes #0 = { nocallback nofree nosync nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{}
!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DILocalVariable(name: "v_68", scope: !2, file: !3, line: 522, type: !10)
!2 = distinct !DILexicalBlock(scope: !4, file: !3, line: 522, column: 9)
!3 = !DIFile(filename: "41097217.c", directory: "rt.outdir")
!4 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 480, type: !5, scopeLine: 481, spFlags: DISPFlagDefinition, unit: !8, retainedNodes: !9)
!5 = !DISubroutineType(types: !6)
!6 = !{!7}
!7 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
!8 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 15.0.0.prerel", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, retainedTypes: !9, globals: !9, splitDebugInlining: false, nameTableKind: None)
!9 = !{}
!10 = !DIBasicType(name: "long", size: 32, encoding: DW_ATE_signed)
!11 = !DILocation(line: 522, column: 23, scope: !2)
