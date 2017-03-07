; RUN: llvm-as < %s -o %t
; RUN: llvm-dis < %t -o - | FileCheck %s
; Created at -02 from:
; bool alpha(int);
; bool bravo(int charlie) { return (alpha(charlie)); }
; static int delta(int charlie) { return charlie + 1; }
; __attribute__((nodebug)) bool echo(int foxtrot) {
;   return (bravo(delta(foxtrot)));
; }

source_filename = "t.c"
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.12.0"

define zeroext i1 @_Z5bravoi(i32 %charlie) local_unnamed_addr #0 !dbg !7 {
entry:
  tail call void @llvm.dbg.value(metadata i32 %charlie, i64 0, metadata !13, metadata !14), !dbg !15
  %call = tail call zeroext i1 @_Z5alphai(i32 %charlie), !dbg !16
  ret i1 %call, !dbg !17
}

declare zeroext i1 @_Z5alphai(i32) local_unnamed_addr

define zeroext i1 @_Z4echoi(i32 %foxtrot) local_unnamed_addr #0 {
entry:
; This should not set off the FnArg Verifier. The two variables are in differrent scopes.
  tail call void @llvm.dbg.value(metadata i32 %foxtrot, i64 0, metadata !18, metadata !14), !dbg !23
  %add.i = add nsw i32 %foxtrot, 1, !dbg !24
  tail call void @llvm.dbg.value(metadata i32 %add.i, i64 0, metadata !13, metadata !14), !dbg !15
  %call.i = tail call zeroext i1 @_Z5alphai(i32 %add.i), !dbg !16
  ret i1 %call.i
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #2

attributes #0 = { ssp uwtable }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 5.0.0 (trunk 297153) (llvm/trunk 297155)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"PIC Level", i32 2}
!6 = !{!"clang version 5.0.0 (trunk 297153) (llvm/trunk 297155)"}
!7 = distinct !DISubprogram(name: "bravo", linkageName: "_Z5bravoi", scope: !1, file: !1, line: 2, type: !8, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !12)
!8 = !DISubroutineType(types: !9)
!9 = !{!10, !11}
!10 = !DIBasicType(name: "bool", size: 8, encoding: DW_ATE_boolean)
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
; CHECK: !DILocalVariable(name: "charlie", arg: 1
!13 = !DILocalVariable(name: "charlie", arg: 1, scope: !7, file: !1, line: 2, type: !11)
!14 = !DIExpression()
!15 = !DILocation(line: 2, column: 16, scope: !7)
!16 = !DILocation(line: 2, column: 35, scope: !7)
!17 = !DILocation(line: 2, column: 27, scope: !7)
; CHECK: !DILocalVariable(name: "charlie", arg: 1
!18 = !DILocalVariable(name: "charlie", arg: 1, scope: !19, file: !1, line: 3, type: !11)
!19 = distinct !DISubprogram(name: "delta", linkageName: "_ZL5deltai", scope: !1, file: !1, line: 3, type: !20, isLocal: true, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: true, unit: !0, variables: !22)
!20 = !DISubroutineType(types: !21)
!21 = !{!11, !11}
!22 = !{!18}
!23 = !DILocation(line: 3, column: 22, scope: !19)
!24 = !DILocation(line: 3, column: 48, scope: !19)
