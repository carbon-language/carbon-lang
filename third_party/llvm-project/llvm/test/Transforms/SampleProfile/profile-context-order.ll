;; Test for different function processing orders affecting inlining in sample profile loader.

;; There is an SCC _Z5funcAi -> _Z8funcLeafi -> _Z5funcAi in the program.
;; With -use-profiled-call-graph=0, the top-down processing order of
;; that SCC is (_Z8funcLeafi, _Z5funcAi), which is determinined based on
;; the static call graph. With -use-profiled-call-graph=1, call edges
;; from profile are considered, thus the order becomes (_Z5funcAi, _Z8funcLeafi)
;; which leads to _Z8funcLeafi inlined into _Z5funcAi.
; RUN: opt < %s -passes=sample-profile -use-profiled-call-graph=1 -sample-profile-file=%S/Inputs/profile-context-order.prof -S | FileCheck %s -check-prefix=INLINE
; RUN: opt < %s -passes=sample-profile -use-profiled-call-graph=0 -sample-profile-file=%S/Inputs/profile-context-order.prof -S | FileCheck %s -check-prefix=NOINLINE

;; There is an indirect call _Z5funcAi -> _Z3fibi in the program.
;; With -use-profiled-call-graph=0, the processing order computed
;; based on the static call graph is (_Z3fibi, _Z5funcAi). With 
;; -use-profiled-call-graph=1, the indirect call edge from profile is
;; considered, thus the order becomes (_Z5funcAi, _Z3fibi) which leads to
;; _Z3fibi inlined into _Z5funcAi.
; RUN: opt < %s -passes=sample-profile -use-profiled-call-graph=1 -sample-profile-file=%S/Inputs/profile-context-order.prof -S | FileCheck %s -check-prefix=ICALL-INLINE

@factor = dso_local global i32 3, align 4, !dbg !0
@fp = dso_local global i32 (i32)* null, align 8

define dso_local i32 @main() local_unnamed_addr #0 !dbg !18 {
entry:
  store i32 (i32)* @_Z3fibi, i32 (i32)** @fp, align 8, !dbg !25
  br label %for.body, !dbg !25

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add3, !dbg !27

for.body:                                         ; preds = %for.body, %entry
  %x.011 = phi i32 [ 300000, %entry ], [ %dec, %for.body ]
  %r.010 = phi i32 [ 0, %entry ], [ %add3, %for.body ]
  %call = tail call i32 @_Z5funcBi(i32 %x.011), !dbg !32
  %add = add nuw nsw i32 %x.011, 1, !dbg !31
  %call1 = tail call i32 @_Z5funcAi(i32 %add), !dbg !28
  %add2 = add i32 %call, %r.010, !dbg !34
  %add3 = add i32 %add2, %call1, !dbg !35
  %dec = add nsw i32 %x.011, -1, !dbg !36
  %cmp = icmp eq i32 %x.011, 0, !dbg !38
  br i1 %cmp, label %for.cond.cleanup, label %for.body, !dbg !25
}

; INLINE: define dso_local i32 @_Z5funcAi
; INLINE-NOT: call i32 @_Z8funcLeafi
; NOINLINE: define dso_local i32 @_Z5funcAi
; NOINLINE: call i32 @_Z8funcLeafi
; ICALL-INLINE: define dso_local i32 @_Z5funcAi
; ICALL-INLINE: call i32 @_Z3foo
define dso_local i32 @_Z5funcAi(i32 %x) local_unnamed_addr #0 !dbg !40 {
entry:
  %add = add nsw i32 %x, 100000, !dbg !44
  %0 = load i32 (i32)*, i32 (i32)** @fp, align 8
  %call = call i32 %0(i32 8), !dbg !45
  %call1 = tail call i32 @_Z8funcLeafi(i32 %add), !dbg !46
  ret i32 %call, !dbg !46
}

; INLINE: define dso_local i32 @_Z8funcLeafi
; NOINLINE: define dso_local i32 @_Z8funcLeafi
; ICALL-INLINE: define dso_local i32 @_Z8funcLeafi
; ICALL-NOINLINE: define dso_local i32 @_Z8funcLeafi
define dso_local i32 @_Z8funcLeafi(i32 %x) local_unnamed_addr #1 !dbg !54 {
entry:
  %cmp = icmp sgt i32 %x, 0, !dbg !57
  br i1 %cmp, label %while.body, label %while.cond2.preheader, !dbg !59

while.cond2.preheader:                            ; preds = %entry
  %cmp313 = icmp slt i32 %x, 0, !dbg !60
  br i1 %cmp313, label %while.body4, label %if.end, !dbg !63

while.body:                                       ; preds = %while.body, %entry
  %x.addr.016 = phi i32 [ %sub, %while.body ], [ %x, %entry ]
  %tmp = load volatile i32, i32* @factor, align 4, !dbg !64
  %call = tail call i32 @_Z5funcAi(i32 %tmp), !dbg !67
  %sub = sub nsw i32 %x.addr.016, %call, !dbg !68
  %cmp1 = icmp sgt i32 %sub, 0, !dbg !69
  br i1 %cmp1, label %while.body, label %if.end, !dbg !71

while.body4:                                      ; preds = %while.body4, %while.cond2.preheader
  %x.addr.114 = phi i32 [ %add, %while.body4 ], [ %x, %while.cond2.preheader ]
  %tmp1 = load volatile i32, i32* @factor, align 4, !dbg !72
  %call5 = tail call i32 @_Z5funcBi(i32 %tmp1), !dbg !74
  %add = add nsw i32 %call5, %x.addr.114, !dbg !75
  %cmp3 = icmp slt i32 %add, 0, !dbg !60
  br i1 %cmp3, label %while.body4, label %if.end, !dbg !63

if.end:                                           ; preds = %while.body4, %while.body, %while.cond2.preheader
  %x.addr.2 = phi i32 [ 0, %while.cond2.preheader ], [ %sub, %while.body ], [ %add, %while.body4 ]
  ret i32 %x.addr.2, !dbg !76
}

define dso_local i32 @_Z5funcBi(i32 %x) local_unnamed_addr #0 !dbg !47 {
entry:
  %sub = add nsw i32 %x, -100000, !dbg !51
  %call = tail call i32 @_Z8funcLeafi(i32 %sub), !dbg !52
  ret i32 %call, !dbg !53
}

define dso_local i32 @_Z3fibi(i32 %x) local_unnamed_addr #1 !dbg !77 {
entry:
  %sub = add nsw i32 %x, -100000, !dbg !78
  %call = tail call i32 @_Z3foo(i32 %sub), !dbg !78
  ret i32 %sub, !dbg !78
}

declare i32 @_Z3foo(i32)

attributes #0 = { nofree noinline nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }
attributes #1 = { nofree nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" "use-sample-profile" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15, !16}
!llvm.ident = !{!17}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "factor", scope: !2, file: !3, line: 21, type: !13, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 11.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !12, splitDebugInlining: false, debugInfoForProfiling: true, nameTableKind: None)
!3 = !DIFile(filename: "merged.cpp", directory: "/local/autofdo")
!4 = !{}
!5 = !{!6, !10, !11}
!6 = !DISubprogram(name: "funcA", linkageName: "_Z5funcAi", scope: !3, file: !3, line: 6, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!7 = !DISubroutineType(types: !8)
!8 = !{!9, !9}
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !DISubprogram(name: "funcB", linkageName: "_Z5funcBi", scope: !3, file: !3, line: 7, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!11 = !DISubprogram(name: "funcLeaf", linkageName: "_Z8funcLeafi", scope: !3, file: !3, line: 22, type: !7, flags: DIFlagPrototyped, spFlags: DISPFlagOptimized, retainedNodes: !4)
!12 = !{!0}
!13 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !9)
!14 = !{i32 7, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{i32 1, !"wchar_size", i32 4}
!17 = !{!"clang version 11.0.0"}
!18 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 11, type: !19, scopeLine: 11, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !21)
!19 = !DISubroutineType(types: !20)
!20 = !{!9}
!21 = !{!22, !23}
!22 = !DILocalVariable(name: "r", scope: !18, file: !3, line: 12, type: !9)
!23 = !DILocalVariable(name: "x", scope: !24, file: !3, line: 13, type: !9)
!24 = distinct !DILexicalBlock(scope: !18, file: !3, line: 13, column: 3)
!25 = !DILocation(line: 13, column: 3, scope: !26)
!26 = !DILexicalBlockFile(scope: !24, file: !3, discriminator: 2)
!27 = !DILocation(line: 17, column: 3, scope: !18)
!28 = !DILocation(line: 14, column: 10, scope: !29)
!29 = distinct !DILexicalBlock(scope: !30, file: !3, line: 13, column: 37)
!30 = distinct !DILexicalBlock(scope: !24, file: !3, line: 13, column: 3)
!31 = !DILocation(line: 14, column: 29, scope: !29)
!32 = !DILocation(line: 14, column: 21, scope: !33)
!33 = !DILexicalBlockFile(scope: !29, file: !3, discriminator: 2)
!34 = !DILocation(line: 14, column: 19, scope: !29)
!35 = !DILocation(line: 14, column: 7, scope: !29)
!36 = !DILocation(line: 13, column: 33, scope: !37)
!37 = !DILexicalBlockFile(scope: !30, file: !3, discriminator: 6)
!38 = !DILocation(line: 13, column: 26, scope: !39)
!39 = !DILexicalBlockFile(scope: !30, file: !3, discriminator: 2)
!40 = distinct !DISubprogram(name: "funcA", linkageName: "_Z5funcAi", scope: !3, file: !3, line: 26, type: !7, scopeLine: 26, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!44 = !DILocation(line: 26, column: 22, scope: !40)
!45 = !DILocation(line: 28, column: 11, scope: !40)
!46 = !DILocation(line: 27, column: 3, scope: !40)
!47 = distinct !DISubprogram(name: "funcB", linkageName: "_Z5funcBi", scope: !3, file: !3, line: 32, type: !7, scopeLine: 32, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!51 = !DILocation(line: 33, column: 22, scope: !47)
!52 = !DILocation(line: 33, column: 11, scope: !47)
!53 = !DILocation(line: 35, column: 3, scope: !47)
!54 = distinct !DISubprogram(name: "funcLeaf", linkageName: "_Z8funcLeafi", scope: !3, file: !3, line: 48, type: !7, scopeLine: 48, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!57 = !DILocation(line: 49, column: 9, scope: !58)
!58 = distinct !DILexicalBlock(scope: !54, file: !3, line: 49, column: 7)
!59 = !DILocation(line: 49, column: 7, scope: !54)
!60 = !DILocation(line: 58, column: 14, scope: !61)
!61 = !DILexicalBlockFile(scope: !62, file: !3, discriminator: 2)
!62 = distinct !DILexicalBlock(scope: !58, file: !3, line: 56, column: 8)
!63 = !DILocation(line: 58, column: 5, scope: !61)
!64 = !DILocation(line: 52, column: 16, scope: !65)
!65 = distinct !DILexicalBlock(scope: !66, file: !3, line: 51, column: 19)
!66 = distinct !DILexicalBlock(scope: !58, file: !3, line: 49, column: 14)
!67 = !DILocation(line: 52, column: 12, scope: !65)
!68 = !DILocation(line: 52, column: 9, scope: !65)
!69 = !DILocation(line: 51, column: 14, scope: !70)
!70 = !DILexicalBlockFile(scope: !66, file: !3, discriminator: 2)
!71 = !DILocation(line: 51, column: 5, scope: !70)
!72 = !DILocation(line: 59, column: 16, scope: !73)
!73 = distinct !DILexicalBlock(scope: !62, file: !3, line: 58, column: 19)
!74 = !DILocation(line: 59, column: 12, scope: !73)
!75 = !DILocation(line: 59, column: 9, scope: !73)
!76 = !DILocation(line: 63, column: 3, scope: !54)
!77 = distinct !DISubprogram(name: "funcB", linkageName: "_Z3fibi", scope: !3, file: !3, line: 32, type: !7, scopeLine: 32, flags: DIFlagPrototyped | DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2)
!78 = !DILocation(line: 33, column: 22, scope: !77)
