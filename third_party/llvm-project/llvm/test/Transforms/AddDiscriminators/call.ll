; RUN: opt < %s -add-discriminators -S | FileCheck %s
; RUN: opt < %s -passes=add-discriminators -S | FileCheck %s

; Discriminator support for calls that are defined in one line:
; #1 void bar();
; #2
; #3 void foo() {
; #4  bar();bar()/*discriminator 2*/;bar()/*discriminator 4*/;
; #5 }

; Function Attrs: uwtable
define void @_Z3foov() #0 !dbg !4 {
  call void @_Z3barv(), !dbg !10
; CHECK:  call void @_Z3barv(), !dbg ![[CALL0:[0-9]+]]
  %a = alloca [100 x i8], align 16
  %b = bitcast [100 x i8]* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 100, i8* %b), !dbg !11
  call void @llvm.lifetime.end.p0i8(i64 100, i8* %b), !dbg !11
  call void @_Z3barv(), !dbg !11
; CHECK:  call void @_Z3barv(), !dbg ![[CALL1:[0-9]+]]
  call void @_Z3barv(), !dbg !12
; CHECK:  call void @_Z3barv(), !dbg ![[CALL2:[0-9]+]]
  ret void, !dbg !13
}

declare void @_Z3barv() #1
declare void @llvm.lifetime.start.p0i8(i64, i8* nocapture) nounwind argmemonly
declare void @llvm.lifetime.end.p0i8(i64, i8* nocapture) nounwind argmemonly

attributes #0 = { uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 3.8.0 (trunk 250915) (llvm/trunk 251830)", isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug, enums: !2)
!1 = !DIFile(filename: "c.cc", directory: "/tmp")
!2 = !{}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 3.8.0 (trunk 250915) (llvm/trunk 251830)"}
!10 = !DILocation(line: 4, column: 3, scope: !4)
!11 = !DILocation(line: 4, column: 9, scope: !4)
!12 = !DILocation(line: 4, column: 15, scope: !4)
!13 = !DILocation(line: 5, column: 1, scope: !4)

; CHECK: ![[CALL1]] = !DILocation(line: 4, column: 9, scope: ![[CALL1BLOCK:[0-9]+]])
; CHECK: ![[CALL1BLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 2)
; CHECK: ![[CALL2]] = !DILocation(line: 4, column: 15, scope: ![[CALL2BLOCK:[0-9]+]])
; CHECK: ![[CALL2BLOCK]] = !DILexicalBlockFile({{.*}} discriminator: 4)
