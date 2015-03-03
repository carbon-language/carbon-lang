; RUN: echo '!9 = !{!"%/T/linkagename.ll", !0}' > %t1
; RUN: cat %s %t1 > %t2
; RUN: opt -insert-gcov-profiling -disable-output < %t2
; RUN: grep _Z3foov %T/linkagename.gcno
; RUN: rm %T/linkagename.gcno

define void @_Z3foov() {
entry:
  ret void, !dbg !8
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!10}
!llvm.gcov = !{!9}

!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.3 (trunk 177323)", isOptimized: false, emissionKind: 0, file: !2, enums: !3, retainedTypes: !3, subprograms: !4, globals: !3, imports:  !3)
!1 = !MDFile(filename: "hello.cc", directory: "/home/nlewycky")
!2 = !MDFile(filename: "hello.cc", directory: "/home/nlewycky")
!3 = !{i32 0}
!4 = !{!5}
!5 = !MDSubprogram(name: "foo", linkageName: "_Z3foov", line: 1, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 1, file: !1, scope: !1, type: !6, function: void ()* @_Z3foov, variables: !3)
!6 = !MDSubroutineType(types: !7)
!7 = !{null}
!8 = !MDLocation(line: 1, scope: !5)


!10 = !{i32 1, !"Debug Info Version", i32 3}
