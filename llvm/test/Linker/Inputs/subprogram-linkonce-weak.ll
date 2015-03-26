define weak i32 @foo(i32 %a, i32 %b) {
entry:
  %sum = call i32 @fastadd(i32 %a, i32 %b), !dbg !MDLocation(line: 52, scope: !3)
  ret i32 %sum, !dbg !MDLocation(line: 53, scope: !3)
}

declare i32 @fastadd(i32, i32)

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!1}
!1 = !MDCompileUnit(language: DW_LANG_C99, file: !2, subprograms: !{!3}, emissionKind: 1)
!2 = !MDFile(filename: "foo.c", directory: "/path/to/dir")
!3 = !MDSubprogram(file: !2, scope: !2, line: 51, name: "foo", function: i32 (i32, i32)* @foo, type: !4)
!4 = !MDSubroutineType(types: !{})
