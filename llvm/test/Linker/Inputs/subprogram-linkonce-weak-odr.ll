define weak_odr i32 @foo(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b, !dbg !DILocation(line: 2, scope: !3)
  ret i32 %sum, !dbg !DILocation(line: 3, scope: !3)
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

!llvm.dbg.cu = !{!1}
!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, subprograms: !{!3}, emissionKind: 1)
!2 = !DIFile(filename: "foo.c", directory: "/path/to/dir")
!3 = !DISubprogram(file: !4, scope: !4, line: 1, name: "foo", function: i32 (i32, i32)* @foo, type: !5)
!4 = !DIFile(filename: "foo.h", directory: "/path/to/dir")
!5 = !DISubroutineType(types: !{})
