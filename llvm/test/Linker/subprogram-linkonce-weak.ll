; RUN: llvm-link %s %S/Inputs/subprogram-linkonce-weak.ll -S -o %t1
; RUN: FileCheck %s -check-prefix=LW -check-prefix=CHECK <%t1
; RUN: llvm-link %S/Inputs/subprogram-linkonce-weak.ll %s -S -o %t2
; RUN: FileCheck %s -check-prefix=WL -check-prefix=CHECK <%t2

; This testcase tests the following flow:
;  - File A defines a linkonce version of @foo which has inlined into @bar.
;  - File B defines a weak version of @foo (different definition).
;  - Linkage rules state File B version of @foo wins.
;  - @bar still has inlined debug info related to the linkonce @foo.
;
; This should fix PR22792, although the testcase was hand-written.  There's a
; RUN line with a crasher for llc at the end with checks for the DWARF output.

; The LW prefix means linkonce (this file) first, then weak (the other file).
; The WL prefix means weak (the other file) first, then linkonce (this file).

; We'll see @bar before @foo if this file is first.
; LW-LABEL: define i32 @bar(
; LW: %sum = add i32 %a, %b, !dbg ![[FOOINBAR:[0-9]+]]
; LW: ret i32 %sum, !dbg ![[BARRET:[0-9]+]]
; LW-LABEL: define weak i32 @foo(
; LW: %sum = call i32 @fastadd(i32 %a, i32 %b), !dbg ![[FOOCALL:[0-9]+]]
; LW: ret i32 %sum, !dbg ![[FOORET:[0-9]+]]

; We'll see @foo before @bar if this file is second.
; WL-LABEL: define weak i32 @foo(
; WL: %sum = call i32 @fastadd(i32 %a, i32 %b), !dbg ![[FOOCALL:[0-9]+]]
; WL: ret i32 %sum, !dbg ![[FOORET:[0-9]+]]
; WL-LABEL: define i32 @bar(
; WL: %sum = add i32 %a, %b, !dbg ![[FOOINBAR:[0-9]+]]
; WL: ret i32 %sum, !dbg ![[BARRET:[0-9]+]]

define i32 @bar(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b, !dbg !MDLocation(line: 2, scope: !4,
                                          inlinedAt: !MDLocation(line: 12, scope: !3))
  ret i32 %sum, !dbg !MDLocation(line: 13, scope: !3)
}

define linkonce i32 @foo(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b, !dbg !MDLocation(line: 2, scope: !4)
  ret i32 %sum, !dbg !MDLocation(line: 3, scope: !4)
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK-LABEL: !llvm.dbg.cu =
; LW-SAME: !{![[LCU:[0-9]+]], ![[WCU:[0-9]+]]}
; WL-SAME: !{![[WCU:[0-9]+]], ![[LCU:[0-9]+]]}
!llvm.dbg.cu = !{!1}

; LW: ![[LCU]] = !MDCompileUnit({{.*}} subprograms: ![[LSPs:[0-9]+]]
; LW: ![[LSPs]] = !{![[BARSP:[0-9]+]], ![[FOOSP:[0-9]+]]}
; LW: ![[BARSP]] = !MDSubprogram(name: "bar",
; LW-SAME: function: i32 (i32, i32)* @bar
; LW: ![[FOOSP]] = {{.*}}!MDSubprogram(name: "foo",
; LW-NOT: function:
; LW-SAME: ){{$}}
; LW: ![[WCU]] = !MDCompileUnit({{.*}} subprograms: ![[WSPs:[0-9]+]]
; LW: ![[WSPs]] = !{![[WEAKFOOSP:[0-9]+]]}
; LW: ![[WEAKFOOSP]] = !MDSubprogram(name: "foo",
; LW-SAME: function: i32 (i32, i32)* @foo
; LW: ![[FOOINBAR]] = !MDLocation(line: 2, scope: ![[FOOSP]], inlinedAt: ![[BARIA:[0-9]+]])
; LW: ![[BARIA]] = !MDLocation(line: 12, scope: ![[BARSP]])
; LW: ![[BARRET]] = !MDLocation(line: 13, scope: ![[BARSP]])
; LW: ![[FOOCALL]] = !MDLocation(line: 2, scope: ![[WEAKFOOSP]])
; LW: ![[FOORET]] = !MDLocation(line: 3, scope: ![[WEAKFOOSP]])

; Same as above, but reordered.
; WL: ![[WCU]] = !MDCompileUnit({{.*}} subprograms: ![[WSPs:[0-9]+]]
; WL: ![[WSPs]] = !{![[WEAKFOOSP:[0-9]+]]}
; WL: ![[WEAKFOOSP]] = !MDSubprogram(name: "foo",
; WL-SAME: function: i32 (i32, i32)* @foo
; WL: ![[LCU]] = !MDCompileUnit({{.*}} subprograms: ![[LSPs:[0-9]+]]
; WL: ![[LSPs]] = !{![[BARSP:[0-9]+]], ![[FOOSP:[0-9]+]]}
; WL: ![[BARSP]] = !MDSubprogram(name: "bar",
; WL-SAME: function: i32 (i32, i32)* @bar
; WL: ![[FOOSP]] = {{.*}}!MDSubprogram(name: "foo",
; Note, for symmetry, this should be "NOT: function:" and "SAME: ){{$}}".
; WL-SAME: function: i32 (i32, i32)* @foo
; WL: ![[FOOCALL]] = !MDLocation(line: 2, scope: ![[WEAKFOOSP]])
; WL: ![[FOORET]] = !MDLocation(line: 3, scope: ![[WEAKFOOSP]])
; WL: ![[FOOINBAR]] = !MDLocation(line: 2, scope: ![[FOOSP]], inlinedAt: ![[BARIA:[0-9]+]])
; WL: ![[BARIA]] = !MDLocation(line: 12, scope: ![[BARSP]])
; WL: ![[BARRET]] = !MDLocation(line: 13, scope: ![[BARSP]])

!1 = !MDCompileUnit(language: DW_LANG_C99, file: !2, subprograms: !{!3, !4}, emissionKind: 2)
!2 = !MDFile(filename: "bar.c", directory: "/path/to/dir")
!3 = !MDSubprogram(file: !2, scope: !2, line: 11, name: "bar", function: i32 (i32, i32)* @bar, type: !5)
!4 = !MDSubprogram(file: !2, scope: !2, line: 1, name: "foo", function: i32 (i32, i32)* @foo, type: !5)
!5 = !MDSubroutineType(types: !{})

; Crasher for llc.
; REQUIRES: object-emission
; RUN: %llc_dwarf -filetype=obj -O0 %t1 -o %t1.o
; RUNDISABLED: llvm-dwarfdump %t1.o -debug-dump=info | FileCheck %s -check-prefix=DWLW
; RUN: %llc_dwarf -filetype=obj -O0 %t2 -o %t2.o
; RUNDISABLED: llvm-dwarfdump %t2.o -debug-dump=info | FileCheck %s -check-prefix=DWWL
; Getting different dwarfdump output on different platforms, so I've
; temporarily disabled the Dwarf FileChecks while leaving in the crash tests.
; I'll keep using PR22792 to track this.

; DWLW:     DW_TAG_compile_unit
; DWLW:       DW_AT_name {{.*}}"bar.c"
; DWLW:       DW_TAG_subprogram
; DWLW-NOT:     DW_AT_{{[lowhigh]*}}_pc
; DWLW:         DW_AT_name {{.*}}foo
; DWLW-NOT:     DW_AT_{{[lowhigh]*}}_pc
; DWLW:       DW_TAG_subprogram
; DWLW:         DW_AT_low_pc
; DWLW:         DW_AT_high_pc
; DWLW:         DW_AT_name {{.*}}bar
; DWLW:         DW_TAG_inlined_subroutine
; DWLW:           DW_AT_abstract_origin
; DWLW:     DW_TAG_compile_unit
; DWLW:       DW_AT_name {{.*}}"foo.c"
; DWLW:       DW_TAG_subprogram
; DWLW:         DW_AT_low_pc
; DWLW:         DW_AT_high_pc
; DWLW:         DW_AT_name {{.*}}foo

; The DWARF output is already symmetric (just reordered).
; DWWL:     DW_TAG_compile_unit
; DWWL:       DW_AT_name {{.*}}"foo.c"
; DWWL:       DW_TAG_subprogram
; DWWL:         DW_AT_low_pc
; DWWL:         DW_AT_high_pc
; DWWL:         DW_AT_name {{.*}}foo
; DWWL:     DW_TAG_compile_unit
; DWWL:       DW_AT_name {{.*}}"bar.c"
; DWWL:       DW_TAG_subprogram
; DWWL-NOT:     DW_AT_{{[lowhigh]*}}_pc
; DWWL:         DW_AT_name {{.*}}foo
; DWWL-NOT:     DW_AT_{{[lowhigh]*}}_pc
; DWWL:       DW_TAG_subprogram
; DWWL:         DW_AT_low_pc
; DWWL:         DW_AT_high_pc
; DWWL:         DW_AT_name {{.*}}bar
; DWWL:         DW_TAG_inlined_subroutine
; DWWL:           DW_AT_abstract_origin
