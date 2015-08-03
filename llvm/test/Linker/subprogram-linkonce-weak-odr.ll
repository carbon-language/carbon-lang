; RUN: llvm-link %s %S/Inputs/subprogram-linkonce-weak-odr.ll -S -o %t1
; RUN: FileCheck %s -check-prefix=LW -check-prefix=CHECK <%t1
; RUN: llvm-link %S/Inputs/subprogram-linkonce-weak-odr.ll %s -S -o %t2
; RUN: FileCheck %s -check-prefix=WL -check-prefix=CHECK <%t2

; This testcase tests the following flow:
;  - File A defines a linkonce_odr version of @foo which has inlined into @bar.
;  - File B defines a weak_odr version of @foo (identical definition).
;  - Linkage rules state File B version of @foo wins.
;  - Debug info for the subprograms of @foo match exactly.  Without
;    intervention, the same subprogram would show up in both compile units, and
;    it would get associated with the compile unit where it was linkonce.
;  - @bar has inlined debug info related to the linkonce_odr @foo.
;
; This checks a corner case for the fix for PR22792, where subprograms match
; exactly.  It's a companion for subprogram-linkonce-weak.ll.

; The LW prefix means linkonce (this file) first, then weak (the other file).
; The WL prefix means weak (the other file) first, then linkonce (this file).

; We'll see @bar before @foo if this file is first.
; LW-LABEL: define i32 @bar(
; LW: %sum = add i32 %a, %b, !dbg ![[FOOINBAR:[0-9]+]]
; LW: ret i32 %sum, !dbg ![[BARRET:[0-9]+]]
; LW-LABEL: define weak_odr i32 @foo(
; LW: %sum = add i32 %a, %b, !dbg ![[FOOADD:[0-9]+]]
; LW: ret i32 %sum, !dbg ![[FOORET:[0-9]+]]

; We'll see @foo before @bar if this file is second.
; WL-LABEL: define weak_odr i32 @foo(
; WL: %sum = add i32 %a, %b, !dbg ![[FOOADD:[0-9]+]]
; WL: ret i32 %sum, !dbg ![[FOORET:[0-9]+]]
; WL-LABEL: define i32 @bar(
; WL: %sum = add i32 %a, %b, !dbg ![[FOOINBAR:[0-9]+]]
; WL: ret i32 %sum, !dbg ![[BARRET:[0-9]+]]

define i32 @bar(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b, !dbg !DILocation(line: 2, scope: !4,
                                          inlinedAt: !DILocation(line: 12, scope: !3))
  ret i32 %sum, !dbg !DILocation(line: 13, scope: !3)
}

define linkonce_odr i32 @foo(i32 %a, i32 %b) {
entry:
  %sum = add i32 %a, %b, !dbg !DILocation(line: 2, scope: !4)
  ret i32 %sum, !dbg !DILocation(line: 3, scope: !4)
}

!llvm.module.flags = !{!0}
!0 = !{i32 2, !"Debug Info Version", i32 3}

; CHECK-LABEL: !llvm.dbg.cu =
; LW-SAME: !{![[LCU:[0-9]+]], ![[WCU:[0-9]+]]}
; WL-SAME: !{![[WCU:[0-9]+]], ![[LCU:[0-9]+]]}
!llvm.dbg.cu = !{!1}

; LW: ![[LCU]] = distinct !DICompileUnit({{.*}} subprograms: ![[LSPs:[0-9]+]]
; LW: ![[LSPs]] = !{![[BARSP:[0-9]+]], ![[FOOSP:[0-9]+]]}
; LW: ![[BARSP]] = !DISubprogram(name: "bar",
; LW-SAME: function: i32 (i32, i32)* @bar
; LW: ![[FOOSP]] = {{.*}}!DISubprogram(name: "foo",
; LW-NOT: function:
; LW-SAME: ){{$}}
; LW: ![[WCU]] = distinct !DICompileUnit({{.*}} subprograms: ![[WSPs:[0-9]+]]
; LW: ![[WSPs]] = !{![[WEAKFOOSP:[0-9]+]]}
; LW: ![[WEAKFOOSP]] = !DISubprogram(name: "foo",
; LW-SAME: function: i32 (i32, i32)* @foo
; LW: ![[FOOINBAR]] = !DILocation(line: 2, scope: ![[FOOSP]], inlinedAt: ![[BARIA:[0-9]+]])
; LW: ![[BARIA]] = !DILocation(line: 12, scope: ![[BARSP]])
; LW: ![[BARRET]] = !DILocation(line: 13, scope: ![[BARSP]])
; LW: ![[FOOADD]] = !DILocation(line: 2, scope: ![[WEAKFOOSP]])
; LW: ![[FOORET]] = !DILocation(line: 3, scope: ![[WEAKFOOSP]])

; Same as above, but reordered.
; WL: ![[WCU]] = distinct !DICompileUnit({{.*}} subprograms: ![[WSPs:[0-9]+]]
; WL: ![[WSPs]] = !{![[WEAKFOOSP:[0-9]+]]}
; WL: ![[WEAKFOOSP]] = !DISubprogram(name: "foo",
; WL-SAME: function: i32 (i32, i32)* @foo
; WL: ![[LCU]] = distinct !DICompileUnit({{.*}} subprograms: ![[LSPs:[0-9]+]]
; Note: for symmetry, LSPs would have a different copy of the subprogram.
; WL: ![[LSPs]] = !{![[BARSP:[0-9]+]], ![[WEAKFOOSP:[0-9]+]]}
; WL: ![[BARSP]] = !DISubprogram(name: "bar",
; WL-SAME: function: i32 (i32, i32)* @bar
; WL: ![[FOOADD]] = !DILocation(line: 2, scope: ![[WEAKFOOSP]])
; WL: ![[FOORET]] = !DILocation(line: 3, scope: ![[WEAKFOOSP]])
; WL: ![[FOOINBAR]] = !DILocation(line: 2, scope: ![[WEAKFOOSP]], inlinedAt: ![[BARIA:[0-9]+]])
; WL: ![[BARIA]] = !DILocation(line: 12, scope: ![[BARSP]])
; WL: ![[BARRET]] = !DILocation(line: 13, scope: ![[BARSP]])

!1 = distinct !DICompileUnit(language: DW_LANG_C99, file: !2, subprograms: !{!3, !4}, emissionKind: 1)
!2 = !DIFile(filename: "bar.c", directory: "/path/to/dir")
!3 = !DISubprogram(file: !2, scope: !2, line: 11, name: "bar", function: i32 (i32, i32)* @bar, type: !6)
!4 = !DISubprogram(file: !5, scope: !5, line: 1, name: "foo", function: i32 (i32, i32)* @foo, type: !6)
!5 = !DIFile(filename: "foo.h", directory: "/path/to/dir")
!6 = !DISubroutineType(types: !{})

; Crasher for llc.
; REQUIRES: object-emission
; RUN: %llc_dwarf -filetype=obj -O0 %t1 -o %t1.o
; RUN: llvm-dwarfdump %t1.o -debug-dump=all | FileCheck %s -check-prefix=DWLW -check-prefix=DW
; RUN: %llc_dwarf -filetype=obj -O0 %t2 -o %t2.o
; RUN: llvm-dwarfdump %t2.o -debug-dump=all | FileCheck %s -check-prefix=DWWL -check-prefix=DW
; Check that the debug info puts the subprogram (with PCs) in the correct
; compile unit.

; DW-LABEL: .debug_info contents:
; DWLW:     DW_TAG_compile_unit
; DWLW:       DW_AT_name {{.*}}"bar.c"
; Note: If we stop emitting foo here, the comment below for DWWL (and the
; check) should be copied up here.
; DWLW:       DW_TAG_subprogram
; DWLW-NOT:     DW_AT_low_pc
; DWLW-NOT:     DW_AT_high_pc
; DWLW:         DW_AT_name {{.*}}foo
; DWLW:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}foo.h"
; DWLW:         DW_AT_decl_line {{.*}}(1)
; DWLW:       DW_TAG_subprogram
; DWLW:         DW_AT_low_pc
; DWLW:         DW_AT_high_pc
; DWLW:         DW_AT_name {{.*}}bar
; DWLW:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}bar.c"
; DWLW:         DW_AT_decl_line {{.*}}(11)
; DWLW:         DW_TAG_inlined_subroutine
; DWLW:           DW_AT_abstract_origin
; DWLW:     DW_TAG_compile_unit
; DWLW:       DW_AT_name {{.*}}"foo.c"
; DWLW:       DW_TAG_subprogram
; DWLW:         DW_AT_low_pc
; DWLW:         DW_AT_high_pc
; DWLW:         DW_AT_name {{.*}}foo
; DWLW:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}foo.h"
; DWLW:         DW_AT_decl_line {{.*}}(1)

; The DWARF output is already symmetric (just reordered).
; DWWL:     DW_TAG_compile_unit
; DWWL:       DW_AT_name {{.*}}"foo.c"
; DWWL:       DW_TAG_subprogram
; DWWL:         DW_AT_low_pc
; DWWL:         DW_AT_high_pc
; DWWL:         DW_AT_name {{.*}}foo
; DWWL:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}foo.h"
; DWWL:         DW_AT_decl_line {{.*}}(1)
; DWWL:     DW_TAG_compile_unit
; DWWL:       DW_AT_name {{.*}}"bar.c"
; Note: for symmetry, foo would also show up in this compile unit
; (alternatively, it wouldn't show up in the DWLW case).  If we start emitting
; foo here, this should be updated by checking that we don't emit low_pc and
; high_pc for it.
; DWWL-NOT:     DW_AT_name {{.*}}foo
; DWWL:       DW_TAG_subprogram
; DWWL-NOT:     DW_AT_name {{.*}}foo
; DWWL:         DW_AT_low_pc
; DWWL:         DW_AT_high_pc
; DWWL-NOT:     DW_AT_name {{.*}}foo
; DWWL:         DW_AT_name {{.*}}bar
; DWWL:         DW_AT_decl_file {{.*}}"/path/to/dir{{/|\\}}bar.c"
; DWWL:         DW_AT_decl_line {{.*}}(11)
; DWWL:         DW_TAG_inlined_subroutine
; DWWL:           DW_AT_abstract_origin

; DW-LABEL:   .debug_line contents:
; Check that we have the right things in the line table as well.

; DWLW-LABEL: file_names[{{ *}}1]{{.*}} bar.c
; DWLW-LABEL: file_names[{{ *}}2]{{.*}} foo.h
; DWLW:        2 0 2 0 0 is_stmt prologue_end
; DWLW-LABEL: file_names[{{ *}}1]{{.*}} foo.h
; DWLW:        2 0 1 0 0 is_stmt prologue_end
; DWLW-NOT:                      prologue_end

; DWWL-LABEL: file_names[{{ *}}1]{{.*}} foo.h
; DWWL:        2 0 1 0 0 is_stmt prologue_end
; DWWL-LABEL: file_names[{{ *}}1]{{.*}} bar.c
; DWWL-LABEL: file_names[{{ *}}2]{{.*}} foo.h
; DWWL:        2 0 2 0 0 is_stmt prologue_end
; DWWL-NOT:                      prologue_end
