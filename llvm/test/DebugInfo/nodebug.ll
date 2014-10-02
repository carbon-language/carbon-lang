; REQUIRES: object-emission

; RUN: %llc_dwarf < %s -filetype=obj | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Test that a nodebug function (a function not appearing in the debug info IR
; metadata subprogram list) with DebugLocs on its IR doesn't cause crashes/does
; the right thing.

; Build with clang from the following:
; extern int i;
; inline __attribute__((always_inline)) void f1() {
;   i = 3;
; }
;
; __attribute__((nodebug)) void f2() {
;   f1();
; }

; Check that there's only one DW_TAG_subprogram, nothing for the 'f2' function.
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "f1"
; CHECK-NOT: DW_TAG_subprogram

@i = external global i32

; Function Attrs: uwtable
define void @_Z2f2v() #0 {
entry:
  store i32 3, i32* @i, align 4, !dbg !11
  ret void
}

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/nodebug.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"nodebug.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f1", metadata !"f1", metadata !"_Z2f1v", i32 2, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, null, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [f1]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/nodebug.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!10 = metadata !{metadata !"clang version 3.5.0 "}
!11 = metadata !{i32 3, i32 0, metadata !4, null}
