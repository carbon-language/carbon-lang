; Test that coverage instrumentation does not lose debug location.

; RUN: opt < %s -asan -asan-module -asan-coverage=1 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readnone uwtable
define void @_Z1fv() #0 {
entry:
  ret void, !dbg !11
}

; CHECK: call void @__sanitizer_cov(), !dbg !

attributes #0 = { sanitize_address nounwind readnone uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 (208682)", i1 true, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp//tmp/1.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"/tmp/1.cc", metadata !"/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"f", metadata !"f", metadata !"_Z1fv", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, void ()* @_Z1fv, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [f]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp//tmp/1.cc]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!10 = metadata !{metadata !"clang version 3.5.0 (208682)"}
!11 = metadata !{i32 2, i32 0, metadata !4, null}
