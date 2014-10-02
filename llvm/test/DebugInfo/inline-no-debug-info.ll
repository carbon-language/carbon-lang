; RUN: opt < %s -inline -S | FileCheck %s

; This was generated from the following source:
; int a, b;
; __attribute__((__always_inline__)) static void callee2() { b = 2; }
; __attribute__((__nodebug__)) void callee() { a = 1; callee2(); }
; void caller() { callee(); }
; by running
;   clang -S test.c -emit-llvm -O1 -gline-tables-only -fno-strict-aliasing

; CHECK-LABEL: @caller(

; This instruction did not have a !dbg metadata in the callee.
; CHECK: store i32 1, {{.*}}, !dbg [[A:!.*]]

; This instruction came from callee with a !dbg metadata.
; CHECK: store i32 2, {{.*}}, !dbg [[B:!.*]]

; The remaining instruction from the caller.
; CHECK: ret void, !dbg [[A]]

; Debug location of the code in caller() and of the inlined code that did not
; have any debug location before.
; CHECK-DAG: [[A]] = metadata !{i32 4, i32 0, metadata !{{[01-9]+}}, null}

; Debug location of the inlined code.
; CHECK-DAG: [[B]] = metadata !{i32 2, i32 0, metadata !{{[01-9]+}}, metadata [[A]]}


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@a = common global i32 0, align 4
@b = common global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @callee() #0 {
entry:
  store i32 1, i32* @a, align 4
  store i32 2, i32* @b, align 4, !dbg !11
  ret void
}

; Function Attrs: nounwind uwtable
define void @caller() #0 {
entry:
  tail call void @callee(), !dbg !12
  ret void, !dbg !12
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!8, !9}
!llvm.ident = !{!10}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.5.0 (210174)\001\00\000\00\002", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/code/llvm/build0/test.c] [DW_LANG_C99]
!1 = metadata !{metadata !"test.c", metadata !"/code/llvm/build0"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !7}
!4 = metadata !{metadata !"0x2e\00caller\00caller\00\004\000\001\000\006\000\001\004", metadata !1, metadata !5, metadata !6, null, void ()* @caller, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 4] [def] [caller]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/code/llvm/build0/test.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !"0x2e\00callee2\00callee2\00\002\001\001\000\006\000\001\002", metadata !1, metadata !5, metadata !6, null, null, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 2] [local] [def] [callee2]
!8 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!9 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!10 = metadata !{metadata !"clang version 3.5.0 (210174)"}
!11 = metadata !{i32 2, i32 0, metadata !7, null}
!12 = metadata !{i32 4, i32 0, metadata !4, null}
