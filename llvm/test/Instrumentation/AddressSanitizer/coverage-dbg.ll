; Test that coverage instrumentation does not lose debug location.

; RUN: opt < %s -asan -asan-module -asan-coverage=1 -S | FileCheck %s

; C++ source:
; 1: struct A {
; 2:  int f();
; 3:  int x;
; 4: };
; 5:
; 6: int A::f() {
; 7:    return x;
; 8: }
; clang++ ../1.cc -O3 -g -S -emit-llvm  -fno-strict-aliasing
; and add sanitize_address to @_ZN1A1fEv

; Test that __sanitizer_cov call has !dbg pointing to the opening { of A::f().
; CHECK: call void @__sanitizer_cov(), !dbg [[A:!.*]]
; CHECK: [[A]] = metadata !{i32 6, i32 0, metadata !{{.*}}, null}


target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

%struct.A = type { i32 }

; Function Attrs: nounwind readonly uwtable
define i32 @_ZN1A1fEv(%struct.A* nocapture readonly %this) #0 align 2 {
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.A* %this}, i64 0, metadata !15), !dbg !20
  %x = getelementptr inbounds %struct.A* %this, i64 0, i32 0, !dbg !21
  %0 = load i32* %x, align 4, !dbg !21
  ret i32 %0, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata) #1

attributes #0 = { sanitize_address nounwind readonly uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 (210251)", i1 true, metadata !"", i32 0, metadata !2, metadata !3, metadata !12, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/code/llvm/build0/../1.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"../1.cc", metadata !"/code/llvm/build0"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"A", i32 1, i64 32, i64 32, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 1, size 32, align 32, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !8}
!6 = metadata !{i32 786445, metadata !1, metadata !"_ZTS1A", metadata !"x", i32 3, i64 32, i64 32, i64 0, i32 0, metadata !7} ; [ DW_TAG_member ] [x] [line 3, size 32, align 32, offset 0] [from int]
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"f", metadata !"f", metadata !"_ZN1A1fEv", i32 2, metadata !9, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 true, null, null, i32 0, null, i32 2} ; [ DW_TAG_subprogram ] [line 2] [f]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{metadata !7, metadata !11}
!11 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!12 = metadata !{metadata !13}
!13 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"f", metadata !"f", metadata !"_ZN1A1fEv", i32 6, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 true, i32 (%struct.A*)* @_ZN1A1fEv, null, metadata !8, metadata !14, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [f]
!14 = metadata !{metadata !15}
!15 = metadata !{i32 786689, metadata !13, metadata !"this", null, i32 16777216, metadata !16, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!16 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS1A]
!17 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!18 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!19 = metadata !{metadata !"clang version 3.5.0 (210251)"}
!20 = metadata !{i32 0, i32 0, metadata !13, null}
!21 = metadata !{i32 7, i32 0, metadata !13, null}
