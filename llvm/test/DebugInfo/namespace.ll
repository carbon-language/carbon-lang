; RUN: llc -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "A"
; CHECK-NOT: NULL
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}= "i"

; IR generated from clang/test/CodeGenCXX/debug-info-namespace.cpp, file paths
; changed to protect the guilty. The C++ source code is simply:
; namespace A {
; int i;
; }

@_ZN1A1iE = global i32 0, align 4

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 4, metadata !1, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !3, metadata !3, metadata !3, metadata !4, metadata !""} ; [ DW_TAG_compile_unit ] [/home/foobar/debug-info-namespace.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 786473, metadata !2}          ; [ DW_TAG_file_type ] [/home/foobar/debug-info-namespace.cpp]
!2 = metadata !{metadata !"debug-info-namespace.cpp", metadata !"/home/foobar/"}
!3 = metadata !{i32 0}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786484, i32 0, metadata !6, metadata !"i", metadata !"i", metadata !"_ZN1A1iE", metadata !1, i32 4, metadata !7, i32 0, i32 1, i32* @_ZN1A1iE, null} ; [ DW_TAG_variable ] [i] [line 4] [def]
!6 = metadata !{i32 786489, metadata !1, null, metadata !"A", i32 3} ; [ DW_TAG_namespace ] [A] [line 3]
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
