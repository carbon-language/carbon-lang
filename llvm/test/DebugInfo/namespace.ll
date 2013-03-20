; RUN: llc -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "A"
; CHECK-NOT: NULL
; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "B"
; CHECK-NOT: NULL
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}= "i"

; IR generated from clang/test/CodeGenCXX/debug-info-namespace.cpp, file paths
; changed to protect the guilty. The C++ source code is simply:
; namespace A {
; #line 1 "foo.cpp"
; namespace B {
; int i;
; }
; }

@_ZN1A1B1iE = global i32 0, align 4

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, i32 0, i32 4, metadata !1, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !3, metadata !3, metadata !3, metadata !4, metadata !""} ; [ DW_TAG_compile_unit ] [/home/foo/debug-info-namespace.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{i32 786473, metadata !2}          ; [ DW_TAG_file_type ] [/home/foo/debug-info-namespace.cpp]
!2 = metadata !{metadata !"debug-info-namespace.cpp", metadata !"/home/foo"}
!3 = metadata !{i32 0}
!4 = metadata !{metadata !5}
!5 = metadata !{i32 786484, i32 0, metadata !6, metadata !"i", metadata !"i", metadata !"_ZN1A1B1iE", metadata !7, i32 2, metadata !10, i32 0, i32 1, i32* @_ZN1A1B1iE, null} ; [ DW_TAG_variable ] [i] [line 2] [def]
!6 = metadata !{i32 786489, metadata !7, metadata !9, metadata !"B", i32 1} ; [ DW_TAG_namespace ] [B] [line 1]
!7 = metadata !{i32 786473, metadata !8}          ; [ DW_TAG_file_type ] [/home/foo/foo.cpp]
!8 = metadata !{metadata !"foo.cpp", metadata !"/home/foo"}
!9 = metadata !{i32 786489, metadata !1, null, metadata !"A", i32 3} ; [ DW_TAG_namespace ] [A] [line 3]
!10 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
