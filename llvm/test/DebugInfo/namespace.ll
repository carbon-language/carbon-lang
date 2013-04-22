; REQUIRES: object-emission

; RUN: llc -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; CHECK: debug_info contents
; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "A"
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F1:[0-9]]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x03)
; CHECK-NOT: NULL
; CHECK: [[NS2:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "B"
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2:[0-9]]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x01)
; CHECK-NOT: NULL
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}= "i"
; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name{{.*}}= "func"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x07)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS2]]})
; CHECK: file_names[  [[F1]]]{{.*}}debug-info-namespace.cpp
; CHECK: file_names[  [[F2]]]{{.*}}foo.cpp

; IR generated from clang/test/CodeGenCXX/debug-info-namespace.cpp, file paths
; changed to protect the guilty. The C++ source code is simply:
; namespace A {
; #line 1 "foo.cpp"
; namespace B {
; int i;
; }
; }
;
; int func() {
;   using namespace A::B;
;   return i;
; }

@_ZN1A1B1iE = global i32 0, align 4

; Function Attrs: nounwind uwtable 
define i32 @_Z4funcv() #0 {
entry:
  %0 = load i32* @_ZN1A1B1iE, align 4, !dbg !16
  ret i32 %0, !dbg !16
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !10, metadata !14, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/llvm/src/tools/clang//usr/local/google/home/blaikie/dev/llvm/src/tools/clang/test/CodeGenCXX/debug-info-namespace.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"/usr/local/google/home/blaikie/dev/llvm/src/tools/clang/test/CodeGenCXX/debug-info-namespace.cpp", metadata !"/usr/local/google/home/blaikie/dev/llvm/src/tools/clang"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"func", metadata !"func", metadata !"_Z4funcv", i32 6, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z4funcv, null, null, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [func]
!5 = metadata !{metadata !"foo.cpp", metadata !"/usr/local/google/home/blaikie/dev/llvm/src/tools/clang"}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/llvm/src/tools/clang/foo.cpp]
!7 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786484, i32 0, metadata !12, metadata !"i", metadata !"i", metadata !"_ZN1A1B1iE", metadata !6, i32 2, metadata !9, i32 0, i32 1, i32* @_ZN1A1B1iE, null} ; [ DW_TAG_variable ] [i] [line 2] [def]
!12 = metadata !{i32 786489, metadata !5, metadata !13, metadata !"B", i32 1} ; [ DW_TAG_namespace ] [B] [line 1]
!13 = metadata !{i32 786489, metadata !1, null, metadata !"A", i32 3} ; [ DW_TAG_namespace ] [A] [line 3]
!14 = metadata !{metadata !15}
!15 = metadata !{i32 786490, metadata !4, metadata !12, i32 7} ; [ DW_TAG_imported_module ]
!16 = metadata !{i32 8, i32 0, metadata !4, null}
