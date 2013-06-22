; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated with `clang++ -g -emit-llvm -S` from the following code:
; template<int x, int*, template<typename> class y, int ...z>  int func() { return 3; }
; template<typename> struct y_impl {};
; int glbl = func<3, &glbl, y_impl, 1, 2>();

; CHECK: [[INT:0x[0-9a-f]*]]:{{ *}}DW_TAG_base_type
; CHECK-NEXT: DW_AT_name{{.*}} = "int"

; CHECK: DW_AT_name{{.*}}"func<3, &glbl, y_impl, 1, 2>"
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_name{{.*}}= "x"

; This could be made shorter by encoding it as _sdata rather than data4, or
; even as data1. DWARF strongly urges implementations to prefer 
; _sdata/_udata rather than dataN

; CHECK-NEXT: DW_AT_const_value [DW_FORM_data4]{{.*}}(0x00000003)

; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INTPTR:0x[0-9a-f]*]]}

; The address of the global 'glbl', followed by DW_OP_stack_value (9f), to use
; the value immediately, rather than indirecting through the address.

; CHECK-NEXT: DW_AT_location [DW_FORM_block1]{{ *}}(<0x0a> 03 00 00 00 00 00 00 00 00 9f )
; CHECK-NOT: NULL

; CHECK: DW_TAG_GNU_template_template_param
; CHECK-NEXT: DW_AT_name{{.*}}= "y"
; CHECK-NEXT: DW_AT_GNU_template_name{{.*}}= "y_impl"
; CHECK-NOT: NULL

; CHECK: DW_TAG_GNU_template_parameter_pack
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_const_value  [DW_FORM_data4]{{.*}}(0x00000001)
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_const_value  [DW_FORM_data4]{{.*}}(0x00000002)

; CHECK: [[INTPTR]]:{{ *}}DW_TAG_pointer_type
; CHECK-NEXT: DW_AT_type{{.*}} => {[[INT]]}

; 

@glbl = global i32 0, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  %call = call i32 @_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv(), !dbg !26
  store i32 %call, i32* @glbl, align 4, !dbg !26
  ret void, !dbg !26
}

; Function Attrs: nounwind uwtable
define linkonce_odr i32 @_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv() #0 {
entry:
  ret i32 3, !dbg !27
}

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !28
  ret void, !dbg !28
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!25}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !23, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/tmp/templ.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"templ.cpp", metadata !"/tmp"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !8, metadata !21}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"__cxx_global_var_init", metadata !"__cxx_global_var_init", metadata !"", i32 3, metadata !6, i1 true, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @__cxx_global_var_init, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [local] [def] [__cxx_global_var_init]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/templ.cpp]
!6 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"func<3, &glbl, y_impl, 1, 2>", metadata !"func<3, &glbl, y_impl, 1, 2>", metadata !"_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv", i32 1, metadata !9, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv, metadata !12, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [func<3, &glbl, y_impl, 1, 2>]
!9 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{metadata !11}
!11 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13, metadata !14, metadata !16, metadata !17}
!13 = metadata !{i32 786480, null, metadata !"x", metadata !11, i32 3, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!14 = metadata !{i32 786480, null, metadata !"", metadata !15, i32* @glbl, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!15 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!16 = metadata !{i32 803078, null, metadata !"y", null, metadata !"y_impl", null, i32 0, i32 0} ; [ DW_TAG_GNU_template_template_param ]
!17 = metadata !{i32 803079, null, metadata !"z", null, metadata !18, null, i32 0, i32 0} ; [ DW_TAG_GNU_template_parameter_pack ]
!18 = metadata !{metadata !19, metadata !20}
!19 = metadata !{i32 786480, null, metadata !"", metadata !11, i32 1, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!20 = metadata !{i32 786480, null, metadata !"", metadata !11, i32 2, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!21 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"_GLOBAL__I_a", metadata !"_GLOBAL__I_a", metadata !"", i32 1, metadata !22, i1 true, i1 true, i32 0, i32 0, null, i32 64, i1 false, void ()* @_GLOBAL__I_a, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [local] [def] [_GLOBAL__I_a]
!22 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !2, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = metadata !{metadata !24}
!24 = metadata !{i32 786484, i32 0, null, metadata !"glbl", metadata !"glbl", metadata !"", metadata !5, i32 3, metadata !11, i32 0, i32 1, i32* @glbl, null} ; [ DW_TAG_variable ] [glbl] [line 3] [def]
!25 = metadata !{i32 2, metadata !"Dwarf Version", i32 3}
!26 = metadata !{i32 3, i32 0, metadata !4, null}
!27 = metadata !{i32 1, i32 0, metadata !8, null}
!28 = metadata !{i32 1, i32 0, metadata !21, null}
