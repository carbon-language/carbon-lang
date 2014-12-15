; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated with `clang++ -g -emit-llvm -S` from the following code:
; template<int x, int*, template<typename> class y, int ...z>  int func() { return 3; }
; template<typename> struct y_impl { struct nested { }; };
; int glbl = func<3, &glbl, y_impl, 1, 2>();
; y_impl<int>::nested n;

; CHECK: [[INT:0x[0-9a-f]*]]:{{ *}}DW_TAG_base_type
; CHECK-NEXT: DW_AT_name{{.*}} = "int"

; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}"y_impl<int>"
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_type_parameter

; CHECK: DW_AT_name{{.*}}"func<3, &glbl, y_impl, 1, 2>"
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_name{{.*}}= "x"

; This could be made shorter by encoding it as _sdata rather than data4, or
; even as data1. DWARF strongly urges implementations to prefer 
; _sdata/_udata rather than dataN

; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata]{{.*}}(3)

; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INTPTR:0x[0-9a-f]*]]}

; The address of the global 'glbl', followed by DW_OP_stack_value (9f), to use
; the value immediately, rather than indirecting through the address.

; CHECK-NEXT: DW_AT_location [DW_FORM_exprloc]{{ *}}(<0xa> 03 00 00 00 00 00 00 00 00 9f )
; CHECK-NOT: NULL

; CHECK: DW_TAG_GNU_template_template_param
; CHECK-NEXT: DW_AT_name{{.*}}= "y"
; CHECK-NEXT: DW_AT_GNU_template_name{{.*}}= "y_impl"
; CHECK-NOT: NULL

; CHECK: DW_TAG_GNU_template_parameter_pack
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_const_value  [DW_FORM_sdata]{{.*}}(1)
; CHECK-NOT: NULL
; CHECK: DW_TAG_template_value_parameter
; CHECK-NEXT: DW_AT_type{{.*}}=> {[[INT]]}
; CHECK-NEXT: DW_AT_const_value  [DW_FORM_sdata]{{.*}}(2)

; CHECK: [[INTPTR]]:{{ *}}DW_TAG_pointer_type
; CHECK-NEXT: DW_AT_type{{.*}} => {[[INT]]}

%"struct.y_impl<int>::nested" = type { i8 }

@glbl = global i32 0, align 4
@n = global %"struct.y_impl<int>::nested" zeroinitializer, align 1
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]

define internal void @__cxx_global_var_init() section ".text.startup" {
entry:
  %call = call i32 @_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv(), !dbg !33
  store i32 %call, i32* @glbl, align 4, !dbg !33
  ret void, !dbg !33
}

; Function Attrs: nounwind uwtable
define linkonce_odr i32 @_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv() #0 {
entry:
  ret i32 3, !dbg !34
}

define internal void @_GLOBAL__I_a() section ".text.startup" {
entry:
  call void @__cxx_global_var_init(), !dbg !35
  ret void, !dbg !35
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!31, !36}
!llvm.ident = !{!32}

!0 = !{!"0x11\004\00clang version 3.4 (trunk 192849) (llvm/trunk 192850)\000\00\000\00\000", !1, !2, !3, !9, !28, !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/bar.cpp] [DW_LANG_C_plus_plus]
!1 = !{!"bar.cpp", !"/usr/local/google/home/echristo/tmp"}
!2 = !{}
!3 = !{!4, !8}
!4 = !{!"0x13\00y_impl<int>\002\008\008\000\000\000", !1, null, null, !2, null, !5, !"_ZTS6y_implIiE"} ; [ DW_TAG_structure_type ] [y_impl<int>] [line 2, size 8, align 8, offset 0] [def] [from ]
!5 = !{!6}
!6 = !{!"0x2f\00\000\000", null, !7, null} ; [ DW_TAG_template_type_parameter ]
!7 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = !{!"0x13\00nested\002\008\008\000\000\000", !1, !"_ZTS6y_implIiE", null, !2, null, null, !"_ZTSN6y_implIiE6nestedE"} ; [ DW_TAG_structure_type ] [nested] [line 2, size 8, align 8, offset 0] [def] [from ]
!9 = !{!10, !14, !26}
!10 = !{!"0x2e\00__cxx_global_var_init\00__cxx_global_var_init\00\003\001\001\000\006\00256\000\003", !1, !11, !12, null, void ()* @__cxx_global_var_init, null, null, !2} ; [ DW_TAG_subprogram ] [line 3] [local] [def] [__cxx_global_var_init]
!11 = !{!"0x29", !1}         ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/bar.cpp]
!12 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !13, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!13 = !{null}
!14 = !{!"0x2e\00func<3, &glbl, y_impl, 1, 2>\00func<3, &glbl, y_impl, 1, 2>\00_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv\001\000\001\000\006\00256\000\001", !1, !11, !15, null, i32 ()* @_Z4funcILi3EXadL_Z4glblEE6y_implJLi1ELi2EEEiv, !17, null, !2} ; [ DW_TAG_subprogram ] [line 1] [def] [func<3, &glbl, y_impl, 1, 2>]
!15 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = !{!7}
!17 = !{!18, !19, !21, !22}
!18 = !{!"0x30\00x\000\000", null, !7, i32 3, null} ; [ DW_TAG_template_value_parameter ]
!19 = !{!"0x30\00\000\000", null, !20, i32* @glbl, null} ; [ DW_TAG_template_value_parameter ]
!20 = !{!"0xf\00\000\0064\0064\000\000", null, null, !7} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!21 = !{!"0x4106\00y\000\000", null, null, !"y_impl", null} ; [ DW_TAG_GNU_template_template_param ]
!22 = !{!"0x4107\00z\000\000", null, null, !23, null} ; [ DW_TAG_GNU_template_parameter_pack ]
!23 = !{!24, !25}
!24 = !{!"0x30\00\000\000", null, !7, i32 1, null} ; [ DW_TAG_template_value_parameter ]
!25 = !{!"0x30\00\000\000", null, !7, i32 2, null} ; [ DW_TAG_template_value_parameter ]
!26 = !{!"0x2e\00\00\00_GLOBAL__I_a\001\001\001\000\006\0064\000\001", !1, !11, !27, null, void ()* @_GLOBAL__I_a, null, null, !2} ; [ DW_TAG_subprogram ] [line 1] [local] [def]
!27 = !{!"0x15\00\000\000\000\000\000\000", i32 0, null, null, !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = !{!29, !30}
!29 = !{!"0x34\00glbl\00glbl\00\003\000\001", null, !11, !7, i32* @glbl, null} ; [ DW_TAG_variable ] [glbl] [line 3] [def]
!30 = !{!"0x34\00n\00n\00\004\000\001", null, !11, !8, %"struct.y_impl<int>::nested"* @n, null} ; [ DW_TAG_variable ] [n] [line 4] [def]
!31 = !{i32 2, !"Dwarf Version", i32 4}
!32 = !{!"clang version 3.4 (trunk 192849) (llvm/trunk 192850)"}
!33 = !{i32 3, i32 0, !10, null}
!34 = !{i32 1, i32 0, !14, null}
!35 = !{i32 1, i32 0, !26, null}
!36 = !{i32 1, !"Debug Info Version", i32 2}
