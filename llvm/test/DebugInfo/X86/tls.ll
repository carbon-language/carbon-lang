; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=SINGLE --check-prefix=SINGLE-64 %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=i386-linux-gnu \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=SINGLE --check-prefix=SINGLE-32 %s

; RUN: llc %s -o - -filetype=asm -O0 -mtriple=x86_64-unknown-linux-gnu -split-dwarf=Enable \
; RUN:   | FileCheck --check-prefix=CHECK --check-prefix=FISSION %s

; FIXME: add relocation and DWARF expression support to llvm-dwarfdump & use
; that here instead of raw assembly printing

; FISSION: .section    .debug_info.dwo,
; 3 bytes of data in this DW_FORM_block1 representation of the location of 'tls'
; FISSION: .byte 3{{ *}}# DW_AT_location
; DW_OP_GNU_const_index (0xfx == 252) to refer to the debug_addr table
; FISSION-NEXT: .byte 252
; an index of zero into the debug_addr table
; FISSION-NEXT: .byte 0

; SINGLE: .section     .debug_info,
; 10 bytes of data in this DW_FORM_block1 representation of the location of 'tls'
; SINGLE-64: .byte     10 # DW_AT_location
; DW_OP_const8u (0x0e == 14) of address
; SINGLE-64-NEXT: .byte        14
; SINGLE-64-NEXT: .quad tls@DTPOFF

; SINGLE-32: .byte     6 # DW_AT_location
; DW_OP_const4u (0x0e == 12) of address
; SINGLE-32-NEXT: .byte        12
; SINGLE-32-NEXT: .long tls@DTPOFF

; DW_OP_GNU_push_tls_address
; CHECK-NEXT: .byte 224

; FISSION: DW_TAG_variable
; FISSION: .byte 2 # DW_AT_location
; DW_OP_GNU_addr_index
; FISSION-NEXT: .byte 251
; FISSION-NEXT: .byte 1

; FISSION: DW_TAG_template_value_parameter
; FISSION: .byte 3 # DW_AT_location
; DW_OP_GNU_addr_index
; FISSION-NEXT: .byte 251
; FISSION-NEXT: .byte 1
; DW_OP_stack_value
; FISSION-NEXT: .byte 159

; check that the expected TLS address description is the first thing in the debug_addr section
; FISSION: .section    .debug_addr
; FISSION-NEXT: .quad  tls@DTPOFF
; FISSION-NEXT: .quad  glbl
; FISSION-NOT: .quad  glbl

; Generated from:

; __thread int tls;
; int glbl;
;
; template <int *I>
; int func() {
;   return 0;
; }
;
; template int func<&glbl>(); // create a second reference to 'glbl'


@tls = thread_local global i32 0, align 4
@glbl = global i32 0, align 4

; Function Attrs: nounwind uwtable
define weak_odr i32 @_Z4funcIXadL_Z4glblEEEiv() #0 {
entry:
  ret i32 0, !dbg !18
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!15, !16}
!llvm.ident = !{!17}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !12, metadata !2, metadata !"-.dwo"} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/tls.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"tls.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"func<&glbl>", metadata !"func<&glbl>", metadata !"_Z4funcIXadL_Z4glblEEEiv", i32 5, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @_Z4funcIXadL_Z4glblEEEiv, metadata !9, null, metadata !2, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [func<&glbl>]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/tls.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{metadata !10}
!10 = metadata !{i32 786480, null, metadata !"I", metadata !11, i32* @glbl, null, i32 0, i32 0} ; [ DW_TAG_template_value_parameter ]
!11 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!12 = metadata !{metadata !13, metadata !14}
!13 = metadata !{i32 786484, i32 0, null, metadata !"tls", metadata !"tls", metadata !"", metadata !5, i32 1, metadata !8, i32 0, i32 1, i32* @tls, null} ; [ DW_TAG_variable ] [tls] [line 1] [def]
!14 = metadata !{i32 786484, i32 0, null, metadata !"glbl", metadata !"glbl", metadata !"", metadata !5, i32 2, metadata !8, i32 0, i32 1, i32* @glbl, null} ; [ DW_TAG_variable ] [glbl] [line 2] [def]
!15 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!16 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!17 = metadata !{metadata !"clang version 3.5 "}
!18 = metadata !{i32 6, i32 0, metadata !4, null}
