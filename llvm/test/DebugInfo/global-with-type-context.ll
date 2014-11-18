; REQUIRES: object-emission

; RUN: %llc_dwarf -filetype=obj -O0 < %s > %t
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s

; IR generated from clang -g with the following source:
; struct F {
;   static const int i = 2;
;   virtual ~F();
; };
;
; void f1() {
;   int i = F::i;
; }

; Make sure we correctly handle context of a global variable being a type identifier.
; CHECK:  [[STRUCT:.*]]: DW_TAG_structure_type
; CHECK: DW_AT_name [DW_FORM_strp] {{.*}}= "F")
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_specification {{.*}} "i"
; CHECK-NEXT: DW_AT_const_value [DW_FORM_sdata] (2)

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-darwin14.0.0"

; Function Attrs: nounwind
define void @_Z2f1v() #0 {
entry:
  %i = alloca i32, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i}, metadata !29, metadata !30), !dbg !31
  store i32 2, i32* %i, align 4, !dbg !31
  ret void, !dbg !32
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26, !27}
!llvm.ident = !{!28}

!0 = metadata !{metadata !"0x11\004\00clang version 3.6.0 (trunk 222175)\000\00\000\00\001", metadata !1, metadata !2, metadata !3, metadata !20, metadata !24, metadata !2} ; [ DW_TAG_compile_unit ] [<stdin>] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"<stdin>", metadata !"."}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x13\00F\001\0064\0064\000\000\000", metadata !5, null, null, metadata !6, metadata !"_ZTS1F", null, metadata !"_ZTS1F"} ; [ DW_TAG_structure_type ] [F] [line 1, size 64, align 64, offset 0] [def] [from ]
!5 = metadata !{metadata !"test.cpp", metadata !"."}
!6 = metadata !{metadata !7, metadata !14, metadata !16}
!7 = metadata !{metadata !"0xd\00_vptr$F\000\0064\000\000\0064", metadata !5, metadata !8, metadata !9} ; [ DW_TAG_member ] [_vptr$F] [line 0, size 64, align 0, offset 0] [artificial] [from ]
!8 = metadata !{metadata !"0x29", metadata !5}    ; [ DW_TAG_file_type ]
!9 = metadata !{metadata !"0xf\00\000\0064\000\000\000", null, null, metadata !10} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 0, offset 0] [from __vtbl_ptr_type]
!10 = metadata !{metadata !"0xf\00__vtbl_ptr_type\000\0064\000\000\000", null, null, metadata !11} ; [ DW_TAG_pointer_type ] [__vtbl_ptr_type] [line 0, size 64, align 0, offset 0] [from ]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = metadata !{metadata !"0xd\00i\002\000\000\000\004096", metadata !5, metadata !"_ZTS1F", metadata !15, i32 2} ; [ DW_TAG_member ] [i] [line 2, size 0, align 0, offset 0] [static] [from ]
!15 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !13} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from int]
!16 = metadata !{metadata !"0x2e\00~F\00~F\00\003\000\000\001\000\00256\000\003", metadata !5, metadata !"_ZTS1F", metadata !17, metadata !"_ZTS1F", null, null, null, null} ; [ DW_TAG_subprogram ] [line 3] [~F]
!17 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19}
!19 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088\00", null, null, metadata !"_ZTS1F"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1F]
!20 = metadata !{metadata !21}
!21 = metadata !{metadata !"0x2e\00f1\00f1\00_Z2f1v\006\000\001\000\000\00256\000\006", metadata !5, metadata !8, metadata !22, null, void ()* @_Z2f1v, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 6] [def] [f1]
!22 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !23, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = metadata !{null}
!24 = metadata !{metadata !25}
!25 = metadata !{metadata !"0x34\00i\00i\00\002\001\001", metadata !"_ZTS1F", metadata !8, metadata !15, i32 2, metadata !14} ; [ DW_TAG_variable ] [i] [line 2] [local] [def]
!26 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!27 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!28 = metadata !{metadata !"clang version 3.6.0 (trunk 222175)"}
!29 = metadata !{metadata !"0x100\00i\007\000", metadata !21, metadata !8, metadata !13} ; [ DW_TAG_auto_variable ] [i] [line 7]
!30 = metadata !{metadata !"0x102"}               ; [ DW_TAG_expression ]
!31 = metadata !{i32 7, i32 0, metadata !21, null}
!32 = metadata !{i32 8, i32 0, metadata !21, null}
