; RUN: llc -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_const_value [DW_FORM_udata]	(0)
; CHECK-NEXT: DW_AT_name {{.*}}"a"
;
; ModuleID = 'union.c'
; generated at -O1 from:
; union mfi_evt {
;   struct {
;     int reserved;
;   } members;
; } mfi_aen_setup() {
;   union mfi_evt a;
;   a.members.reserved = 0;
; }
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%union.mfi_evt = type { %struct.anon }
%struct.anon = type { i32 }

; Function Attrs: nounwind readnone ssp uwtable
define i32 @mfi_aen_setup() #0 {
entry:
  tail call void @llvm.dbg.declare(metadata %union.mfi_evt* undef, metadata !16, metadata !21), !dbg !22
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !16, metadata !21), !dbg !22
  ret i32 undef, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind readnone ssp uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18, !19}
!llvm.ident = !{!20}

!0 = !{!"0x11\0012\00clang version 3.7.0 (trunk 226915) (llvm/trunk 226905)\001\00\000\00\001", !1, !2, !2, !3, !2, !2} ; [ DW_TAG_compile_unit ] [/union.c] [DW_LANG_C99]
!1 = !{!"union.c", !""}
!2 = !{}
!3 = !{!4}
!4 = !{!"0x2e\00mfi_aen_setup\00mfi_aen_setup\00\005\000\001\000\000\000\001\005", !1, !5, !6, null, i32 ()* @mfi_aen_setup, null, null, !15} ; [ DW_TAG_subprogram ] [line 5] [def] [mfi_aen_setup]
!5 = !{!"0x29", !1}                               ; [ DW_TAG_file_type ] [/union.c]
!6 = !{!"0x15\00\000\000\000\000\000\000", null, null, null, !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = !{!8}
!8 = !{!"0x17\00mfi_evt\001\0032\0032\000\000\000", !1, null, null, !9, null, null, null} ; [ DW_TAG_union_type ] [mfi_evt] [line 1, size 32, align 32, offset 0] [def] [from ]
!9 = !{!10}
!10 = !{!"0xd\00members\004\0032\0032\000\000", !1, !8, !11} ; [ DW_TAG_member ] [members] [line 4, size 32, align 32, offset 0] [from ]
!11 = !{!"0x13\00\002\0032\0032\000\000\000", !1, !8, null, !12, null, null, null} ; [ DW_TAG_structure_type ] [line 2, size 32, align 32, offset 0] [def] [from ]
!12 = !{!13}
!13 = !{!"0xd\00reserved\003\0032\0032\000\000", !1, !11, !14} ; [ DW_TAG_member ] [reserved] [line 3, size 32, align 32, offset 0] [from int]
!14 = !{!"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!15 = !{!16}
!16 = !{!"0x100\00a\006\000", !4, !5, !8}         ; [ DW_TAG_auto_variable ] [a] [line 6]
!17 = !{i32 2, !"Dwarf Version", i32 2}
!18 = !{i32 2, !"Debug Info Version", i32 2}
!19 = !{i32 1, !"PIC Level", i32 2}
!20 = !{!"clang version 3.7.0 (trunk 226915) (llvm/trunk 226905)"}
!21 = !{!"0x102"}                                 ; [ DW_TAG_expression ]
!22 = !MDLocation(line: 6, column: 17, scope: !4)
!23 = !MDLocation(line: 8, column: 1, scope: !4)
