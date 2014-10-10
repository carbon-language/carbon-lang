; RUN: llc -mtriple=x86_64-apple-darwin < %s -filetype=obj \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s
; RUN: llc -mtriple=x86_64-linux-gnu < %s -filetype=obj \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=LINUX %s
; RUN: llc -mtriple=x86_64-apple-darwin < %s -filetype=obj -regalloc=basic \
; RUN:     | llvm-dwarfdump -debug-dump=info - | FileCheck --check-prefix=CHECK --check-prefix=DARWIN %s

; CHECK: DW_TAG_subprogram
; CHECK:   DW_AT_abstract_origin {{.*}} "foo"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "sp"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} "nums"

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_name {{.*}} "foo"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "sp"
; CHECK:   DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_name {{.*}} "nums"

;CHECK: DW_TAG_inlined_subroutine
;CHECK-NEXT: DW_AT_abstract_origin {{.*}} "foo"
;CHECK-NEXT: DW_AT_low_pc [DW_FORM_addr]
;CHECK-NEXT: DW_AT_high_pc [DW_FORM_data4]
;CHECK-NEXT: DW_AT_call_file
;CHECK-NEXT: DW_AT_call_line

;CHECK: DW_TAG_formal_parameter
;FIXME: Linux shouldn't drop this parameter either...
;CHECK-NOT: DW_TAG
;DARWIN:   DW_AT_abstract_origin {{.*}} "sp"
;DARWIN: DW_TAG_formal_parameter
;CHECK: DW_AT_abstract_origin {{.*}} "nums"
;CHECK-NOT: DW_TAG_formal_parameter

%struct.S1 = type { float*, i32 }

@p = common global %struct.S1 zeroinitializer, align 8

define i32 @foo(%struct.S1* nocapture %sp, i32 %nums) nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !{%struct.S1* %sp}, i64 0, metadata !9, metadata !{metadata !"0x102"}), !dbg !20
  tail call void @llvm.dbg.value(metadata !{i32 %nums}, i64 0, metadata !18, metadata !{metadata !"0x102"}), !dbg !21
  %tmp2 = getelementptr inbounds %struct.S1* %sp, i64 0, i32 1, !dbg !22
  store i32 %nums, i32* %tmp2, align 4, !dbg !22
  %call = tail call float* @bar(i32 %nums) nounwind optsize, !dbg !27
  %tmp5 = getelementptr inbounds %struct.S1* %sp, i64 0, i32 0, !dbg !27
  store float* %call, float** %tmp5, align 8, !dbg !27
  %cmp = icmp ne float* %call, null, !dbg !29
  %cond = zext i1 %cmp to i32, !dbg !29
  ret i32 %cond, !dbg !29
}

declare float* @bar(i32) optsize

define void @foobar() nounwind optsize ssp {
entry:
  tail call void @llvm.dbg.value(metadata !30, i64 0, metadata !9, metadata !{metadata !"0x102"}) nounwind, !dbg !31
  tail call void @llvm.dbg.value(metadata !34, i64 0, metadata !18, metadata !{metadata !"0x102"}) nounwind, !dbg !35
  store i32 1, i32* getelementptr inbounds (%struct.S1* @p, i64 0, i32 1), align 8, !dbg !36
  %call.i = tail call float* @bar(i32 1) nounwind optsize, !dbg !37
  store float* %call.i, float** getelementptr inbounds (%struct.S1* @p, i64 0, i32 0), align 8, !dbg !37
  ret void, !dbg !38
}

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!43}

!0 = metadata !{metadata !"0x2e\00foo\00foo\00\008\000\001\000\006\00256\001\008", metadata !1, metadata !1, metadata !3, null, i32 (%struct.S1*, i32)* @foo, null, null, metadata !41} ; [ DW_TAG_subprogram ] [line 8] [def] [foo]
!1 = metadata !{metadata !"0x29", metadata !42} ; [ DW_TAG_file_type ]
!2 = metadata !{metadata !"0x11\0012\00clang version 2.9 (trunk 125693)\001\00\000\00\001", metadata !42, metadata !8, metadata !8, metadata !39, metadata !40,  metadata !44} ; [ DW_TAG_compile_unit ]
!3 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !42, metadata !1, null, metadata !4, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!4 = metadata !{metadata !5}
!5 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, metadata !2} ; [ DW_TAG_base_type ]
!6 = metadata !{metadata !"0x2e\00foobar\00foobar\00\0015\000\001\000\006\000\001\000", metadata !1, metadata !1, metadata !7, null, void ()* @foobar, null, null, null} ; [ DW_TAG_subprogram ] [line 15] [def] [scope 0] [foobar]
!7 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", metadata !42, metadata !1, null, metadata !8, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{null}
!9 = metadata !{metadata !"0x101\00sp\0016777223\000", metadata !0, metadata !1, metadata !10, metadata !32} ; [ DW_TAG_arg_variable ]
!10 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, metadata !2, metadata !11} ; [ DW_TAG_pointer_type ]
!11 = metadata !{metadata !"0x16\00S1\004\000\000\000\000", metadata !42, metadata !2, metadata !12} ; [ DW_TAG_typedef ]
!12 = metadata !{metadata !"0x13\00S1\001\00128\0064\000\000\000", metadata !42, metadata !2, null, metadata !13, null, null, null} ; [ DW_TAG_structure_type ] [S1] [line 1, size 128, align 64, offset 0] [def] [from ]
!13 = metadata !{metadata !14, metadata !17}
!14 = metadata !{metadata !"0xd\00m\002\0064\0064\000\000", metadata !42, metadata !1, metadata !15} ; [ DW_TAG_member ]
!15 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, metadata !2, metadata !16} ; [ DW_TAG_pointer_type ]
!16 = metadata !{metadata !"0x24\00float\000\0032\0032\000\000\004", null, metadata !2} ; [ DW_TAG_base_type ]
!17 = metadata !{metadata !"0xd\00nums\003\0032\0032\0064\000", metadata !42, metadata !1, metadata !5} ; [ DW_TAG_member ]
!18 = metadata !{metadata !"0x101\00nums\0033554439\000", metadata !0, metadata !1, metadata !5, metadata !32} ; [ DW_TAG_arg_variable ]
!19 = metadata !{metadata !"0x34\00p\00p\00\0014\000\001", metadata !2, metadata !1, metadata !11, %struct.S1* @p, null} ; [ DW_TAG_variable ]
!20 = metadata !{i32 7, i32 13, metadata !0, null}
!21 = metadata !{i32 7, i32 21, metadata !0, null}
!22 = metadata !{i32 9, i32 3, metadata !23, null}
!23 = metadata !{metadata !"0xb\008\001\000", metadata !1, metadata !0} ; [ DW_TAG_lexical_block ]
!27 = metadata !{i32 10, i32 3, metadata !23, null}
!29 = metadata !{i32 11, i32 3, metadata !23, null}
!30 = metadata !{%struct.S1* @p}
!31 = metadata !{i32 7, i32 13, metadata !0, metadata !32}
!32 = metadata !{i32 16, i32 3, metadata !33, null}
!33 = metadata !{metadata !"0xb\0015\0015\001", metadata !1, metadata !6} ; [ DW_TAG_lexical_block ]
!34 = metadata !{i32 1}
!35 = metadata !{i32 7, i32 21, metadata !0, metadata !32}
!36 = metadata !{i32 9, i32 3, metadata !23, metadata !32}
!37 = metadata !{i32 10, i32 3, metadata !23, metadata !32}
!38 = metadata !{i32 17, i32 1, metadata !33, null}
!39 = metadata !{metadata !0, metadata !6}
!40 = metadata !{metadata !19}
!41 = metadata !{metadata !9, metadata !18}
!42 = metadata !{metadata !"nm2.c", metadata !"/private/tmp"}
!43 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!44 = metadata !{}
