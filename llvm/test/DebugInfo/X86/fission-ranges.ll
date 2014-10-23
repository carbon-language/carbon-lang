; RUN: llc -split-dwarf=Enable -O0 %s -mtriple=x86_64-unknown-linux-gnu -filetype=obj -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; RUN: llvm-objdump -h %t | FileCheck --check-prefix=HDR %s

; CHECK: .debug_info contents:
; CHECK: DW_TAG_compile_unit
; CHECK-NEXT: DW_AT_stmt_list
; CHECK-NEXT: DW_AT_GNU_dwo_name
; CHECK-NEXT: DW_AT_comp_dir
; CHECK-NEXT: DW_AT_GNU_dwo_id
; CHECK-NEXT: DW_AT_GNU_addr_base [DW_FORM_sec_offset]                   (0x00000000)


; CHECK: .debug_info.dwo contents:
; CHECK: DW_AT_location [DW_FORM_sec_offset]   ([[A:0x[0-9a-z]*]])
; CHECK: DW_AT_location [DW_FORM_sec_offset]   ([[E:0x[0-9a-z]*]])
; CHECK: DW_AT_location [DW_FORM_sec_offset]   ([[B:0x[0-9a-z]*]])
; CHECK: DW_AT_location [DW_FORM_sec_offset]   ([[D:0x[0-9a-z]*]])
; CHECK: DW_AT_ranges [DW_FORM_sec_offset]   (0x00000000
; CHECK: .debug_loc contents:
; CHECK-NOT: Beginning address offset
; CHECK: .debug_loc.dwo contents:

; Don't assume these locations are entirely correct - feel free to update them
; if they've changed due to a bugfix, change in register allocation, etc.

; CHECK: [[A]]: Beginning address index: 2
; CHECK-NEXT:                    Length: 190
; CHECK-NEXT:      Location description: 11 00
; CHECK-NEXT: {{^$}}
; CHECK-NEXT:   Beginning address index: 3
; CHECK-NEXT:                    Length: 23
; CHECK-NEXT:      Location description: 50 93 04
; CHECK: [[E]]: Beginning address index: 4
; CHECK-NEXT:                    Length: 21
; CHECK-NEXT:      Location description: 50 93 04
; CHECK: [[B]]: Beginning address index: 5
; CHECK-NEXT:                    Length: 19
; CHECK-NEXT:      Location description: 50 93 04
; CHECK: [[D]]: Beginning address index: 6
; CHECK-NEXT:                    Length: 23
; CHECK-NEXT:      Location description: 50 93 04

; Make sure we don't produce any relocations in any .dwo section (though in particular, debug_info.dwo)
; HDR-NOT: .rela.{{.*}}.dwo

; Make sure we have enough stuff in the debug_addr to cover the address indexes
; (6 is the last index in debug_loc.dwo, making 7 entries of 8 bytes each, 7 * 8
; == 56 base 10 == 38 base 16)

; HDR: .debug_addr 00000038
; HDR-NOT: .rela.{{.*}}.dwo

; From the code:

; extern int c;
; static void foo (int p)
; {
;   int a, b; 
;   unsigned int d, e;

;   for (a = 0; a < 30; a++)
;     for (d = 0; d < 30; d++)
;       for (b = 0; b < 30; b++)
;         for (e = 0; e < 30; e++)
;           {
;             int *w = &c; 
;             *w &= p; 
;           }
; }

; void 
; bar ()
; {
;   foo (1);
; }

; compiled with:

; clang -g -S -gsplit-dwarf -O1 small.c

@c = external global i32

; Function Attrs: nounwind uwtable
define void @bar() #0 {
entry:
  tail call fastcc void @foo(), !dbg !27
  ret void, !dbg !28
}

; Function Attrs: nounwind uwtable
define internal fastcc void @foo() #0 {
entry:
  tail call void @llvm.dbg.value(metadata !29, i64 0, metadata !13, metadata !{metadata !"0x102"}), !dbg !30
  tail call void @llvm.dbg.value(metadata !44, i64 0, metadata !14, metadata !{metadata !"0x102"}), !dbg !31
  %c.promoted9 = load i32* @c, align 4, !dbg !32, !tbaa !33
  br label %for.cond1.preheader, !dbg !31

for.cond1.preheader:                              ; preds = %for.inc16, %entry
  %and.lcssa.lcssa.lcssa10 = phi i32 [ %c.promoted9, %entry ], [ %and, %for.inc16 ]
  %a.08 = phi i32 [ 0, %entry ], [ %inc17, %for.inc16 ]
  br label %for.cond4.preheader, !dbg !37

for.cond4.preheader:                              ; preds = %for.inc13, %for.cond1.preheader
  %and.lcssa.lcssa7 = phi i32 [ %and.lcssa.lcssa.lcssa10, %for.cond1.preheader ], [ %and, %for.inc13 ]
  %d.06 = phi i32 [ 0, %for.cond1.preheader ], [ %inc14, %for.inc13 ]
  br label %for.cond7.preheader, !dbg !38

for.cond7.preheader:                              ; preds = %for.inc10, %for.cond4.preheader
  %and.lcssa5 = phi i32 [ %and.lcssa.lcssa7, %for.cond4.preheader ], [ %and, %for.inc10 ]
  %b.03 = phi i32 [ 0, %for.cond4.preheader ], [ %inc11, %for.inc10 ]
  br label %for.body9, !dbg !39

for.body9:                                        ; preds = %for.body9, %for.cond7.preheader
  %and2 = phi i32 [ %and.lcssa5, %for.cond7.preheader ], [ %and, %for.body9 ], !dbg !40
  %e.01 = phi i32 [ 0, %for.cond7.preheader ], [ %inc, %for.body9 ]
  tail call void @llvm.dbg.value(metadata !41, i64 0, metadata !19, metadata !{metadata !"0x102"}), !dbg !40
  %and = and i32 %and2, 1, !dbg !32
  %inc = add i32 %e.01, 1, !dbg !39
  tail call void @llvm.dbg.value(metadata !{i32 %inc}, i64 0, metadata !18, metadata !{metadata !"0x102"}), !dbg !39
  %exitcond = icmp eq i32 %inc, 30, !dbg !39
  br i1 %exitcond, label %for.inc10, label %for.body9, !dbg !39

for.inc10:                                        ; preds = %for.body9
  %inc11 = add nsw i32 %b.03, 1, !dbg !38
  tail call void @llvm.dbg.value(metadata !{i32 %inc11}, i64 0, metadata !15, metadata !{metadata !"0x102"}), !dbg !38
  %exitcond11 = icmp eq i32 %inc11, 30, !dbg !38
  br i1 %exitcond11, label %for.inc13, label %for.cond7.preheader, !dbg !38

for.inc13:                                        ; preds = %for.inc10
  %inc14 = add i32 %d.06, 1, !dbg !37
  tail call void @llvm.dbg.value(metadata !{i32 %inc14}, i64 0, metadata !16, metadata !{metadata !"0x102"}), !dbg !37
  %exitcond12 = icmp eq i32 %inc14, 30, !dbg !37
  br i1 %exitcond12, label %for.inc16, label %for.cond4.preheader, !dbg !37

for.inc16:                                        ; preds = %for.inc13
  %inc17 = add nsw i32 %a.08, 1, !dbg !31
  tail call void @llvm.dbg.value(metadata !{i32 %inc17}, i64 0, metadata !14, metadata !{metadata !"0x102"}), !dbg !31
  %exitcond13 = icmp eq i32 %inc17, 30, !dbg !31
  br i1 %exitcond13, label %for.end18, label %for.cond1.preheader, !dbg !31

for.end18:                                        ; preds = %for.inc16
  store i32 %and, i32* @c, align 4, !dbg !32, !tbaa !33
  ret void, !dbg !42
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!26, !43}

!0 = metadata !{metadata !"0x11\0012\00clang version 3.4 (trunk 191700) (llvm/trunk 191710)\001\00\000\00small.dwo\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/echristo/tmp/small.c] [DW_LANG_C99]
!1 = metadata !{metadata !"small.c", metadata !"/usr/local/google/home/echristo/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !8}
!4 = metadata !{metadata !"0x2e\00bar\00bar\00\0018\000\001\000\006\000\001\0019", metadata !1, metadata !5, metadata !6, null, void ()* @bar, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 18] [def] [scope 19] [bar]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/echristo/tmp/small.c]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null}
!8 = metadata !{metadata !"0x2e\00foo\00foo\00\002\001\001\000\006\00256\001\003", metadata !1, metadata !5, metadata !9, null, void ()* @foo, null, null, metadata !12} ; [ DW_TAG_subprogram ] [line 2] [local] [def] [scope 3] [foo]
!9 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11}
!11 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !13, metadata !14, metadata !15, metadata !16, metadata !18, metadata !19}
!13 = metadata !{metadata !"0x101\00p\0016777218\000", metadata !8, metadata !5, metadata !11} ; [ DW_TAG_arg_variable ] [p] [line 2]
!14 = metadata !{metadata !"0x100\00a\004\000", metadata !8, metadata !5, metadata !11} ; [ DW_TAG_auto_variable ] [a] [line 4]
!15 = metadata !{metadata !"0x100\00b\004\000", metadata !8, metadata !5, metadata !11} ; [ DW_TAG_auto_variable ] [b] [line 4]
!16 = metadata !{metadata !"0x100\00d\005\000", metadata !8, metadata !5, metadata !17} ; [ DW_TAG_auto_variable ] [d] [line 5]
!17 = metadata !{metadata !"0x24\00unsigned int\000\0032\0032\000\000\007", null, null} ; [ DW_TAG_base_type ] [unsigned int] [line 0, size 32, align 32, offset 0, enc DW_ATE_unsigned]
!18 = metadata !{metadata !"0x100\00e\005\000", metadata !8, metadata !5, metadata !17} ; [ DW_TAG_auto_variable ] [e] [line 5]
!19 = metadata !{metadata !"0x100\00w\0012\000", metadata !20, metadata !5, metadata !25} ; [ DW_TAG_auto_variable ] [w] [line 12]
!20 = metadata !{metadata !"0xb\0011\000\004", metadata !1, metadata !21} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!21 = metadata !{metadata !"0xb\0010\000\003", metadata !1, metadata !22} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!22 = metadata !{metadata !"0xb\009\000\002", metadata !1, metadata !23} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!23 = metadata !{metadata !"0xb\008\000\001", metadata !1, metadata !24} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!24 = metadata !{metadata !"0xb\007\000\000", metadata !1, metadata !8} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/echristo/tmp/small.c]
!25 = metadata !{metadata !"0xf\00\000\0064\0064\000\000", null, null, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!26 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!27 = metadata !{i32 20, i32 0, metadata !4, null}
!28 = metadata !{i32 21, i32 0, metadata !4, null}
!29 = metadata !{i32 1}
!30 = metadata !{i32 2, i32 0, metadata !8, null}
!31 = metadata !{i32 7, i32 0, metadata !24, null}
!32 = metadata !{i32 13, i32 0, metadata !20, null}
!33 = metadata !{metadata !34, metadata !34, i64 0}
!34 = metadata !{metadata !"int", metadata !35, i64 0}
!35 = metadata !{metadata !"omnipotent char", metadata !36, i64 0}
!36 = metadata !{metadata !"Simple C/C++ TBAA"}
!37 = metadata !{i32 8, i32 0, metadata !23, null}
!38 = metadata !{i32 9, i32 0, metadata !22, null}
!39 = metadata !{i32 10, i32 0, metadata !21, null}
!40 = metadata !{i32 12, i32 0, metadata !20, null}
!41 = metadata !{i32* @c}
!42 = metadata !{i32 15, i32 0, metadata !8, null}
!43 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
!44 = metadata !{i32 0}
