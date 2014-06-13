; RUN: llc %s -filetype=obj -O0 -mtriple=i386-unknown-linux-gnu -dwarf-version=4 -o %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; From the code:

; debug-loc-offset1.cc
; int bar (int b) {
;   return b+4;
; }

; debug-loc-offset2.cc
; struct A {
;   int var;
;   virtual char foo();
; };

; void baz(struct A a) {
;   int z = 2;
;   if (a.var > 2)
;     z++;
;   if (a.foo() == 'a')
;     z++;
; }

; Compiled separately for i386-pc-linux-gnu and linked together.
; This ensures that we have multiple compile units so that we can verify that
; debug_loc entries are relative to the low_pc of the CU. The loc entry for
; the byval argument in foo.cpp is in the second CU and so should have
; an offset relative to that CU rather than from the beginning of the text
; section.

; Checking that we have two compile units with two sets of high/lo_pc.
; CHECK: .debug_info contents
; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_low_pc
; CHECK: DW_AT_high_pc

; CHECK: DW_TAG_compile_unit
; CHECK: DW_AT_low_pc
; CHECK: DW_AT_high_pc

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name [DW_FORM_strp]{{.*}}"_Z3baz1A"
; CHECK-NOT: {{DW_TAG|NULL}}
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_location [DW_FORM_sec_offset]   (0x00000000)
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name [DW_FORM_strp]{{.*}}"a"

; CHECK: DW_TAG_variable
; CHECK: DW_AT_location [DW_FORM_exprloc]
; CHECK-NOT: DW_AT_location

; CHECK: .debug_loc contents:
; CHECK: 0x00000000: Beginning address offset: 0x0000000000000000
; CHECK:                Ending address offset: 0x000000000000001a

%struct.A = type { i32 (...)**, i32 }

; Function Attrs: nounwind
define i32 @_Z3bari(i32 %b) #0 {
entry:
  %b.addr = alloca i32, align 4
  store i32 %b, i32* %b.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %b.addr}, metadata !21), !dbg !22
  %0 = load i32* %b.addr, align 4, !dbg !23
  %add = add nsw i32 %0, 4, !dbg !23
  ret i32 %add, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

define void @_Z3baz1A(%struct.A* %a) #2 {
entry:
  %z = alloca i32, align 4
  call void @llvm.dbg.declare(metadata !{%struct.A* %a}, metadata !24), !dbg !25
  call void @llvm.dbg.declare(metadata !{i32* %z}, metadata !26), !dbg !27
  store i32 2, i32* %z, align 4, !dbg !27
  %var = getelementptr inbounds %struct.A* %a, i32 0, i32 1, !dbg !28
  %0 = load i32* %var, align 4, !dbg !28
  %cmp = icmp sgt i32 %0, 2, !dbg !28
  br i1 %cmp, label %if.then, label %if.end, !dbg !28

if.then:                                          ; preds = %entry
  %1 = load i32* %z, align 4, !dbg !30
  %inc = add nsw i32 %1, 1, !dbg !30
  store i32 %inc, i32* %z, align 4, !dbg !30
  br label %if.end, !dbg !30

if.end:                                           ; preds = %if.then, %entry
  %call = call signext i8 @_ZN1A3fooEv(%struct.A* %a), !dbg !31
  %conv = sext i8 %call to i32, !dbg !31
  %cmp1 = icmp eq i32 %conv, 97, !dbg !31
  br i1 %cmp1, label %if.then2, label %if.end4, !dbg !31

if.then2:                                         ; preds = %if.end
  %2 = load i32* %z, align 4, !dbg !33
  %inc3 = add nsw i32 %2, 1, !dbg !33
  store i32 %inc3, i32* %z, align 4, !dbg !33
  br label %if.end4, !dbg !33

if.end4:                                          ; preds = %if.then2, %if.end
  ret void, !dbg !34
}

declare signext i8 @_ZN1A3fooEv(%struct.A*) #2

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0, !9}
!llvm.module.flags = !{!18, !19}
!llvm.ident = !{!20, !20}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 (210479)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/llvm_cmake_gcc/debug-loc-offset1.cc] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"debug-loc-offset1.cc", metadata !"/llvm_cmake_gcc"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"bar", metadata !"bar", metadata !"_Z3bari", i32 1, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z3bari, null, null, metadata !2, i32 1} ; [ DW_TAG_subprogram ] [line 1] [def] [bar]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/llvm_cmake_gcc/debug-loc-offset1.cc]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786449, metadata !10, i32 4, metadata !"clang version 3.5.0 (210479)", i1 false, metadata !"", i32 0, metadata !2, metadata !11, metadata !13, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/llvm_cmake_gcc/debug-loc-offset2.cc] [DW_LANG_C_plus_plus]
!10 = metadata !{metadata !"debug-loc-offset2.cc", metadata !"/llvm_cmake_gcc"}
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786451, metadata !10, null, metadata !"A", i32 1, i64 0, i64 0, i32 0, i32 4, null, null, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 1, size 0, align 0, offset 0] [decl] [from ]
!13 = metadata !{metadata !14}
!14 = metadata !{i32 786478, metadata !10, metadata !15, metadata !"baz", metadata !"baz", metadata !"_Z3baz1A", i32 6, metadata !16, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%struct.A*)* @_Z3baz1A, null, null, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [baz]
!15 = metadata !{i32 786473, metadata !10}        ; [ DW_TAG_file_type ] [/llvm_cmake_gcc/debug-loc-offset2.cc]
!16 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !17, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{null, metadata !12}
!18 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!19 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!20 = metadata !{metadata !"clang version 3.5.0 (210479)"}
!21 = metadata !{i32 786689, metadata !4, metadata !"b", metadata !5, i32 16777217, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [b] [line 1]
!22 = metadata !{i32 1, i32 0, metadata !4, null}
!23 = metadata !{i32 2, i32 0, metadata !4, null}
!24 = metadata !{i32 786689, metadata !14, metadata !"a", metadata !15, i32 16777222, metadata !"_ZTS1A", i32 8192, i32 0} ; [ DW_TAG_arg_variable ] [a] [line 6]
!25 = metadata !{i32 6, i32 0, metadata !14, null}
!26 = metadata !{i32 786688, metadata !14, metadata !"z", metadata !15, i32 7, metadata !8, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [z] [line 7]
!27 = metadata !{i32 7, i32 0, metadata !14, null}
!28 = metadata !{i32 8, i32 0, metadata !29, null} ; [ DW_TAG_imported_declaration ]
!29 = metadata !{i32 786443, metadata !10, metadata !14, i32 8, i32 0, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/llvm_cmake_gcc/debug-loc-offset2.cc]
!30 = metadata !{i32 9, i32 0, metadata !29, null}
!31 = metadata !{i32 10, i32 0, metadata !32, null}
!32 = metadata !{i32 786443, metadata !10, metadata !14, i32 10, i32 0, i32 0, i32 1} ; [ DW_TAG_lexical_block ] [/llvm_cmake_gcc/debug-loc-offset2.cc]
!33 = metadata !{i32 11, i32 0, metadata !32, null}
!34 = metadata !{i32 12, i32 0, metadata !14, null}
