; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-unknown-unknown -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; IR generated from clang -g with the following source:
; struct foo {
;   foo(const foo&);
;   int i;
; };
;
; void func(foo f, foo g) {
;   f.i++;
; }

; CHECK: debug_info contents
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name{{.*}}"_Z4func3fooS_"
; CHECK-NOT: NULL
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"f"
; CHECK-NOT: NULL
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}"g"

%struct.foo = type { i32 }

; Function Attrs: nounwind uwtable
define void @_Z4func3fooS_(%struct.foo* %f, %struct.foo* %g) #0 {
entry:
  call void @llvm.dbg.declare(metadata !{%struct.foo* %f}, metadata !19, metadata !{metadata !"0x102"}), !dbg !20
  call void @llvm.dbg.declare(metadata !{%struct.foo* %g}, metadata !21, metadata !{metadata !"0x102"}), !dbg !20
  %i = getelementptr inbounds %struct.foo* %f, i32 0, i32 0, !dbg !22
  %0 = load i32* %i, align 4, !dbg !22
  %inc = add nsw i32 %0, 1, !dbg !22
  store i32 %inc, i32* %i, align 4, !dbg !22
  ret void, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!24}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 \000\00\000\00\001", metadata !1, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/scratch/scratch.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"scratch.cpp", metadata !"/usr/local/google/home/blaikie/dev/scratch"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{metadata !"0x2e\00func\00func\00_Z4func3fooS_\006\000\001\000\006\00256\000\006", metadata !1, metadata !5, metadata !6, null, void (%struct.foo*, %struct.foo*)* @_Z4func3fooS_, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 6] [def] [func]
!5 = metadata !{metadata !"0x29", metadata !1}          ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/scratch/scratch.cpp]
!6 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !7, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{null, metadata !8, metadata !8}
!8 = metadata !{metadata !"0x13\00foo\001\0032\0032\000\000\000", metadata !1, null, null, metadata !9, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 1, size 32, align 32, offset 0] [def] [from ]
!9 = metadata !{metadata !10, metadata !12}
!10 = metadata !{metadata !"0xd\00i\003\0032\0032\000\000", metadata !1, metadata !8, metadata !11} ; [ DW_TAG_member ] [i] [line 3, size 32, align 32, offset 0] [from int]
!11 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!12 = metadata !{metadata !"0x2e\00foo\00foo\00\002\000\000\000\006\00256\000\002", metadata !1, metadata !8, metadata !13, null, null, null, i32 0, metadata !18} ; [ DW_TAG_subprogram ] [line 2] [foo]
!13 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !14, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!14 = metadata !{null, metadata !15, metadata !16}
!15 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from foo]
!16 = metadata !{metadata !"0x10\00\000\000\000\000\000", null, null, metadata !17} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{metadata !"0x26\00\000\000\000\000\000", null, null, metadata !8} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from foo]
!18 = metadata !{i32 786468}
!19 = metadata !{metadata !"0x101\00f\0016777222\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [f] [line 6]
!20 = metadata !{i32 6, i32 0, metadata !4, null}
!21 = metadata !{metadata !"0x101\00g\0033554438\000", metadata !4, metadata !5, metadata !8} ; [ DW_TAG_arg_variable ] [g] [line 6]
!22 = metadata !{i32 7, i32 0, metadata !4, null}
!23 = metadata !{i32 8, i32 0, metadata !4, null}
!24 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
