; REQUIRES: object-emission

; RUN: llc -mtriple=x86_64-linux -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; From source:
; struct foo {
;   int __attribute__((always_inline)) func(int x) { return x + 2; }
; };

; int i;

; int main() {
;   return foo().func(i);
; }

; Ensure we omit DW_AT_object_pointer on inlined subroutines.
; CHECK: DW_TAG_inlined_subroutine
; CHECK-NEXT: DW_AT_abstract_origin {{.*}}{[[ABSTRACT_ORIGIN:0x[0-9a-e]*]]}
; CHECK-NOT: NULL
; CHECK-NOT: DW_AT_object_pointer
; CHECK: DW_TAG_formal_parameter
; CHECK-NOT: DW_AT_artificial
; CHECK: DW_TAG

; But make sure we emit DW_AT_object_pointer on the declaration.
; CHECK: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} "foo"
; CHECK-NOT: NULL
; CHECK: [[DECLARATION:0x[0-9a-e]*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_object_pointer

; But don't put it on the abstract definition, either.
; CHECK: [[ABSTRACT_ORIGIN]]: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_specification {{.*}}{[[DECLARATION]]}
; CHECK-NOT: NULL
; CHECK-NOT: DW_AT_object_pointer
; CHECK: DW_TAG_formal_parameter

%struct.foo = type { i8 }

@i = global i32 0, align 4

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  %this.addr.i = alloca %struct.foo*, align 8
  %x.addr.i = alloca i32, align 4
  %retval = alloca i32, align 4
  %tmp = alloca %struct.foo, align 1
  store i32 0, i32* %retval
  %0 = load i32* @i, align 4, !dbg !23
  store %struct.foo* %tmp, %struct.foo** %this.addr.i, align 8
  call void @llvm.dbg.declare(metadata !{%struct.foo** %this.addr.i}, metadata !24), !dbg !26
  store i32 %0, i32* %x.addr.i, align 4
  call void @llvm.dbg.declare(metadata !{i32* %x.addr.i}, metadata !27), !dbg !28
  %this1.i = load %struct.foo** %this.addr.i
  %1 = load i32* %x.addr.i, align 4, !dbg !28
  %add.i = add nsw i32 %1, 2, !dbg !28
  ret i32 %add.i, !dbg !23
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!20, !21}
!llvm.ident = !{!22}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !12, metadata !18, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/inline.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"inline.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"foo", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS3foo"} ; [ DW_TAG_structure_type ] [foo] [line 1, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6}
!6 = metadata !{i32 786478, metadata !1, metadata !"_ZTS3foo", metadata !"func", metadata !"func", metadata !"_ZN3foo4funcEi", i32 2, metadata !7, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !11, i32 2} ; [ DW_TAG_subprogram ] [line 2] [func]
!7 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !10, metadata !9}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS3foo]
!11 = metadata !{i32 786468}
!12 = metadata !{metadata !13, metadata !17}
!13 = metadata !{i32 786478, metadata !1, metadata !14, metadata !"main", metadata !"main", metadata !"", i32 7, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 7} ; [ DW_TAG_subprogram ] [line 7] [def] [main]
!14 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/tmp/dbginfo/inline.cpp]
!15 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{metadata !9}
!17 = metadata !{i32 786478, metadata !1, metadata !"_ZTS3foo", metadata !"func", metadata !"func", metadata !"_ZN3foo4funcEi", i32 2, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, null, null, metadata !6, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [func]
!18 = metadata !{metadata !19}
!19 = metadata !{i32 786484, i32 0, null, metadata !"i", metadata !"i", metadata !"", metadata !14, i32 5, metadata !9, i32 0, i32 1, i32* @i, null} ; [ DW_TAG_variable ] [i] [line 5] [def]
!20 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!21 = metadata !{i32 1, metadata !"Debug Info Version", i32 1}
!22 = metadata !{metadata !"clang version 3.5.0 "}
!23 = metadata !{i32 8, i32 0, metadata !13, null} ; [ DW_TAG_imported_declaration ]
!24 = metadata !{i32 786689, metadata !17, metadata !"this", null, i32 16777216, metadata !25, i32 1088, i32 0} ; [ DW_TAG_arg_variable ] [this] [line 0]
!25 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !"_ZTS3foo"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from _ZTS3foo]
!26 = metadata !{i32 0, i32 0, metadata !17, metadata !23}
!27 = metadata !{i32 786689, metadata !17, metadata !"x", metadata !14, i32 33554434, metadata !9, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [x] [line 2]
!28 = metadata !{i32 2, i32 0, metadata !17, metadata !23}
