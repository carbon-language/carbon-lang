; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s | llvm-dwarfdump -debug-dump=info - | FileCheck %s

; Generate from clang with the following source. Note that the definition of
; the inline function follows its use to workaround another bug that should be
; fixed soon.
; namespace ns {
; int func(int i);
; }
; extern int x;
; int main() { return ns::func(x); }
; int __attribute__((always_inline)) ns::func(int i) { return i * 2; }

; CHECK: DW_TAG_namespace
; CHECK-NEXT: DW_AT_name {{.*}} "ns"
; CHECK-NOT: DW_TAG
; CHECK: [[ABS_DEF:0x.*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:   DW_AT_MIPS_linkage_name {{.*}} "_ZN2ns4funcEi"
; CHECK-NOT: DW_TAG
; CHECK: [[ABS_PRM:0x.*]]:   DW_TAG_formal_parameter
; CHECK:   NULL
; CHECK-NOT: NULL
; CHECK:   DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_abstract_origin {{.*}} {[[ABS_DEF]]}
; CHECK-NOT: DW_TAG
; CHECK:     DW_TAG_formal_parameter
; CHECK:       DW_AT_abstract_origin {{.*}} {[[ABS_PRM]]}
; CHECK:     NULL
; CHECK:   NULL
; CHECK: NULL

@x = external global i32

; Function Attrs: uwtable
define i32 @main() #0 {
entry:
  %i.addr.i = alloca i32, align 4
  %retval = alloca i32, align 4
  store i32 0, i32* %retval
  %0 = load i32* @x, align 4, !dbg !16
  store i32 %0, i32* %i.addr.i, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i.addr.i}, metadata !17), !dbg !18
  %1 = load i32* %i.addr.i, align 4, !dbg !18
  %mul.i = mul nsw i32 %1, 2, !dbg !18
  ret i32 %mul.i, !dbg !16
}

; Function Attrs: alwaysinline nounwind uwtable
define i32 @_ZN2ns4funcEi(i32 %i) #1 {
entry:
  %i.addr = alloca i32, align 4
  store i32 %i, i32* %i.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %i.addr}, metadata !17), !dbg !19
  %0 = load i32* %i.addr, align 4, !dbg !19
  %mul = mul nsw i32 %0, 2, !dbg !19
  ret i32 %mul, !dbg !19
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #2

attributes #0 = { uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { alwaysinline nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!13, !14}
!llvm.ident = !{!15}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.5.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/tmp/dbginfo/namespace_inline_function_definition.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"namespace_inline_function_definition.cpp", metadata !"/tmp/dbginfo"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !9}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 5, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 5} ; [ DW_TAG_subprogram ] [line 5] [def] [main]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/tmp/dbginfo/namespace_inline_function_definition.cpp]
!6 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786478, metadata !1, metadata !10, metadata !"func", metadata !"func", metadata !"_ZN2ns4funcEi", i32 6, metadata !11, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_ZN2ns4funcEi, null, null, metadata !2, i32 6} ; [ DW_TAG_subprogram ] [line 6] [def] [func]
!10 = metadata !{i32 786489, metadata !1, null, metadata !"ns", i32 1} ; [ DW_TAG_namespace ] [ns] [line 1]
!11 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !12, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !8, metadata !8}
!13 = metadata !{i32 2, metadata !"Dwarf Version", i32 4}
!14 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!15 = metadata !{metadata !"clang version 3.5.0 "}
!16 = metadata !{i32 5, i32 0, metadata !4, null}
!17 = metadata !{i32 786689, metadata !9, metadata !"i", metadata !5, i32 16777222, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [i] [line 6]
!18 = metadata !{i32 6, i32 0, metadata !9, metadata !16}
!19 = metadata !{i32 6, i32 0, metadata !9, null}
