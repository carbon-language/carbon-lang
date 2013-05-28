; RUN: llc -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s

; rdar://13926659
; When a callee from a different CU is inlined, we should generate
; inlined_subroutine with abstract_origin pointing to a DIE in a different
; CU.
; CHECK: debug_info contents
; CHECK: DW_TAG_inlined_subroutine
; CHECK: DW_AT_abstract_origin [DW_FORM_ref_addr] (0x{{[0]*}}[[ORIGIN:.*]])
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_abstract_origin [DW_FORM_ref_addr] (0x{{[0]*}}[[PARAM_ORIGIN:.*]])
; CHECK: 0x{{[0]*}}[[ORIGIN]]: DW_TAG_subprogram
; CHECK: DW_AT_name {{.*}} ( .debug_str[{{.*}}] = "bar2")
; CHECK: DW_AT_inline
; CHECK: 0x{{[0]*}}[[PARAM_ORIGIN]]: DW_TAG_formal_parameter
; CHECK: DW_TAG_subprogram
; CHECK: DW_AT_abstract_origin [DW_FORM_ref4] (cu + {{.*}} => {0x{{[0]*}}[[ORIGIN]]})

target triple = "x86_64-apple-macosx"
%struct.Foo = type { i32, i32 }

; Function Attrs: noinline ssp uwtable
define i32 @_Z3fooi(i32 %a) #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %g = alloca %struct.Foo, align 4
  store i32 %a, i32* %2, align 4
  call void @llvm.dbg.declare(metadata !{i32* %2}, metadata !17), !dbg !18
  call void @llvm.dbg.declare(metadata !{%struct.Foo* %g}, metadata !19), !dbg !24
  %3 = getelementptr inbounds %struct.Foo* %g, i32 0, i32 0, !dbg !24
  %4 = load i32* %2, align 4, !dbg !24
  store i32 %4, i32* %3, align 4, !dbg !24
  %5 = getelementptr inbounds %struct.Foo* %g, i32 0, i32 1, !dbg !24
  %6 = load i32* %2, align 4, !dbg !24
  store i32 %6, i32* %5, align 4, !dbg !24
  %7 = load i32* %2, align 4, !dbg !25
  %8 = load i32* %2, align 4, !dbg !25
  %9 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.start(i64 4, i8* %9)
  store i32 %8, i32* %1, align 4
  call void @llvm.dbg.declare(metadata !{i32* %1}, metadata !26), !dbg !27
  %10 = load i32* %1, align 4, !dbg !28
  %11 = mul nsw i32 2, %10, !dbg !28
  %12 = bitcast i32* %1 to i8*, !dbg !28
  call void @llvm.lifetime.end(i64 4, i8* %12), !dbg !28
  %13 = add nsw i32 %7, %11, !dbg !25
  ret i32 %13, !dbg !25
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

; Function Attrs: alwaysinline nounwind ssp uwtable
define i32 @_Z4bar2i(i32 %ab) #2 {
  %1 = alloca i32, align 4
  store i32 %ab, i32* %1, align 4
  call void @llvm.dbg.declare(metadata !{i32* %1}, metadata !26), !dbg !29
  %2 = load i32* %1, align 4, !dbg !30
  %3 = mul nsw i32 2, %2, !dbg !30
  ret i32 %3, !dbg !30
}

; Function Attrs: ssp uwtable
define i32 @main() #3 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1
  %2 = call i32 @_Z3fooi(i32 44), !dbg !31
  ret i32 %2, !dbg !31
}

; Function Attrs: nounwind
declare void @llvm.lifetime.start(i64, i8* nocapture) #4

; Function Attrs: nounwind
declare void @llvm.lifetime.end(i64, i8* nocapture) #4

attributes #0 = { noinline ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }
attributes #2 = { alwaysinline nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #4 = { nounwind }

!llvm.dbg.cu = !{!0, !9}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 (trunk 182336)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [foo.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"foo.cpp", metadata !"."}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"foo", metadata !"foo", metadata !"_Z3fooi", i32 3, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z3fooi, null, null, metadata !2, i32 3} ; [ DW_TAG_subprogram ] [line 3] [def] [foo]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ]
!6 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786449, metadata !10, i32 4, metadata !"clang version 3.4 (trunk 182336)", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !11, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [bar.cpp] [DW_LANG_C_plus_plus]
!10 = metadata !{metadata !"bar.cpp", metadata !"."}
!11 = metadata !{metadata !12, metadata !14}
!12 = metadata !{i32 786478, metadata !10, metadata !13, metadata !"bar2", metadata !"bar2", metadata !"_Z4bar2i", i32 2, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i32)* @_Z4bar2i, null, null, metadata !2, i32 2} ; [ DW_TAG_subprogram ] [line 2] [def] [bar2]
!13 = metadata !{i32 786473, metadata !10}        ; [ DW_TAG_file_type ]
!14 = metadata !{i32 786478, metadata !10, metadata !13, metadata !"main", metadata !"main", metadata !"", i32 15, metadata !15, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 15} ; [ DW_TAG_subprogram ] [line 15] [def] [main]
!15 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !16, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{metadata !8}
!17 = metadata !{i32 786689, metadata !4, metadata !"a", metadata !5, i32 16777219, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [a] [line 3]
!18 = metadata !{i32 3, i32 0, metadata !4, null}
!19 = metadata !{i32 786688, metadata !4, metadata !"g", metadata !5, i32 9, metadata !20, i32 0, i32 0} ; [ DW_TAG_auto_variable ] [g] [line 9]
!20 = metadata !{i32 786451, metadata !1, metadata !4, metadata !"Foo", i32 4, i64 64, i64 32, i32 0, i32 0, null, metadata !21, i32 0, null, null} ; [ DW_TAG_structure_type ] [Foo] [line 4, size 64, align 32, offset 0] [from ]
!21 = metadata !{metadata !22, metadata !23}
!22 = metadata !{i32 786445, metadata !1, metadata !20, metadata !"a", i32 5, i64 32, i64 32, i64 0, i32 0, metadata !8} ; [ DW_TAG_member ] [a] [line 5, size 32, align 32, offset 0] [from int]
!23 = metadata !{i32 786445, metadata !1, metadata !20, metadata !"b", i32 6, i64 32, i64 32, i64 32, i32 0, metadata !8} ; [ DW_TAG_member ] [b] [line 6, size 32, align 32, offset 32] [from int]
!24 = metadata !{i32 9, i32 0, metadata !4, null}
!25 = metadata !{i32 10, i32 0, metadata !4, null}
!26 = metadata !{i32 786689, metadata !12, metadata !"ab", metadata !13, i32 16777218, metadata !8, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [ab] [line 2]
!27 = metadata !{i32 2, i32 0, metadata !12, metadata !25}
!28 = metadata !{i32 11, i32 0, metadata !12, metadata !25}
!29 = metadata !{i32 2, i32 0, metadata !12, null}
!30 = metadata !{i32 11, i32 0, metadata !12, null}
!31 = metadata !{i32 16, i32 0, metadata !14, null}
