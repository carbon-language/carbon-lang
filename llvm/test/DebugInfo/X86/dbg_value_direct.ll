; RUN: llc -filetype=obj -O0 -stack-protector-buffer-size=1 < %s
; Test that we handle DBG_VALUEs in a register without crashing.
;
; Generated from: (with -fsanitize=address)
; class C
; {
; public:
;   C (const C &)
;   { }
; };
; class A
; {
;   struct D:C {};
;   D _M_dataplus;
; public:
;   A (int *);
; };
; A operator+ (A, const char *)
; {
;   A a(0);
;   return a;
; }

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.9.0"

%class.A = type { %"struct.A::D" }
%"struct.A::D" = type { i8 }

@__asan_mapping_offset = linkonce_odr constant i64 17592186044416
@__asan_mapping_scale = linkonce_odr constant i64 3
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 1, void ()* @asan.module_ctor }]
@__asan_gen_ = private unnamed_addr constant [16 x i8] c"1 32 8 5 .addr \00", align 1

; Function Attrs: ssp sanitize_address uwtable
define void @_Zpl1APKc(%class.A* noalias sret %agg.result, %class.A*, i8*) #0 {
entry:
  %MyAlloca = alloca [96 x i8], align 32
  %2 = ptrtoint [96 x i8]* %MyAlloca to i64
  %3 = add i64 %2, 32
  %4 = inttoptr i64 %3 to i8**
  %5 = inttoptr i64 %2 to i64*
  store i64 1102416563, i64* %5
  %6 = add i64 %2, 8
  %7 = inttoptr i64 %6 to i64*
  store i64 ptrtoint ([16 x i8]* @__asan_gen_ to i64), i64* %7
  %8 = add i64 %2, 16
  %9 = inttoptr i64 %8 to i64*
  store i64 ptrtoint (void (%class.A*, %class.A*, i8*)* @_Zpl1APKc to i64), i64* %9
  %10 = lshr i64 %2, 3
  %11 = add i64 %10, 17592186044416
  %12 = inttoptr i64 %11 to i32*
  store i32 -235802127, i32* %12
  %13 = add i64 %11, 4
  %14 = inttoptr i64 %13 to i32*
  store i32 -185273344, i32* %14
  %15 = add i64 %11, 8
  %16 = inttoptr i64 %15 to i32*
  store i32 -202116109, i32* %16
  call void @llvm.dbg.declare(metadata !{%class.A* %0}, metadata !47), !dbg !48
  %17 = ptrtoint i8** %4 to i64
  %18 = lshr i64 %17, 3
  %19 = add i64 %18, 17592186044416
  %20 = inttoptr i64 %19 to i8*
  %21 = load i8* %20
  %22 = icmp ne i8 %21, 0
  call void @llvm.dbg.declare(metadata !{i8** %4}, metadata !49)
  br i1 %22, label %23, label %24

; <label>:23                                      ; preds = %entry
  call void @__asan_report_store8(i64 %17)
  call void asm sideeffect "", ""()
  unreachable

; <label>:24                                      ; preds = %entry
  store i8* %1, i8** %4, align 8
  call void @llvm.dbg.declare(metadata !{%class.A* %agg.result}, metadata !50), !dbg !51
  call void @_ZN1AC1EPi(%class.A* %agg.result, i32* null), !dbg !51
  store i64 1172321806, i64* %5, !dbg !52
  %25 = inttoptr i64 %11 to i32*, !dbg !52
  store i32 0, i32* %25, !dbg !52
  %26 = add i64 %11, 4, !dbg !52
  %27 = inttoptr i64 %26 to i32*, !dbg !52
  store i32 0, i32* %27, !dbg !52
  %28 = add i64 %11, 8, !dbg !52
  %29 = inttoptr i64 %28 to i32*, !dbg !52
  store i32 0, i32* %29, !dbg !52
  ret void, !dbg !52
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

declare void @_ZN1AC1EPi(%class.A*, i32*)

define internal void @asan.module_ctor() {
  call void @__asan_init_v3()
  %1 = load volatile i64* @__asan_mapping_offset
  %2 = load volatile i64* @__asan_mapping_scale
  ret void
}

declare void @__asan_init_v3()

declare void @__asan_report_load1(i64)

declare void @__asan_report_load2(i64)

declare void @__asan_report_load4(i64)

declare void @__asan_report_load8(i64)

declare void @__asan_report_load16(i64)

declare void @__asan_report_store1(i64)

declare void @__asan_report_store2(i64)

declare void @__asan_report_store4(i64)

declare void @__asan_report_store8(i64)

declare void @__asan_report_store16(i64)

declare void @__asan_report_load_n(i64, i64)

declare void @__asan_report_store_n(i64, i64)

declare void @__asan_handle_no_return()

declare i64 @__asan_stack_malloc(i64, i64)

declare void @__asan_stack_free(i64, i64, i64)

declare void @__asan_poison_stack_memory(i64, i64)

declare void @__asan_unpoison_stack_memory(i64, i64)

declare void @__asan_before_dynamic_init(i64)

declare void @__asan_after_dynamic_init()

declare void @__asan_register_globals(i64, i64)

declare void @__asan_unregister_globals(i64, i64)

attributes #0 = { ssp sanitize_address uwtable }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!46}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"operator+", metadata !"operator+", metadata !"_Zpl1APKc", i32 14, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A*, %class.A*, i8*)* @_Zpl1APKc, null, null, metadata !2, i32 15} ; [ DW_TAG_subprogram ] [line 14] [def] [scope 15] [operator+]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] []
!6 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8, metadata !8, metadata !43}
!8 = metadata !{i32 786434, metadata !1, null, metadata !"A", i32 7, i64 8, i64 8, i32 0, i32 0, null, metadata !9, i32 0, null, null} ; [ DW_TAG_class_type ] [A] [line 7, size 8, align 8, offset 0] [def] [from ]
!9 = metadata !{metadata !10, metadata !30, metadata !37}
!10 = metadata !{i32 786445, metadata !1, metadata !8, metadata !"_M_dataplus", i32 10, i64 8, i64 8, i64 0, i32 1, metadata !11} ; [ DW_TAG_member ] [_M_dataplus] [line 10, size 8, align 8, offset 0] [private] [from D]
!11 = metadata !{i32 786451, metadata !1, metadata !8, metadata !"D", i32 9, i64 8, i64 8, i32 0, i32 0, null, metadata !12, i32 0, null, null} ; [ DW_TAG_structure_type ] [D] [line 9, size 8, align 8, offset 0] [def] [from ]
!12 = metadata !{metadata !13, metadata !23}
!13 = metadata !{i32 786460, null, metadata !11, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !14} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from C]
!14 = metadata !{i32 786434, metadata !1, null, metadata !"C", i32 1, i64 8, i64 8, i32 0, i32 0, null, metadata !15, i32 0, null, null} ; [ DW_TAG_class_type ] [C] [line 1, size 8, align 8, offset 0] [def] [from ]
!15 = metadata !{metadata !16}
!16 = metadata !{i32 786478, metadata !1, metadata !14, metadata !"C", metadata !"C", metadata !"", i32 4, metadata !17, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !22, i32 4} ; [ DW_TAG_subprogram ] [line 4] [C]
!17 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19, metadata !20}
!19 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !14} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from C]
!20 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !21} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!21 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !14} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from C]
!22 = metadata !{i32 786468}
!23 = metadata !{i32 786478, metadata !1, metadata !11, metadata !"D", metadata !"D", metadata !"", i32 9, metadata !24, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !29, i32 9} ; [ DW_TAG_subprogram ] [line 9] [D]
!24 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !25, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!25 = metadata !{null, metadata !26, metadata !27}
!26 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !11} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from D]
!27 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !28} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!28 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !11} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from D]
!29 = metadata !{i32 786468}
!30 = metadata !{i32 786478, metadata !1, metadata !8, metadata !"A", metadata !"A", metadata !"", i32 12, metadata !31, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !36, i32 12} ; [ DW_TAG_subprogram ] [line 12] [A]
!31 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !32, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!32 = metadata !{null, metadata !33, metadata !34}
!33 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A]
!34 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !35} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!35 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!36 = metadata !{i32 786468}
!37 = metadata !{i32 786478, metadata !1, metadata !8, metadata !"A", metadata !"A", metadata !"", i32 7, metadata !38, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !42, i32 7} ; [ DW_TAG_subprogram ] [line 7] [A]
!38 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !39, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!39 = metadata !{null, metadata !33, metadata !40}
!40 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !41} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!41 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !8} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from A]
!42 = metadata !{i32 786468}
!43 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !44} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!44 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !45} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from char]
!45 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!46 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!47 = metadata !{i32 786689, metadata !4, metadata !"", metadata !5, i32 16777230, metadata !8, i32 8192, i32 0} ; [ DW_TAG_arg_variable ] [line 14]
!48 = metadata !{i32 14, i32 0, metadata !4, null}
!49 = metadata !{i32 786689, metadata !4, metadata !"", metadata !5, i32 33554446, metadata !43, i32 0, i32 0, i64 2} ; [ DW_TAG_arg_variable ] [line 14]
!50 = metadata !{i32 786688, metadata !4, metadata !"a", metadata !5, i32 16, metadata !8, i32 8192, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 16]
!51 = metadata !{i32 16, i32 0, metadata !4, null}
!52 = metadata !{i32 17, i32 0, metadata !4, null}
