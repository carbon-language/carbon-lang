; RUN: llc -filetype=obj -O0 -stack-protector-buffer-size=1 < %s
; Test that we handle DBG_VALUEs in a register without crashing.
;
; Generated and reduced from: (with -fsanitize=address)
; class C;
; template < typename, typename = int, typename = C > class A;
; class B
; {
; };
; class C:B
; {
; public:
;     C (const C &):B ()
;     {
;     }
; };
; template < typename _CharT, typename, typename _Alloc > class A
; {
;     struct D:_Alloc
;     {
;     };
;     D _M_dataplus;
; public:
;     A (_CharT *);
; };
;
; template < typename _CharT, typename _Traits,
;          typename _Alloc > A < _CharT > operator+ (A < _Traits, _Alloc >,
;                  const _CharT *)
; {
;     A < _CharT > a (0);
;     return a;
; }
;
; int
; main ()
; {
;     A < int >b = 0;
;     A < char >c = b + "/glob_test_root/*a";
; }

%class.A = type { %"struct.A<int, int, C>::D" }
%"struct.A<int, int, C>::D" = type { i8 }
%class.A.0 = type { %"struct.A<char, int, C>::D" }
%"struct.A<char, int, C>::D" = type { i8 }
define i32 @main() {
  ret i32 0, !dbg !103
}
declare void @llvm.dbg.declare(metadata, metadata)
define linkonce_odr void @_ZplIciiE1AIT_i1CES0_IT0_T1_S2_EPKS1_(%class.A.0* noalias sret %agg.result, %class.A*, i8*) {
entry:
  %MyAlloca = alloca [96 x i8], align 32
  %2 = ptrtoint [96 x i8]* %MyAlloca to i64
  %3 = add i64 %2, 32
  %4 = inttoptr i64 %3 to i8**
  %5 = inttoptr i64 %2 to i64*
  %6 = add i64 %2, 8
  %7 = inttoptr i64 %6 to i64*
  %8 = add i64 %2, 16
  %9 = inttoptr i64 %8 to i64*
  %10 = lshr i64 %2, 3
  %11 = add i64 %10, 17592186044416
  %12 = inttoptr i64 %11 to i32*
  %13 = add i64 %11, 4
  %14 = inttoptr i64 %13 to i32*
  %15 = add i64 %11, 8
  %16 = inttoptr i64 %15 to i32*
  %17 = ptrtoint i8** %4 to i64
  %18 = lshr i64 %17, 3
  %19 = add i64 %18, 17592186044416
  %20 = inttoptr i64 %19 to i8*
  %21 = load i8* %20
  %22 = icmp ne i8 %21, 0
  br i1 %22, label %23, label %24
  unreachable
  call void @llvm.dbg.declare(metadata !{%class.A.0* %agg.result}, metadata !107), !dbg !108
  call void @_ZN1AIci1CEC1EPc(%class.A.0* %agg.result, i8* null), !dbg !108
  ret void, !dbg !109
}
declare void @_ZN1AIci1CEC1EPc(%class.A.0*, i8*)
!llvm.dbg.cu = !{!0}
!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.4 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !2, metadata !2, metadata !""} ; [ DW_TAG_compile_unit ] [/1.ii] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"1.ii", metadata !""}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4, metadata !9}
!4 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"main", metadata !"main", metadata !"", i32 32, metadata !6, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 ()* @main, null, null, metadata !2, i32 33} ; [ DW_TAG_subprogram ] [line 32] [def] [scope 33] [main]
!5 = metadata !{i32 786473, metadata !1}          ; [ DW_TAG_file_type ] [/1.ii]
!6 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !7, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!7 = metadata !{metadata !8}
!8 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!9 = metadata !{i32 786478, metadata !1, metadata !5, metadata !"operator+<char, int, int>", metadata !"operator+<char, int, int>", metadata !"_ZplIciiE1AIT_i1CES0_IT0_T1_S2_EPKS1_", i32 24, metadata !10, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void (%class.A.0*, %class.A*, i8*)* @_ZplIciiE1AIT_i1CES0_IT0_T1_S2_EPKS1_, metadata !90, null, metadata !2, i32 26} ; [ DW_TAG_subprogram ] [line 24] [def] [scope 26] [operator+<char, int, int>]
!10 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !11, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!11 = metadata !{metadata !12, metadata !57, metadata !88}
!12 = metadata !{i32 786434, metadata !1, null, metadata !"A<char, int, C>", i32 13, i64 8, i64 8, i32 0, i32 0, null, metadata !13, i32 0, null, metadata !53} ; [ DW_TAG_class_type ] [A<char, int, C>] [line 13, size 8, align 8, offset 0] [def] [from ]
!13 = metadata !{metadata !14, metadata !36, metadata !43, metadata !47}
!14 = metadata !{i32 786445, metadata !1, metadata !12, metadata !"_M_dataplus", i32 18, i64 8, i64 8, i64 0, i32 1, metadata !15} ; [ DW_TAG_member ] [_M_dataplus] [line 18, size 8, align 8, offset 0] [private] [from D]
!15 = metadata !{i32 786451, metadata !1, metadata !12, metadata !"D", i32 15, i64 8, i64 8, i32 0, i32 0, null, metadata !16, i32 0, null, null} ; [ DW_TAG_structure_type ] [D] [line 15, size 8, align 8, offset 0] [def] [from ]
!16 = metadata !{metadata !17, metadata !29}
!17 = metadata !{i32 786460, null, metadata !15, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !18} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from C]
!18 = metadata !{i32 786434, metadata !1, null, metadata !"C", i32 6, i64 8, i64 8, i32 0, i32 0, null, metadata !19, i32 0, null, null} ; [ DW_TAG_class_type ] [C] [line 6, size 8, align 8, offset 0] [def] [from ]
!19 = metadata !{metadata !20, metadata !22}
!20 = metadata !{i32 786460, null, metadata !18, null, i32 0, i64 0, i64 0, i64 0, i32 1, metadata !21} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [private] [from B]
!21 = metadata !{i32 786434, metadata !1, null, metadata !"B", i32 3, i64 8, i64 8, i32 0, i32 0, null, metadata !2, i32 0, null, null} ; [ DW_TAG_class_type ] [B] [line 3, size 8, align 8, offset 0] [def] [from ]
!22 = metadata !{i32 786478, metadata !1, metadata !18, metadata !"C", metadata !"C", metadata !"", i32 9, metadata !23, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !28, i32 9} ; [ DW_TAG_subprogram ] [line 9] [C]
!23 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !24, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!24 = metadata !{null, metadata !25, metadata !26}
!25 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !18} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from C]
!26 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !27} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !18} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from C]
!28 = metadata !{i32 786468}
!29 = metadata !{i32 786478, metadata !1, metadata !15, metadata !"D", metadata !"D", metadata !"", i32 15, metadata !30, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !35, i32 15} ; [ DW_TAG_subprogram ] [line 15] [D]
!30 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !31, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!31 = metadata !{null, metadata !32, metadata !33}
!32 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !15} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from D]
!33 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !34} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!34 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !15} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from D]
!35 = metadata !{i32 786468}
!36 = metadata !{i32 786478, metadata !1, metadata !12, metadata !"A", metadata !"A", metadata !"", i32 20, metadata !37, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !42, i32 20} ; [ DW_TAG_subprogram ] [line 20] [A]
!37 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !38, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!38 = metadata !{null, metadata !39, metadata !40}
!39 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !12} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A<char, int, C>]
!40 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !41} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from char]
!41 = metadata !{i32 786468, null, null, metadata !"char", i32 0, i64 8, i64 8, i64 0, i32 0, i32 6} ; [ DW_TAG_base_type ] [char] [line 0, size 8, align 8, offset 0, enc DW_ATE_signed_char]
!42 = metadata !{i32 786468}
!43 = metadata !{i32 786478, metadata !1, metadata !12, metadata !"~A", metadata !"~A", metadata !"", i32 13, metadata !44, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !46, i32 13} ; [ DW_TAG_subprogram ] [line 13] [~A]
!44 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !45, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!45 = metadata !{null, metadata !39}
!46 = metadata !{i32 786468}
!47 = metadata !{i32 786478, metadata !1, metadata !12, metadata !"A", metadata !"A", metadata !"", i32 13, metadata !48, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !52, i32 13} ; [ DW_TAG_subprogram ] [line 13] [A]
!48 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !49, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!49 = metadata !{null, metadata !39, metadata !50}
!50 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !51} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!51 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !12} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from A<char, int, C>]
!52 = metadata !{i32 786468}
!53 = metadata !{metadata !54, metadata !55, metadata !56}
!54 = metadata !{i32 786479, null, metadata !"_CharT", metadata !41, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!55 = metadata !{i32 786479, null, metadata !"", metadata !8, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!56 = metadata !{i32 786479, null, metadata !"_Alloc", metadata !18, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!57 = metadata !{i32 786434, metadata !1, null, metadata !"A<int, int, C>", i32 13, i64 8, i64 8, i32 0, i32 0, null, metadata !58, i32 0, null, metadata !86} ; [ DW_TAG_class_type ] [A<int, int, C>] [line 13, size 8, align 8, offset 0] [def] [from ]
!58 = metadata !{metadata !59, metadata !70, metadata !76, metadata !82}
!59 = metadata !{i32 786445, metadata !1, metadata !57, metadata !"_M_dataplus", i32 18, i64 8, i64 8, i64 0, i32 1, metadata !60} ; [ DW_TAG_member ] [_M_dataplus] [line 18, size 8, align 8, offset 0] [private] [from D]
!60 = metadata !{i32 786451, metadata !1, metadata !57, metadata !"D", i32 15, i64 8, i64 8, i32 0, i32 0, null, metadata !61, i32 0, null, null} ; [ DW_TAG_structure_type ] [D] [line 15, size 8, align 8, offset 0] [def] [from ]
!61 = metadata !{metadata !62, metadata !63}
!62 = metadata !{i32 786460, null, metadata !60, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !18} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [from C]
!63 = metadata !{i32 786478, metadata !1, metadata !60, metadata !"D", metadata !"D", metadata !"", i32 15, metadata !64, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !69, i32 15} ; [ DW_TAG_subprogram ] [line 15] [D]
!64 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !65, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!65 = metadata !{null, metadata !66, metadata !67}
!66 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !60} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from D]
!67 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !68} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!68 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !60} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from D]
!69 = metadata !{i32 786468}
!70 = metadata !{i32 786478, metadata !1, metadata !57, metadata !"A", metadata !"A", metadata !"", i32 20, metadata !71, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, metadata !75, i32 20} ; [ DW_TAG_subprogram ] [line 20] [A]
!71 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !72, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!72 = metadata !{null, metadata !73, metadata !74}
!73 = metadata !{i32 786447, i32 0, i32 0, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !57} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from A<int, int, C>]
!74 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !8} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from int]
!75 = metadata !{i32 786468}
!76 = metadata !{i32 786478, metadata !1, metadata !57, metadata !"A", metadata !"A", metadata !"", i32 13, metadata !77, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !81, i32 13} ; [ DW_TAG_subprogram ] [line 13] [A]
!77 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !78, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!78 = metadata !{null, metadata !73, metadata !79}
!79 = metadata !{i32 786448, null, null, null, i32 0, i64 0, i64 0, i64 0, i32 0, metadata !80} ; [ DW_TAG_reference_type ] [line 0, size 0, align 0, offset 0] [from ]
!80 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !57} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from A<int, int, C>]
!81 = metadata !{i32 786468}
!82 = metadata !{i32 786478, metadata !1, metadata !57, metadata !"~A", metadata !"~A", metadata !"", i32 13, metadata !83, i1 false, i1 false, i32 0, i32 0, null, i32 320, i1 false, null, null, i32 0, metadata !85, i32 13} ; [ DW_TAG_subprogram ] [line 13] [~A]
!83 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !84, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!84 = metadata !{null, metadata !73}
!85 = metadata !{i32 786468}
!86 = metadata !{metadata !87, metadata !55, metadata !56}
!87 = metadata !{i32 786479, null, metadata !"_CharT", metadata !8, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!88 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 0, metadata !89} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [from ]
!89 = metadata !{i32 786470, null, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, metadata !41} ; [ DW_TAG_const_type ] [line 0, size 0, align 0, offset 0] [from char]
!90 = metadata !{metadata !54, metadata !91, metadata !92}
!91 = metadata !{i32 786479, null, metadata !"_Traits", metadata !8, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!92 = metadata !{i32 786479, null, metadata !"_Alloc", metadata !8, null, i32 0, i32 0} ; [ DW_TAG_template_type_parameter ]
!103 = metadata !{i32 36, i32 0, metadata !4, null}
!107 = metadata !{i32 786688, metadata !9, metadata !"a", metadata !5, i32 27, metadata !12, i32 8192, i32 0} ; [ DW_TAG_auto_variable ] [a] [line 27]
!108 = metadata !{i32 27, i32 0, metadata !9, null}
!109 = metadata !{i32 28, i32 0, metadata !9, null}
