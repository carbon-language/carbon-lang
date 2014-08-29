; RUN: llc -mtriple=x86_64-apple-darwin %s -o %t -filetype=obj
; RUN: llvm-dwarfdump -debug-dump=info %t | FileCheck %s
;
; Test the DW_AT_accessibility DWARF attribute.
;
;
; Regenerate me:
; clang++ -g tools/clang/test/CodeGenCXX/debug-info-access.cpp -S -emit-llvm -o -
;
;   struct A {
;     void pub_default();
;     static int pub_default_static;
;   };
;
;   class B : public A {
;   public:
;     void pub();
;     static int public_static;
;   protected:
;     void prot();
;   private:
;     void priv_default();
;   };
;
;   union U {
;     void union_pub_default();
;   private:
;     int union_priv;
;   };
;
;   void free() {}
;
;   A a;
;   B b;
;   U u;

; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"pub_default_static")
; CHECK-NOT: DW_AT_accessibility
; CHECK-NOT: DW_TAG
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"pub_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG
;
; CHECK: DW_TAG_inheritance
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(0x01)
;
; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"public_static")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(0x01)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"pub")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(0x01)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"prot")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(0x02)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"priv_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG
;
; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"union_priv")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(0x03)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"union_pub_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"free")
; CHECK-NOT: DW_AT_accessibility
; CHECK-NOT: DW_TAG
;
; ModuleID = '/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp'
target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

%struct.A = type { i8 }
%class.B = type { i8 }
%union.U = type { i32 }

@a = global %struct.A zeroinitializer, align 1
@b = global %class.B zeroinitializer, align 1
@u = global %union.U zeroinitializer, align 4

; Function Attrs: nounwind ssp uwtable
define void @_Z4freev() #0 {
  ret void, !dbg !41
}

attributes #0 = { nounwind ssp uwtable }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!38, !39}
!llvm.ident = !{!40}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.6.0 ", i1 false, metadata !"", i32 0, metadata !2, metadata !3, metadata !29, metadata !34, metadata !2, metadata !"", i32 1} ; [ DW_TAG_compile_unit ] [/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !12, metadata !22}
!4 = metadata !{i32 786451, metadata !1, null, metadata !"A", i32 3, i64 8, i64 8, i32 0, i32 0, null, metadata !5, i32 0, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 3, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !8}
!6 = metadata !{i32 786445, metadata !1, metadata !"_ZTS1A", metadata !"pub_default_static", i32 7, i64 0, i64 0, i64 0, i32 4096, metadata !7, null} ; [ DW_TAG_member ] [pub_default_static] [line 7, size 0, align 0, offset 0] [static] [from int]
!7 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1A", metadata !"pub_default", metadata !"pub_default", metadata !"_ZN1A11pub_defaultEv", i32 5, metadata !9, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, null, i32 5} ; [ DW_TAG_subprogram ] [line 5] [pub_default]
!9 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !10, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11}
!11 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!12 = metadata !{i32 786434, metadata !1, null, metadata !"B", i32 11, i64 8, i64 8, i32 0, i32 0, null, metadata !13, i32 0, null, null, metadata !"_ZTS1B"} ; [ DW_TAG_class_type ] [B] [line 11, size 8, align 8, offset 0] [def] [from ]
!13 = metadata !{metadata !14, metadata !15, metadata !16, metadata !20, metadata !21}
!14 = metadata !{i32 786460, null, metadata !"_ZTS1B", null, i32 0, i64 0, i64 0, i64 0, i32 3, metadata !"_ZTS1A"} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [public] [from _ZTS1A]
!15 = metadata !{i32 786445, metadata !1, metadata !"_ZTS1B", metadata !"public_static", i32 16, i64 0, i64 0, i64 0, i32 4099, metadata !7, null} ; [ DW_TAG_member ] [public_static] [line 16, size 0, align 0, offset 0] [public] [static] [from int]
!16 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1B", metadata !"pub", metadata !"pub", metadata !"_ZN1B3pubEv", i32 14, metadata !17, i1 false, i1 false, i32 0, i32 0, null, i32 259, i1 false, null, null, i32 0, null, i32 14} ; [ DW_TAG_subprogram ] [line 14] [public] [pub]
!17 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !18, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19}
!19 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1B"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1B]
!20 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1B", metadata !"prot", metadata !"prot", metadata !"_ZN1B4protEv", i32 19, metadata !17, i1 false, i1 false, i32 0, i32 0, null, i32 258, i1 false, null, null, i32 0, null, i32 19} ; [ DW_TAG_subprogram ] [line 19] [protected] [prot]
!21 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1B", metadata !"priv_default", metadata !"priv_default", metadata !"_ZN1B12priv_defaultEv", i32 22, metadata !17, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, null, i32 22} ; [ DW_TAG_subprogram ] [line 22] [priv_default]
!22 = metadata !{i32 786455, metadata !1, null, metadata !"U", i32 25, i64 32, i64 32, i64 0, i32 0, null, metadata !23, i32 0, null, null, metadata !"_ZTS1U"} ; [ DW_TAG_union_type ] [U] [line 25, size 32, align 32, offset 0] [def] [from ]
!23 = metadata !{metadata !24, metadata !25}
!24 = metadata !{i32 786445, metadata !1, metadata !"_ZTS1U", metadata !"union_priv", i32 30, i64 32, i64 32, i64 0, i32 1, metadata !7} ; [ DW_TAG_member ] [union_priv] [line 30, size 32, align 32, offset 0] [private] [from int]
!25 = metadata !{i32 786478, metadata !1, metadata !"_ZTS1U", metadata !"union_pub_default", metadata !"union_pub_default", metadata !"_ZN1U17union_pub_defaultEv", i32 27, metadata !26, i1 false, i1 false, i32 0, i32 0, null, i32 256, i1 false, null, null, i32 0, null, i32 27} ; [ DW_TAG_subprogram ] [line 27] [union_pub_default]
!26 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !27, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = metadata !{null, metadata !28}
!28 = metadata !{i32 786447, null, null, metadata !"", i32 0, i64 64, i64 64, i64 0, i32 1088, metadata !"_ZTS1U"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1U]
!29 = metadata !{metadata !30}
!30 = metadata !{i32 786478, metadata !1, metadata !31, metadata !"free", metadata !"free", metadata !"_Z4freev", i32 35, metadata !32, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, void ()* @_Z4freev, null, null, metadata !2, i32 35} ; [ DW_TAG_subprogram ] [line 35] [def] [free]
!31 = metadata !{i32 786473, metadata !1}         ; [ DW_TAG_file_type ] [/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp]
!32 = metadata !{i32 786453, i32 0, null, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !33, i32 0, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = metadata !{null}
!34 = metadata !{metadata !35, metadata !36, metadata !37}
!35 = metadata !{i32 786484, i32 0, null, metadata !"a", metadata !"a", metadata !"", metadata !31, i32 37, metadata !"_ZTS1A", i32 0, i32 1, %struct.A* @a, null} ; [ DW_TAG_variable ] [a] [line 37] [def]
!36 = metadata !{i32 786484, i32 0, null, metadata !"b", metadata !"b", metadata !"", metadata !31, i32 38, metadata !"_ZTS1B", i32 0, i32 1, %class.B* @b, null} ; [ DW_TAG_variable ] [b] [line 38] [def]
!37 = metadata !{i32 786484, i32 0, null, metadata !"u", metadata !"u", metadata !"", metadata !31, i32 39, metadata !"_ZTS1U", i32 0, i32 1, %union.U* @u, null} ; [ DW_TAG_variable ] [u] [line 39] [def]
!38 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!39 = metadata !{i32 2, metadata !"Debug Info Version", i32 1}
!40 = metadata !{metadata !"clang version 3.6.0 "}
!41 = metadata !{i32 35, i32 14, metadata !30, null}
