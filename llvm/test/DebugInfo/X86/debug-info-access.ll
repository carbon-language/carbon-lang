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
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)
;
; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"public_static")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"pub")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_public)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"prot")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_protected)
;
; CHECK: DW_TAG_subprogram
; CHECK:     DW_AT_name {{.*}}"priv_default")
; CHECK-NOT: DW_AT_accessibility
; CHECK: DW_TAG
;
; CHECK: DW_TAG_member
; CHECK:     DW_AT_name {{.*}}"union_priv")
; CHECK-NOT: DW_TAG
; CHECK:     DW_AT_accessibility {{.*}}(DW_ACCESS_private)
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

!0 = metadata !{metadata !"0x11\004\00clang version 3.6.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !3, metadata !29, metadata !34, metadata !2} ; [ DW_TAG_compile_unit ] [/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp", metadata !""}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !12, metadata !22}
!4 = metadata !{metadata !"0x13\00A\003\008\008\000\000\000", metadata !1, null, null, metadata !5, null, null, metadata !"_ZTS1A"} ; [ DW_TAG_structure_type ] [A] [line 3, size 8, align 8, offset 0] [def] [from ]
!5 = metadata !{metadata !6, metadata !8}
!6 = metadata !{metadata !"0xd\00pub_default_static\007\000\000\000\004096", metadata !1, metadata !"_ZTS1A", metadata !7, null} ; [ DW_TAG_member ] [pub_default_static] [line 7, size 0, align 0, offset 0] [static] [from int]
!7 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!8 = metadata !{metadata !"0x2e\00pub_default\00pub_default\00_ZN1A11pub_defaultEv\005\000\000\000\006\00256\000\005", metadata !1, metadata !"_ZTS1A", metadata !9, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 5] [pub_default]
!9 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !10, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!10 = metadata !{null, metadata !11}
!11 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", null, null, metadata !"_ZTS1A"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1A]
!12 = metadata !{metadata !"0x2\00B\0011\008\008\000\000\000", metadata !1, null, null, metadata !13, null, null, metadata !"_ZTS1B"} ; [ DW_TAG_class_type ] [B] [line 11, size 8, align 8, offset 0] [def] [from ]
!13 = metadata !{metadata !14, metadata !15, metadata !16, metadata !20, metadata !21}
!14 = metadata !{metadata !"0x1c\00\000\000\000\000\003", null, metadata !"_ZTS1B", metadata !"_ZTS1A"} ; [ DW_TAG_inheritance ] [line 0, size 0, align 0, offset 0] [public] [from _ZTS1A]
!15 = metadata !{metadata !"0xd\00public_static\0016\000\000\000\004099", metadata !1, metadata !"_ZTS1B", metadata !7, null} ; [ DW_TAG_member ] [public_static] [line 16, size 0, align 0, offset 0] [public] [static] [from int]
!16 = metadata !{metadata !"0x2e\00pub\00pub\00_ZN1B3pubEv\0014\000\000\000\006\00259\000\0014", metadata !1, metadata !"_ZTS1B", metadata !17, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 14] [public] [pub]
!17 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !18, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!18 = metadata !{null, metadata !19}
!19 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", null, null, metadata !"_ZTS1B"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1B]
!20 = metadata !{metadata !"0x2e\00prot\00prot\00_ZN1B4protEv\0019\000\000\000\006\00258\000\0019", metadata !1, metadata !"_ZTS1B", metadata !17, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 19] [protected] [prot]
!21 = metadata !{metadata !"0x2e\00priv_default\00priv_default\00_ZN1B12priv_defaultEv\0022\000\000\000\006\00256\000\0022", metadata !1, metadata !"_ZTS1B", metadata !17, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 22] [priv_default]
!22 = metadata !{metadata !"0x17\00U\0025\0032\0032\000\000\000", metadata !1, null, null, metadata !23, null, null, metadata !"_ZTS1U"} ; [ DW_TAG_union_type ] [U] [line 25, size 32, align 32, offset 0] [def] [from ]
!23 = metadata !{metadata !24, metadata !25}
!24 = metadata !{metadata !"0xd\00union_priv\0030\0032\0032\000\001", metadata !1, metadata !"_ZTS1U", metadata !7} ; [ DW_TAG_member ] [union_priv] [line 30, size 32, align 32, offset 0] [private] [from int]
!25 = metadata !{metadata !"0x2e\00union_pub_default\00union_pub_default\00_ZN1U17union_pub_defaultEv\0027\000\000\000\006\00256\000\0027", metadata !1, metadata !"_ZTS1U", metadata !26, null, null, null, i32 0, null} ; [ DW_TAG_subprogram ] [line 27] [union_pub_default]
!26 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !27, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!27 = metadata !{null, metadata !28}
!28 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", null, null, metadata !"_ZTS1U"} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from _ZTS1U]
!29 = metadata !{metadata !30}
!30 = metadata !{metadata !"0x2e\00free\00free\00_Z4freev\0035\000\001\000\006\00256\000\0035", metadata !1, metadata !31, metadata !32, null, void ()* @_Z4freev, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 35] [def] [free]
!31 = metadata !{metadata !"0x29", metadata !1}         ; [ DW_TAG_file_type ] [/llvm/tools/clang/test/CodeGenCXX/debug-info-access.cpp]
!32 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !33, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!33 = metadata !{null}
!34 = metadata !{metadata !35, metadata !36, metadata !37}
!35 = metadata !{metadata !"0x34\00a\00a\00\0037\000\001", null, metadata !31, metadata !"_ZTS1A", %struct.A* @a, null} ; [ DW_TAG_variable ] [a] [line 37] [def]
!36 = metadata !{metadata !"0x34\00b\00b\00\0038\000\001", null, metadata !31, metadata !"_ZTS1B", %class.B* @b, null} ; [ DW_TAG_variable ] [b] [line 38] [def]
!37 = metadata !{metadata !"0x34\00u\00u\00\0039\000\001", null, metadata !31, metadata !"_ZTS1U", %union.U* @u, null} ; [ DW_TAG_variable ] [u] [line 39] [def]
!38 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!39 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!40 = metadata !{metadata !"clang version 3.6.0 "}
!41 = metadata !{i32 35, i32 14, metadata !30, null}
