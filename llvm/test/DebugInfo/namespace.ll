; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; CHECK: debug_info contents
; CHECK: [[NS1:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "A"
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F1:".*debug-info-namespace.cpp"]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(5)
; CHECK-NOT: NULL
; CHECK: [[NS2:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "B"
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2:".*foo.cpp"]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(1)
; CHECK-NOT: NULL
; CHECK: [[I:0x[0-9a-f]*]]:{{ *}}DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}= "i"
; CHECK: [[VAR_FWD:0x[0-9a-f]*]]:{{ *}}DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}= "var_fwd"
; CHECK-NOT: NULL
; CHECK: [[FOO:0x[0-9a-f]*]]:{{ *}}DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}= "foo"
; CHECK-NEXT: DW_AT_declaration
; CHECK-NOT: NULL
; CHECK: [[BAR:0x[0-9a-f]*]]:{{ *}}DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}= "bar"
; CHECK: [[FUNC1:.*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "f1"
; CHECK: [[BAZ:0x[0-9a-f]*]]:{{.*}}DW_TAG_typedef
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "baz"
; CHECK: [[VAR_DECL:0x[0-9a-f]*]]:{{.*}}DW_TAG_variable
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "var_decl"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_declaration
; CHECK: [[FUNC_DECL:0x[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "func_decl"
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_declaration
; CHECK: [[FUNC_FWD:0x[0-9a-f]*]]:{{.*}}DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "func_fwd"
; CHECK-NOT: DW_AT_declaration
; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "f1"
; CHECK: NULL

; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; This is a bug, it should be in F2 but it inherits the file from its
; enclosing scope
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F1]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(15)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS2]]})
; CHECK: NULL
; CHECK-NOT: NULL

; CHECK: DW_TAG_imported_module
; Same bug as above, this should be F2, not F1
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F1]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(18)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NOT: NULL

; CHECK: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "func"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(26)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(27)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FOO]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(28)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[BAR]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(29)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FUNC1]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(30)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[I]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(31)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[BAZ]]})
; CHECK-NOT: NULL
; CHECK: [[X:0x[0-9a-f]*]]:{{ *}}DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(32)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NEXT: DW_AT_name{{.*}}"X"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(33)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[X]]})
; CHECK-NEXT: DW_AT_name{{.*}}"Y"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(34)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[VAR_DECL]]})
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(35)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FUNC_DECL]]})
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(36)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[VAR_FWD]]})
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(37)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FUNC_FWD]]})

; CHECK: DW_TAG_lexical_block
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(23)
; CHECK-NEXT: DW_AT_import{{.*}}=>
; CHECK: NULL
; CHECK: NULL
; CHECK: NULL

; IR generated from clang/test/CodeGenCXX/debug-info-namespace.cpp, file paths
; changed to protect the guilty. The C++ source code is:
; // RUN...
; // RUN...
; // RUN...
;
; namespace A {
; #line 1 "foo.cpp"
; namespace B {
; extern int i;
; int f1() { return 0; }
; void f1(int) { }
; struct foo;
; struct bar { };
; typedef bar baz;
; extern int var_decl;
; void func_decl(void);
; extern int var_fwd;
; void func_fwd(void);
; }
; }
; namespace A {
; using namespace B;
; }
;
; using namespace A;
; namespace E = A;
; int B::i = f1();
; int func(bool b) {
;   if (b) {
;     using namespace A::B;
;     return i;
;   }
;   using namespace A;
;   using B::foo;
;   using B::bar;
;   using B::f1;
;   using B::i;
;   using B::baz;
;   namespace X = A;
;   namespace Y = X;
;   using B::var_decl;
;   using B::func_decl;
;   using B::var_fwd;
;   using B::func_fwd;
;   return i + X::B::i + Y::B::i;
; }
;
; namespace A {
; using B::i;
; namespace B {
; int var_fwd = i;
; }
; }
; void B::func_fwd() {}

@_ZN1A1B1iE = global i32 0, align 4
@_ZN1A1B7var_fwdE = global i32 0, align 4
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_debug_info_namespace.cpp, i8* null }]

; Function Attrs: nounwind ssp uwtable
define i32 @_ZN1A1B2f1Ev() #0 {
entry:
  ret i32 0, !dbg !60
}

; Function Attrs: nounwind ssp uwtable
define void @_ZN1A1B2f1Ei(i32) #0 {
entry:
  %.addr = alloca i32, align 4
  store i32 %0, i32* %.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %.addr}, metadata !61, metadata !62), !dbg !63
  ret void, !dbg !64
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  %call = call i32 @_ZN1A1B2f1Ev(), !dbg !65
  store i32 %call, i32* @_ZN1A1B1iE, align 4, !dbg !65
  ret void, !dbg !65
}

; Function Attrs: nounwind ssp uwtable
define i32 @_Z4funcb(i1 zeroext %b) #0 {
entry:
  %retval = alloca i32, align 4
  %b.addr = alloca i8, align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  call void @llvm.dbg.declare(metadata !{i8* %b.addr}, metadata !66, metadata !62), !dbg !67
  %0 = load i8* %b.addr, align 1, !dbg !68
  %tobool = trunc i8 %0 to i1, !dbg !68
  br i1 %tobool, label %if.then, label %if.end, !dbg !68

if.then:                                          ; preds = %entry
  %1 = load i32* @_ZN1A1B1iE, align 4, !dbg !69
  store i32 %1, i32* %retval, !dbg !69
  br label %return, !dbg !69

if.end:                                           ; preds = %entry
  %2 = load i32* @_ZN1A1B1iE, align 4, !dbg !70
  %3 = load i32* @_ZN1A1B1iE, align 4, !dbg !70
  %add = add nsw i32 %2, %3, !dbg !70
  %4 = load i32* @_ZN1A1B1iE, align 4, !dbg !70
  %add1 = add nsw i32 %add, %4, !dbg !70
  store i32 %add1, i32* %retval, !dbg !70
  br label %return, !dbg !70

return:                                           ; preds = %if.end, %if.then
  %5 = load i32* %retval, !dbg !71
  ret i32 %5, !dbg !71
}

define internal void @__cxx_global_var_init1() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  %0 = load i32* @_ZN1A1B1iE, align 4, !dbg !72
  store i32 %0, i32* @_ZN1A1B7var_fwdE, align 4, !dbg !72
  ret void, !dbg !72
}

; Function Attrs: nounwind ssp uwtable
define void @_ZN1A1B8func_fwdEv() #0 {
entry:
  ret void, !dbg !73
}

define internal void @_GLOBAL__sub_I_debug_info_namespace.cpp() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  call void @__cxx_global_var_init(), !dbg !74
  call void @__cxx_global_var_init1(), !dbg !74
  ret void, !dbg !74
}

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!57, !58}
!llvm.ident = !{!59}

!0 = metadata !{metadata !"0x11\004\00clang version 3.6.0 \000\00\000\00\001", metadata !1, metadata !2, metadata !3, metadata !9, metadata !30, metadata !33} ; [ DW_TAG_compile_unit ] [/tmp/debug-info-namespace.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"debug-info-namespace.cpp", metadata !"/tmp"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !8}
!4 = metadata !{metadata !"0x13\00foo\005\000\000\000\004\000", metadata !5, metadata !6, null, null, null, null, metadata !"_ZTSN1A1B3fooE"} ; [ DW_TAG_structure_type ] [foo] [line 5, size 0, align 0, offset 0] [decl] [from ]
!5 = metadata !{metadata !"foo.cpp", metadata !"/tmp"}
!6 = metadata !{metadata !"0x39\00B\001", metadata !5, metadata !7} ; [ DW_TAG_namespace ] [B] [line 1]
!7 = metadata !{metadata !"0x39\00A\005", metadata !1, null} ; [ DW_TAG_namespace ] [A] [line 5]
!8 = metadata !{metadata !"0x13\00bar\006\008\008\000\000\000", metadata !5, metadata !6, null, metadata !2, null, null, metadata !"_ZTSN1A1B3barE"} ; [ DW_TAG_structure_type ] [bar] [line 6, size 8, align 8, offset 0] [def] [from ]
!9 = metadata !{metadata !10, metadata !14, metadata !17, metadata !21, metadata !25, metadata !26, metadata !27}
!10 = metadata !{metadata !"0x2e\00f1\00f1\00_ZN1A1B2f1Ev\003\000\001\000\000\00256\000\003", metadata !5, metadata !6, metadata !11, null, i32 ()* @_ZN1A1B2f1Ev, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 3] [def] [f1]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{metadata !13}
!13 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = metadata !{metadata !"0x2e\00f1\00f1\00_ZN1A1B2f1Ei\004\000\001\000\000\00256\000\004", metadata !5, metadata !6, metadata !15, null, void (i32)* @_ZN1A1B2f1Ei, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 4] [def] [f1]
!15 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !16, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!16 = metadata !{null, metadata !13}
!17 = metadata !{metadata !"0x2e\00__cxx_global_var_init\00__cxx_global_var_init\00\0020\001\001\000\000\00256\000\0020", metadata !5, metadata !18, metadata !19, null, void ()* @__cxx_global_var_init, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 20] [local] [def] [__cxx_global_var_init]
!18 = metadata !{metadata !"0x29", metadata !5}   ; [ DW_TAG_file_type ] [/tmp/foo.cpp]
!19 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !20, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!20 = metadata !{null}
!21 = metadata !{metadata !"0x2e\00func\00func\00_Z4funcb\0021\000\001\000\000\00256\000\0021", metadata !5, metadata !18, metadata !22, null, i32 (i1)* @_Z4funcb, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 21] [def] [func]
!22 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !23, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!23 = metadata !{metadata !13, metadata !24}
!24 = metadata !{metadata !"0x24\00bool\000\008\008\000\000\002", null, null} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!25 = metadata !{metadata !"0x2e\00__cxx_global_var_init1\00__cxx_global_var_init1\00\0044\001\001\000\000\00256\000\0044", metadata !5, metadata !18, metadata !19, null, void ()* @__cxx_global_var_init1, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 44] [local] [def] [__cxx_global_var_init1]
!26 = metadata !{metadata !"0x2e\00func_fwd\00func_fwd\00_ZN1A1B8func_fwdEv\0047\000\001\000\000\00256\000\0047", metadata !5, metadata !6, metadata !19, null, void ()* @_ZN1A1B8func_fwdEv, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 47] [def] [func_fwd]
!27 = metadata !{metadata !"0x2e\00\00\00_GLOBAL__sub_I_debug_info_namespace.cpp\000\001\001\000\000\0064\000\000", metadata !1, metadata !28, metadata !29, null, void ()* @_GLOBAL__sub_I_debug_info_namespace.cpp, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 0] [local] [def]
!28 = metadata !{metadata !"0x29", metadata !1}   ; [ DW_TAG_file_type ] [/tmp/debug-info-namespace.cpp]
!29 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", null, null, null, metadata !2, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!30 = metadata !{metadata !31, metadata !32}
!31 = metadata !{metadata !"0x34\00i\00i\00_ZN1A1B1iE\0020\000\001", metadata !6, metadata !18, metadata !13, i32* @_ZN1A1B1iE, null} ; [ DW_TAG_variable ] [i] [line 20] [def]
!32 = metadata !{metadata !"0x34\00var_fwd\00var_fwd\00_ZN1A1B7var_fwdE\0044\000\001", metadata !6, metadata !18, metadata !13, i32* @_ZN1A1B7var_fwdE, null} ; [ DW_TAG_variable ] [var_fwd] [line 44] [def]
!33 = metadata !{metadata !34, metadata !35, metadata !36, metadata !37, metadata !40, metadata !41, metadata !42, metadata !43, metadata !44, metadata !45, metadata !47, metadata !48, metadata !49, metadata !51, metadata !54, metadata !55, metadata !56}
!34 = metadata !{metadata !"0x3a\0015\00", metadata !7, metadata !6} ; [ DW_TAG_imported_module ]
!35 = metadata !{metadata !"0x3a\0018\00", metadata !0, metadata !7} ; [ DW_TAG_imported_module ]
!36 = metadata !{metadata !"0x8\0019\00E", metadata !0, metadata !7} ; [ DW_TAG_imported_declaration ]
!37 = metadata !{metadata !"0x3a\0023\00", metadata !38, metadata !6} ; [ DW_TAG_imported_module ]
!38 = metadata !{metadata !"0xb\0022\0010\001", metadata !5, metadata !39} ; [ DW_TAG_lexical_block ] [/tmp/foo.cpp]
!39 = metadata !{metadata !"0xb\0022\007\000", metadata !5, metadata !21} ; [ DW_TAG_lexical_block ] [/tmp/foo.cpp]
!40 = metadata !{metadata !"0x3a\0026\00", metadata !21, metadata !7} ; [ DW_TAG_imported_module ]
!41 = metadata !{metadata !"0x8\0027\00", metadata !21, metadata !"_ZTSN1A1B3fooE"} ; [ DW_TAG_imported_declaration ]
!42 = metadata !{metadata !"0x8\0028\00", metadata !21, metadata !"_ZTSN1A1B3barE"} ; [ DW_TAG_imported_declaration ]
!43 = metadata !{metadata !"0x8\0029\00", metadata !21, metadata !14} ; [ DW_TAG_imported_declaration ]
!44 = metadata !{metadata !"0x8\0030\00", metadata !21, metadata !31} ; [ DW_TAG_imported_declaration ]
!45 = metadata !{metadata !"0x8\0031\00", metadata !21, metadata !46} ; [ DW_TAG_imported_declaration ]
!46 = metadata !{metadata !"0x16\00baz\007\000\000\000\000", metadata !5, metadata !6, metadata !"_ZTSN1A1B3barE"} ; [ DW_TAG_typedef ] [baz] [line 7, size 0, align 0, offset 0] [from _ZTSN1A1B3barE]
!47 = metadata !{metadata !"0x8\0032\00X", metadata !21, metadata !7} ; [ DW_TAG_imported_declaration ]
!48 = metadata !{metadata !"0x8\0033\00Y", metadata !21, metadata !47} ; [ DW_TAG_imported_declaration ]
!49 = metadata !{metadata !"0x8\0034\00", metadata !21, metadata !50} ; [ DW_TAG_imported_declaration ]
!50 = metadata !{metadata !"0x34\00var_decl\00var_decl\00_ZN1A1B8var_declE\008\000\000", metadata !6, metadata !18, metadata !13, null, null} ; [ DW_TAG_variable ] [var_decl] [line 8]
!51 = metadata !{metadata !"0x8\0035\00", metadata !21, metadata !52} ; [ DW_TAG_imported_declaration ]
!52 = metadata !{metadata !"0x2e\00func_decl\00func_decl\00_ZN1A1B9func_declEv\009\000\000\000\000\00256\000\000", metadata !5, metadata !6, metadata !19, null, null, null, null, metadata !53} ; [ DW_TAG_subprogram ] [line 9] [scope 0] [func_decl]
!53 = metadata !{metadata !"0x24"}
!54 = metadata !{metadata !"0x8\0036\00", metadata !21, metadata !32} ; [ DW_TAG_imported_declaration ]
!55 = metadata !{metadata !"0x8\0037\00", metadata !21, metadata !26} ; [ DW_TAG_imported_declaration ]
!56 = metadata !{metadata !"0x8\0042\00", metadata !7, metadata !31} ; [ DW_TAG_imported_declaration ]
!57 = metadata !{i32 2, metadata !"Dwarf Version", i32 2}
!58 = metadata !{i32 2, metadata !"Debug Info Version", i32 2}
!59 = metadata !{metadata !"clang version 3.6.0 "}
!60 = metadata !{i32 3, i32 12, metadata !10, null}
!61 = metadata !{metadata !"0x101\00\0016777220\000", metadata !14, metadata !18, metadata !13} ; [ DW_TAG_arg_variable ] [line 4]
!62 = metadata !{metadata !"0x102"}               ; [ DW_TAG_expression ]
!63 = metadata !{i32 4, i32 12, metadata !14, null}
!64 = metadata !{i32 4, i32 16, metadata !14, null}
!65 = metadata !{i32 20, i32 12, metadata !17, null}
!66 = metadata !{metadata !"0x101\00b\0016777237\000", metadata !21, metadata !18, metadata !24} ; [ DW_TAG_arg_variable ] [b] [line 21]
!67 = metadata !{i32 21, i32 15, metadata !21, null}
!68 = metadata !{i32 22, i32 7, metadata !21, null}
!69 = metadata !{i32 24, i32 5, metadata !38, null}
!70 = metadata !{i32 38, i32 3, metadata !21, null}
!71 = metadata !{i32 39, i32 1, metadata !21, null}
!72 = metadata !{i32 44, i32 15, metadata !25, null}
!73 = metadata !{i32 47, i32 21, metadata !26, null}
!74 = metadata !{i32 0, i32 0, metadata !75, null}
!75 = metadata !{metadata !"0xb\000", metadata !5, metadata !27} ; [ DW_TAG_lexical_block ] [/tmp/foo.cpp]
