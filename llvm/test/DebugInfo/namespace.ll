; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; CHECK: debug_info contents
; CHECK: [[NS1:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "A"
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F1:".*debug-info-namespace.cpp"]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(3)
; CHECK-NOT: NULL
; CHECK: [[NS2:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "B"
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2:".*foo.cpp"]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(1)
; CHECK-NOT: NULL
; CHECK: [[I:0x[0-9a-f]*]]:{{ *}}DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}= "i"
; CHECK-NOT: NULL
; CHECK: [[FOO:0x[0-9a-f]*]]:{{ *}}DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}= "foo"
; CHECK-NEXT: DW_AT_declaration
; CHECK-NOT: NULL
; CHECK: [[BAR:0x[0-9a-f]*]]:{{ *}}DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name{{.*}}= "bar"
; CHECK: NULL
; CHECK: [[FUNC1:.*]]: DW_TAG_subprogram
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_MIPS_linkage_name
; CHECK-NOT: DW_TAG
; CHECK: DW_AT_name{{.*}}= "f1"
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
; CHECK-NEXT: DW_AT_decl_line{{.*}}(8)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS2]]})
; CHECK: NULL
; CHECK-NOT: NULL

; CHECK: DW_TAG_imported_module
; Same bug as above, this should be F2, not F1
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F1]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(11)
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
; CHECK-NEXT: DW_AT_decl_line{{.*}}(18)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(19)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FOO]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(20)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[BAR]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(21)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[FUNC1]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(22)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[I]]})
; CHECK-NOT: NULL
; CHECK: [[X:0x[0-9a-f]*]]:{{ *}}DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(24)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NEXT: DW_AT_name{{.*}}"X"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_declaration
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(25)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[X]]})
; CHECK-NEXT: DW_AT_name{{.*}}"Y"
; CHECK-NOT: NULL
; CHECK: DW_TAG_lexical_block
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}([[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(15)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS2]]})
; CHECK: NULL
; CHECK: NULL
; CHECK: NULL

; IR generated from clang/test/CodeGenCXX/debug-info-namespace.cpp, file paths
; changed to protect the guilty. The C++ source code is:
; namespace A {
; #line 1 "foo.cpp"
; namespace B {
; int i;
; void f1() { }
; void f1(int) { }
; struct foo;
; struct bar { };
; }
; using namespace B;
; }
;
; using namespace A;
;
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
;   bar x;
;   namespace X = A;
;   namespace Y = X;
;   return i + X::B::i + Y::B::i;
; }

%"struct.A::B::bar" = type { i8 }

@_ZN1A1B1iE = global i32 0, align 4

; Function Attrs: nounwind uwtable
define void @_ZN1A1B2f1Ev() #0 {
entry:
  ret void, !dbg !41
}

; Function Attrs: nounwind uwtable
define void @_ZN1A1B2f1Ei(i32) #0 {
entry:
  %.addr = alloca i32, align 4
  store i32 %0, i32* %.addr, align 4
  call void @llvm.dbg.declare(metadata !{i32* %.addr}, metadata !42, metadata !{metadata !"0x102"}), !dbg !43
  ret void, !dbg !43
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: nounwind uwtable
define i32 @_Z4funcb(i1 zeroext %b) #0 {
entry:
  %retval = alloca i32, align 4
  %b.addr = alloca i8, align 1
  %x = alloca %"struct.A::B::bar", align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  call void @llvm.dbg.declare(metadata !{i8* %b.addr}, metadata !44, metadata !{metadata !"0x102"}), !dbg !45
  %0 = load i8* %b.addr, align 1, !dbg !46
  %tobool = trunc i8 %0 to i1, !dbg !46
  br i1 %tobool, label %if.then, label %if.end, !dbg !46

if.then:                                          ; preds = %entry
  %1 = load i32* @_ZN1A1B1iE, align 4, !dbg !47
  store i32 %1, i32* %retval, !dbg !47
  br label %return, !dbg !47

if.end:                                           ; preds = %entry
  call void @llvm.dbg.declare(metadata !{%"struct.A::B::bar"* %x}, metadata !48, metadata !{metadata !"0x102"}), !dbg !49
  %2 = load i32* @_ZN1A1B1iE, align 4, !dbg !50
  %3 = load i32* @_ZN1A1B1iE, align 4, !dbg !50
  %add = add nsw i32 %2, %3, !dbg !50
  %4 = load i32* @_ZN1A1B1iE, align 4, !dbg !50
  %add1 = add nsw i32 %add, %4, !dbg !50
  store i32 %add1, i32* %retval, !dbg !50
  br label %return, !dbg !50

return:                                           ; preds = %if.end, %if.then
  %5 = load i32* %retval, !dbg !51
  ret i32 %5, !dbg !51
}

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!52}

!0 = metadata !{metadata !"0x11\004\00clang version 3.4 \000\00\000\00\000", metadata !1, metadata !2, metadata !2, metadata !3, metadata !19, metadata !21} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/llvm/build/clang/debug//usr/local/google/home/blaikie/dev/llvm/src/tools/clang/test/CodeGenCXX/debug-info-namespace.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"/usr/local/google/home/blaikie/dev/llvm/src/tools/clang/test/CodeGenCXX/debug-info-namespace.cpp", metadata !"/usr/local/google/home/blaikie/dev/llvm/build/clang/debug"}
!2 = metadata !{}
!3 = metadata !{metadata !4, metadata !10, metadata !14}
!4 = metadata !{metadata !"0x2e\00f1\00f1\00_ZN1A1B2f1Ev\003\000\001\000\006\00256\000\003", metadata !5, metadata !6, metadata !8, null, void ()* @_ZN1A1B2f1Ev, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 3] [def] [f1]
!5 = metadata !{metadata !"foo.cpp", metadata !"/usr/local/google/home/blaikie/dev/llvm/build/clang/debug"}
!6 = metadata !{metadata !"0x39\00B\001", metadata !5, metadata !7} ; [ DW_TAG_namespace ] [B] [line 1]
!7 = metadata !{metadata !"0x39\00A\003", metadata !1, null} ; [ DW_TAG_namespace ] [A] [line 3]
!8 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !9, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!9 = metadata !{null}
!10 = metadata !{metadata !"0x2e\00f1\00f1\00_ZN1A1B2f1Ei\004\000\001\000\006\00256\000\004", metadata !5, metadata !6, metadata !11, null, void (i32)* @_ZN1A1B2f1Ei, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 4] [def] [f1]
!11 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !12, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!12 = metadata !{null, metadata !13}
!13 = metadata !{metadata !"0x24\00int\000\0032\0032\000\000\005", null, null} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!14 = metadata !{metadata !"0x2e\00func\00func\00_Z4funcb\0013\000\001\000\006\00256\000\0013", metadata !5, metadata !15, metadata !16, null, i32 (i1)* @_Z4funcb, null, null, metadata !2} ; [ DW_TAG_subprogram ] [line 13] [def] [func]
!15 = metadata !{metadata !"0x29", metadata !5}         ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/llvm/build/clang/debug/foo.cpp]
!16 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !17, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!17 = metadata !{metadata !13, metadata !18}
!18 = metadata !{metadata !"0x24\00bool\000\008\008\000\000\002", null, null} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!19 = metadata !{metadata !20}
!20 = metadata !{metadata !"0x34\00i\00i\00_ZN1A1B1iE\002\000\001", metadata !6, metadata !15, metadata !13, i32* @_ZN1A1B1iE, null} ; [ DW_TAG_variable ] [i] [line 2] [def]
!21 = metadata !{metadata !22, metadata !23, metadata !24, metadata !26, metadata !27, metadata !29, metadata !37, metadata !38, metadata !39, metadata !40}
!22 = metadata !{metadata !"0x3a\008\00", metadata !7, metadata !6} ; [ DW_TAG_imported_module ]
!23 = metadata !{metadata !"0x3a\0011\00", metadata !0, metadata !7} ; [ DW_TAG_imported_module ]
!24 = metadata !{metadata !"0x3a\0015\00", metadata !25, metadata !6} ; [ DW_TAG_imported_module ]
!25 = metadata !{metadata !"0xb\0014\000\000", metadata !5, metadata !14} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/blaikie/dev/llvm/build/clang/debug/foo.cpp]
!26 = metadata !{metadata !"0x3a\0018\00", metadata !14, metadata !7} ; [ DW_TAG_imported_module ]
!27 = metadata !{metadata !"0x8\0019\00", metadata !14, metadata !28} ; [ DW_TAG_imported_declaration ]
!28 = metadata !{metadata !"0x13\00foo\005\000\000\000\004\000", metadata !5, metadata !6, null, null, null, null, null} ; [ DW_TAG_structure_type ] [foo] [line 5, size 0, align 0, offset 0] [decl] [from ]
!29 = metadata !{metadata !"0x8\0020\00", metadata !14, metadata !30} ; [ DW_TAG_imported_declaration ]
!30 = metadata !{metadata !"0x13\00bar\006\008\008\000\000\000", metadata !5, metadata !6, null, metadata !31, null, null, null} ; [ DW_TAG_structure_type ] [bar] [line 6, size 8, align 8, offset 0] [def] [from ]
!31 = metadata !{metadata !32}
!32 = metadata !{metadata !"0x2e\00bar\00bar\00\006\000\000\000\006\00320\000\006", metadata !5, metadata !30, metadata !33, null, null, null, i32 0, metadata !36} ; [ DW_TAG_subprogram ] [line 6] [bar]
!33 = metadata !{metadata !"0x15\00\000\000\000\000\000\000", i32 0, null, null, metadata !34, null, null, null} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!34 = metadata !{null, metadata !35}
!35 = metadata !{metadata !"0xf\00\000\0064\0064\000\001088", i32 0, null, metadata !30} ; [ DW_TAG_pointer_type ] [line 0, size 64, align 64, offset 0] [artificial] [from bar]
!36 = metadata !{i32 786468}
!37 = metadata !{metadata !"0x8\0021\00", metadata !14, metadata !10} ; [ DW_TAG_imported_declaration ]
!38 = metadata !{metadata !"0x8\0022\00", metadata !14, metadata !20} ; [ DW_TAG_imported_declaration ]
!39 = metadata !{metadata !"0x8\0024\00X", metadata !14, metadata !7} ; [ DW_TAG_imported_declaration ]
!40 = metadata !{metadata !"0x8\0025\00Y", metadata !14, metadata !39} ; [ DW_TAG_imported_declaration ]
!41 = metadata !{i32 3, i32 0, metadata !4, null}
!42 = metadata !{metadata !"0x101\00\0016777220\000", metadata !10, metadata !15, metadata !13} ; [ DW_TAG_arg_variable ] [line 4]
!43 = metadata !{i32 4, i32 0, metadata !10, null}
!44 = metadata !{metadata !"0x101\00b\0016777229\000", metadata !14, metadata !15, metadata !18} ; [ DW_TAG_arg_variable ] [b] [line 13]
!45 = metadata !{i32 13, i32 0, metadata !14, null}
!46 = metadata !{i32 14, i32 0, metadata !14, null}
!47 = metadata !{i32 16, i32 0, metadata !25, null}
!48 = metadata !{metadata !"0x100\00x\0023\000", metadata !14, metadata !15, metadata !30} ; [ DW_TAG_auto_variable ] [x] [line 23]
!49 = metadata !{i32 23, i32 0, metadata !14, null}
!50 = metadata !{i32 26, i32 0, metadata !14, null}
!51 = metadata !{i32 27, i32 0, metadata !14, null}
!52 = metadata !{i32 1, metadata !"Debug Info Version", i32 2}
