; REQUIRES: object-emission

; RUN: llc -O0 -filetype=obj < %s > %t
; RUN: llvm-dwarfdump %t | FileCheck %s
; CHECK: debug_info contents
; CHECK: [[NS1:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "A"
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F1:[0-9]]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x03)
; CHECK-NOT: NULL
; CHECK: [[NS2:0x[0-9a-f]*]]:{{ *}}DW_TAG_namespace
; CHECK-NEXT: DW_AT_name{{.*}} = "B"
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2:[0-9]]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x01)
; CHECK-NOT: NULL
; CHECK: DW_TAG_variable
; CHECK-NEXT: DW_AT_name{{.*}}= "i"
; CHECK: NULL
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; This is a bug, it should be in F2 but it inherits the file from its
; enclosing scope
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F1]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x04)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS2]]})

; CHECK: DW_TAG_subprogram
; CHECK-NEXT: DW_AT_MIPS_linkage_name
; CHECK-NEXT: DW_AT_name{{.*}}= "func"
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x0e)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS1]]})
; CHECK-NOT: NULL
; CHECK: DW_TAG_lexical_block
; CHECK-NOT: NULL
; CHECK: DW_TAG_imported_module
; CHECK-NEXT: DW_AT_decl_file{{.*}}(0x0[[F2]])
; CHECK-NEXT: DW_AT_decl_line{{.*}}(0x0b)
; CHECK-NEXT: DW_AT_import{{.*}}=> {[[NS2]]})

; CHECK: file_names[  [[F1]]]{{.*}}debug-info-namespace.cpp
; CHECK: file_names[  [[F2]]]{{.*}}foo.cpp

; IR generated from clang/test/CodeGenCXX/debug-info-namespace.cpp, file paths
; changed to protect the guilty. The C++ source code is simply:
; namespace A {
; #line 1 "foo.cpp"
; namespace B {
; int i;
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
;   return B::i;
; }

@_ZN1A1B1iE = global i32 0, align 4

; Function Attrs: nounwind uwtable
define i32 @_Z4funcb(i1 zeroext %b) #0 {
entry:
  %retval = alloca i32, align 4
  %b.addr = alloca i8, align 1
  %frombool = zext i1 %b to i8
  store i8 %frombool, i8* %b.addr, align 1
  call void @llvm.dbg.declare(metadata !{i8* %b.addr}, metadata !21), !dbg !22
  %0 = load i8* %b.addr, align 1, !dbg !23
  %tobool = trunc i8 %0 to i1, !dbg !23
  br i1 %tobool, label %if.then, label %if.end, !dbg !23

if.then:                                          ; preds = %entry
  %1 = load i32* @_ZN1A1B1iE, align 4, !dbg !24
  store i32 %1, i32* %retval, !dbg !24
  br label %return, !dbg !24

if.end:                                           ; preds = %entry
  %2 = load i32* @_ZN1A1B1iE, align 4, !dbg !25
  store i32 %2, i32* %retval, !dbg !25
  br label %return, !dbg !25

return:                                           ; preds = %if.end, %if.then
  %3 = load i32* %retval, !dbg !26
  ret i32 %3, !dbg !26
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf"="true" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}

!0 = metadata !{i32 786449, metadata !1, i32 4, metadata !"clang version 3.3 ", i1 false, metadata !"", i32 0, metadata !2, metadata !2, metadata !3, metadata !11, metadata !15, metadata !""} ; [ DW_TAG_compile_unit ] [/usr/local/google/home/blaikie/dev/llvm/src/tools/clang//usr/local/google/home/blaikie/dev/llvm/src/tools/clang/test/CodeGenCXX/debug-info-namespace.cpp] [DW_LANG_C_plus_plus]
!1 = metadata !{metadata !"/usr/local/google/home/blaikie/dev/llvm/src/tools/clang/test/CodeGenCXX/debug-info-namespace.cpp", metadata !"/usr/local/google/home/blaikie/dev/llvm/src/tools/clang"}
!2 = metadata !{i32 0}
!3 = metadata !{metadata !4}
!4 = metadata !{i32 786478, metadata !5, metadata !6, metadata !"func", metadata !"func", metadata !"_Z4funcb", i32 9, metadata !7, i1 false, i1 true, i32 0, i32 0, null, i32 256, i1 false, i32 (i1)* @_Z4funcb, null, null, metadata !2, i32 9} ; [ DW_TAG_subprogram ] [line 9] [def] [func]
!5 = metadata !{metadata !"foo.cpp", metadata !"/usr/local/google/home/blaikie/dev/llvm/src/tools/clang"}
!6 = metadata !{i32 786473, metadata !5}          ; [ DW_TAG_file_type ] [/usr/local/google/home/blaikie/dev/llvm/build/clang/debug/foo.cpp]
!7 = metadata !{i32 786453, i32 0, i32 0, metadata !"", i32 0, i64 0, i64 0, i64 0, i32 0, null, metadata !8, i32 0, i32 0} ; [ DW_TAG_subroutine_type ] [line 0, size 0, align 0, offset 0] [from ]
!8 = metadata !{metadata !9, metadata !10}
!9 = metadata !{i32 786468, null, null, metadata !"int", i32 0, i64 32, i64 32, i64 0, i32 0, i32 5} ; [ DW_TAG_base_type ] [int] [line 0, size 32, align 32, offset 0, enc DW_ATE_signed]
!10 = metadata !{i32 786468, null, null, metadata !"bool", i32 0, i64 8, i64 8, i64 0, i32 0, i32 2} ; [ DW_TAG_base_type ] [bool] [line 0, size 8, align 8, offset 0, enc DW_ATE_boolean]
!11 = metadata !{metadata !12}
!12 = metadata !{i32 786484, i32 0, metadata !13, metadata !"i", metadata !"i", metadata !"_ZN1A1B1iE", metadata !6, i32 2, metadata !9, i32 0, i32 1, i32* @_ZN1A1B1iE, null} ; [ DW_TAG_variable ] [i] [line 2] [def]
!13 = metadata !{i32 786489, metadata !5, metadata !14, metadata !"B", i32 1} ; [ DW_TAG_namespace ] [B] [line 1]
!14 = metadata !{i32 786489, metadata !1, null, metadata !"A", i32 3} ; [ DW_TAG_namespace ] [A] [line 3]
!15 = metadata !{metadata !16, metadata !17, metadata !18, metadata !20}
!16 = metadata !{i32 786490, metadata !14, metadata !13, i32 4} ; [ DW_TAG_imported_module ]
!17 = metadata !{i32 786490, metadata !0, metadata !14, i32 7} ; [ DW_TAG_imported_module ]
!18 = metadata !{i32 786490, metadata !19, metadata !13, i32 11} ; [ DW_TAG_imported_module ]
!19 = metadata !{i32 786443, metadata !5, metadata !4, i32 10, i32 0, i32 0} ; [ DW_TAG_lexical_block ] [/usr/local/google/home/blaikie/dev/llvm/build/clang/debug/foo.cpp]
!20 = metadata !{i32 786490, metadata !4, metadata !14, i32 14} ; [ DW_TAG_imported_module ]
!21 = metadata !{i32 786689, metadata !4, metadata !"b", metadata !6, i32 16777225, metadata !10, i32 0, i32 0} ; [ DW_TAG_arg_variable ] [b] [line 9]
!22 = metadata !{i32 9, i32 0, metadata !4, null}
!23 = metadata !{i32 10, i32 0, metadata !4, null}
!24 = metadata !{i32 12, i32 0, metadata !19, null}
!25 = metadata !{i32 15, i32 0, metadata !4, null}
!26 = metadata !{i32 16, i32 0, metadata !4, null}
