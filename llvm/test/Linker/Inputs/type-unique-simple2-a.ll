; Make sure the backend generates a single DIE and uses ref_addr.
; CHECK: 0x[[BASE:.*]]: DW_TAG_structure_type
; CHECK-NEXT: DW_AT_name {{.*}} = "Base"
; CHECK-NOT: DW_TAG_structure_type
; CHECK: 0x[[INT:.*]]: DW_TAG_base_type
; CHECK-NEXT: DW_AT_name {{.*}} = "int"
; CHECK-NOT: DW_TAG_base_type

; CHECK: DW_TAG_compile_unit
; CHECK: DW_TAG_formal_parameter
; CHECK: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[INT]])
; CHECK: DW_TAG_variable
; CHECK: DW_AT_type [DW_FORM_ref_addr] {{.*}}[[BASE]])

; Make sure llvm-link only generates a single copy of the struct.
; LINK: DW_TAG_structure_type
; LINK-NOT: DW_TAG_structure_type

; Content of header files:
; struct Base {
;   int a;
;   Base *b;
; };
; Content of foo.cpp:
; 
; #include "a.hpp"
; void f(int a) {
;   Base t;
; }
; Content of bar.cpp:
; 
; #include "a.hpp"
; void f(int);
; void g(int a) {
;   Base t;
; }
; int main() {
;   f(0);
;   g(1);
;   return 0;
; }
; ModuleID = 'foo.cpp'

%struct.Base = type { i32, %struct.Base* }

; Function Attrs: nounwind ssp uwtable
define void @_Z1fi(i32 %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %t = alloca %struct.Base, align 8
  store i32 %a, i32* %a.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %a.addr, metadata !17, metadata !MDExpression()), !dbg !18
  call void @llvm.dbg.declare(metadata %struct.Base* %t, metadata !19, metadata !MDExpression()), !dbg !20
  ret void, !dbg !21
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

attributes #0 = { nounwind ssp uwtable "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!16, !22}

!0 = !MDCompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.4 (http://llvm.org/git/clang.git 8a3f9e46cb988d2c664395b21910091e3730ae82) (http://llvm.org/git/llvm.git 4699e9549358bc77824a59114548eecc3f7c523c)", isOptimized: false, emissionKind: 0, file: !1, enums: !2, retainedTypes: !3, subprograms: !11, globals: !2, imports: !2)
!1 = !MDFile(filename: "foo.cpp", directory: ".")
!2 = !{}
!3 = !{!4}
!4 = !MDCompositeType(tag: DW_TAG_structure_type, name: "Base", line: 1, size: 128, align: 64, file: !5, elements: !6, identifier: "_ZTS4Base")
!5 = !MDFile(filename: "./a.hpp", directory: ".")
!6 = !{!7, !9}
!7 = !MDDerivedType(tag: DW_TAG_member, name: "a", line: 2, size: 32, align: 32, file: !5, scope: !"_ZTS4Base", baseType: !8)
!8 = !MDBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !MDDerivedType(tag: DW_TAG_member, name: "b", line: 3, size: 64, align: 64, offset: 64, file: !5, scope: !"_ZTS4Base", baseType: !10)
!10 = !MDDerivedType(tag: DW_TAG_pointer_type, size: 64, align: 64, baseType: !"_ZTS4Base")
!11 = !{!12}
!12 = !MDSubprogram(name: "f", linkageName: "_Z1fi", line: 3, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, scopeLine: 3, file: !1, scope: !13, type: !14, function: void (i32)* @_Z1fi, variables: !2)
!13 = !MDFile(filename: "foo.cpp", directory: ".")
!14 = !MDSubroutineType(types: !15)
!15 = !{null, !8}
!16 = !{i32 2, !"Dwarf Version", i32 2}
!17 = !MDLocalVariable(tag: DW_TAG_arg_variable, name: "a", line: 3, arg: 1, scope: !12, file: !13, type: !8)
!18 = !MDLocation(line: 3, scope: !12)
!19 = !MDLocalVariable(tag: DW_TAG_auto_variable, name: "t", line: 4, scope: !12, file: !13, type: !4)
!20 = !MDLocation(line: 4, scope: !12)
!21 = !MDLocation(line: 5, scope: !12)
!22 = !{i32 1, !"Debug Info Version", i32 3}
