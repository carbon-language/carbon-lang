; RUN: llc -filetype=obj < %s > %t.obj
; RUN: obj2yaml %t.obj | FileCheck %s --check-prefix=YAML
; RUN: llc -filetype=asm < %s | FileCheck %s --check-prefix=ASM

; C++ source to regenerate:
; $ cat t.cpp
; struct Foo {
;   Foo(int x, int y) : X(x), Y(y) {}
;   int method() { return X + Y; }
;   int X;
;   int Y;
; };
; int main(int argc, char **argv) {
;   Foo F {argc, argc};
;   return F.method();
; };
; $ clang-cc1 -triple i686-pc-windows-msvc19.11.25547 -emit-llvm -gcodeview \
;   -debug-info-kind=limited -std=c++14 foo.cpp
;


; ModuleID = 'foo.cpp'
source_filename = "foo.cpp"
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-windows-msvc19.11.25547"

%struct.Foo = type { i32, i32 }

$"\01??0Foo@@QAE@HH@Z" = comdat any

$"\01?method@Foo@@QAEHXZ" = comdat any

; Function Attrs: noinline norecurse nounwind optnone
define i32 @main(i32 %argc, i8** %argv) #0 !dbg !8 {
entry:
  %retval = alloca i32, align 4
  %argv.addr = alloca i8**, align 4
  %argc.addr = alloca i32, align 4
  %F = alloca %struct.Foo, align 4
  store i32 0, i32* %retval, align 4
  store i8** %argv, i8*** %argv.addr, align 4
  call void @llvm.dbg.declare(metadata i8*** %argv.addr, metadata !16, metadata !DIExpression()), !dbg !17
  store i32 %argc, i32* %argc.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %argc.addr, metadata !18, metadata !DIExpression()), !dbg !17
  call void @llvm.dbg.declare(metadata %struct.Foo* %F, metadata !19, metadata !DIExpression()), !dbg !31
  %0 = load i32, i32* %argc.addr, align 4, !dbg !31
  %1 = load i32, i32* %argc.addr, align 4, !dbg !31
  %call = call x86_thiscallcc %struct.Foo* @"\01??0Foo@@QAE@HH@Z"(%struct.Foo* %F, i32 %0, i32 %1), !dbg !31
  %call1 = call x86_thiscallcc i32 @"\01?method@Foo@@QAEHXZ"(%struct.Foo* %F), !dbg !32
  ret i32 %call1, !dbg !32
}

; Function Attrs: nounwind readnone speculatable
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

; Function Attrs: noinline nounwind optnone
define linkonce_odr x86_thiscallcc %struct.Foo* @"\01??0Foo@@QAE@HH@Z"(%struct.Foo* returned %this, i32 %x, i32 %y) unnamed_addr #2 comdat align 2 !dbg !33 {
entry:
  %y.addr = alloca i32, align 4
  %x.addr = alloca i32, align 4
  %this.addr = alloca %struct.Foo*, align 4
  store i32 %y, i32* %y.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %y.addr, metadata !34, metadata !DIExpression()), !dbg !35
  store i32 %x, i32* %x.addr, align 4
  call void @llvm.dbg.declare(metadata i32* %x.addr, metadata !36, metadata !DIExpression()), !dbg !35
  store %struct.Foo* %this, %struct.Foo** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %struct.Foo** %this.addr, metadata !37, metadata !DIExpression()), !dbg !39
  %this1 = load %struct.Foo*, %struct.Foo** %this.addr, align 4
  %X = getelementptr inbounds %struct.Foo, %struct.Foo* %this1, i32 0, i32 0, !dbg !35
  %0 = load i32, i32* %x.addr, align 4, !dbg !35
  store i32 %0, i32* %X, align 4, !dbg !35
  %Y = getelementptr inbounds %struct.Foo, %struct.Foo* %this1, i32 0, i32 1, !dbg !35
  %1 = load i32, i32* %y.addr, align 4, !dbg !35
  store i32 %1, i32* %Y, align 4, !dbg !35
  ret %struct.Foo* %this1, !dbg !35
}

; Function Attrs: noinline nounwind optnone
define linkonce_odr x86_thiscallcc i32 @"\01?method@Foo@@QAEHXZ"(%struct.Foo* %this) #2 comdat align 2 !dbg !40 {
entry:
  %this.addr = alloca %struct.Foo*, align 4
  store %struct.Foo* %this, %struct.Foo** %this.addr, align 4
  call void @llvm.dbg.declare(metadata %struct.Foo** %this.addr, metadata !41, metadata !DIExpression()), !dbg !42
  %this1 = load %struct.Foo*, %struct.Foo** %this.addr, align 4
  %X = getelementptr inbounds %struct.Foo, %struct.Foo* %this1, i32 0, i32 0, !dbg !43
  %0 = load i32, i32* %X, align 4, !dbg !43
  %Y = getelementptr inbounds %struct.Foo, %struct.Foo* %this1, i32 0, i32 1, !dbg !43
  %1 = load i32, i32* %Y, align 4, !dbg !43
  %add = add nsw i32 %0, %1, !dbg !43
  ret i32 %add, !dbg !43
}

attributes #0 = { noinline norecurse nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone speculatable }
attributes #2 = { noinline nounwind optnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="none" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-features"="+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6, !100}
!llvm.ident = !{!7}


!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 6.0.0 ", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "<stdin>", directory: "D:\5Csrc\5Cllvmbuild\5Cclang\5CDebug\5Cx86", checksumkind: CSK_MD5, checksum: "6279449503d9075c38e615e8387667c3")
!2 = !{}
!3 = !{i32 1, !"NumRegisterParameters", i32 0}
!4 = !{i32 2, !"CodeView", i32 1}
!100 = !{i32 2, !"CodeViewGHash", i32 1}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 2}
!7 = !{!"clang version 6.0.0 "}
!8 = distinct !DISubprogram(name: "main", scope: !9, file: !9, line: 8, type: !10, isLocal: false, isDefinition: true, scopeLine: 8, flags: DIFlagPrototyped, isOptimized: false, unit: !0, retainedNodes: !2)
!9 = !DIFile(filename: "foo.cpp", directory: "D:\5Csrc\5Cllvmbuild\5Cclang\5CDebug\5Cx86", checksumkind: CSK_MD5, checksum: "6279449503d9075c38e615e8387667c3")
!10 = !DISubroutineType(types: !11)
!11 = !{!12, !12, !13}
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!13 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !14, size: 32)
!14 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !15, size: 32)
!15 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!16 = !DILocalVariable(name: "argv", arg: 2, scope: !8, file: !9, line: 8, type: !13)
!17 = !DILocation(line: 8, scope: !8)
!18 = !DILocalVariable(name: "argc", arg: 1, scope: !8, file: !9, line: 8, type: !12)
!19 = !DILocalVariable(name: "F", scope: !8, file: !9, line: 9, type: !20)
!20 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Foo", file: !9, line: 1, size: 64, elements: !21, identifier: ".?AUFoo@@")
!21 = !{!22, !23, !24, !28}
!22 = !DIDerivedType(tag: DW_TAG_member, name: "X", scope: !20, file: !9, line: 4, baseType: !12, size: 32)
!23 = !DIDerivedType(tag: DW_TAG_member, name: "Y", scope: !20, file: !9, line: 5, baseType: !12, size: 32, offset: 32)
!24 = !DISubprogram(name: "Foo", scope: !20, file: !9, line: 2, type: !25, isLocal: false, isDefinition: false, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false)
!25 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !26)
!26 = !{null, !27, !12, !12}
!27 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 32, flags: DIFlagArtificial | DIFlagObjectPointer)
!28 = !DISubprogram(name: "method", linkageName: "\01?method@Foo@@QAEHXZ", scope: !20, file: !9, line: 3, type: !29, isLocal: false, isDefinition: false, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false)
!29 = !DISubroutineType(cc: DW_CC_BORLAND_thiscall, types: !30)
!30 = !{!12, !27}
!31 = !DILocation(line: 9, scope: !8)
!32 = !DILocation(line: 10, scope: !8)
!33 = distinct !DISubprogram(name: "Foo", linkageName: "\01??0Foo@@QAE@HH@Z", scope: !20, file: !9, line: 2, type: !25, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !24, retainedNodes: !2)
!34 = !DILocalVariable(name: "y", arg: 3, scope: !33, file: !9, line: 2, type: !12)
!35 = !DILocation(line: 2, scope: !33)
!36 = !DILocalVariable(name: "x", arg: 2, scope: !33, file: !9, line: 2, type: !12)
!37 = !DILocalVariable(name: "this", arg: 1, scope: !33, type: !38, flags: DIFlagArtificial | DIFlagObjectPointer)
!38 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !20, size: 32)
!39 = !DILocation(line: 0, scope: !33)
!40 = distinct !DISubprogram(name: "method", linkageName: "\01?method@Foo@@QAEHXZ", scope: !20, file: !9, line: 3, type: !29, isLocal: false, isDefinition: true, scopeLine: 3, flags: DIFlagPrototyped, isOptimized: false, unit: !0, declaration: !28, retainedNodes: !2)
!41 = !DILocalVariable(name: "this", arg: 1, scope: !40, type: !38, flags: DIFlagArtificial | DIFlagObjectPointer)
!42 = !DILocation(line: 0, scope: !40)
!43 = !DILocation(line: 3, scope: !40)


; YAML: --- !COFF
; YAML: header:
; YAML:   Machine:         IMAGE_FILE_MACHINE_I386
; YAML:   Characteristics: [  ]
; YAML: sections:
; YAML:   - Name:            '.debug$T'
; YAML:     Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_DISCARDABLE, IMAGE_SCN_MEM_READ ]
; YAML:     Alignment:       4
; YAML:     Types:
; YAML:       - Kind:            LF_POINTER
; YAML:         Pointer:
; YAML:           ReferentType:    1136
; YAML:           Attrs:           32778
; YAML:       - Kind:            LF_ARGLIST
; YAML:         ArgList:
; YAML:           ArgIndices:      [ 116, 4096 ]
; YAML:       - Kind:            LF_PROCEDURE
; YAML:         Procedure:
; YAML:           ReturnType:      116
; YAML:           CallConv:        NearC
; YAML:           Options:         [ None ]
; YAML:           ParameterCount:  2
; YAML:           ArgumentList:    4097
; YAML:       - Kind:            LF_FUNC_ID
; YAML:         FuncId:
; YAML:           ParentScope:     0
; YAML:           FunctionType:    4098
; YAML:           Name:            main
; YAML:       - Kind:            LF_STRUCTURE
; YAML:         Class:
; YAML:           MemberCount:     0
; YAML:           Options:         [ None, ForwardReference, HasUniqueName ]
; YAML:           FieldList:       0
; YAML:           Name:            Foo
; YAML:           UniqueName:      '.?AUFoo@@'
; YAML:           DerivationList:  0
; YAML:           VTableShape:     0
; YAML:           Size:            0
; YAML:       - Kind:            LF_POINTER
; YAML:         Pointer:
; YAML:           ReferentType:    4100
; YAML:           Attrs:           33802
; YAML:       - Kind:            LF_ARGLIST
; YAML:         ArgList:
; YAML:           ArgIndices:      [ 116, 116 ]
; YAML:       - Kind:            LF_MFUNCTION
; YAML:         MemberFunction:
; YAML:           ReturnType:      3
; YAML:           ClassType:       4100
; YAML:           ThisType:        4101
; YAML:           CallConv:        ThisCall
; YAML:           Options:         [ None ]
; YAML:           ParameterCount:  2
; YAML:           ArgumentList:    4102
; YAML:           ThisPointerAdjustment: 0
; YAML:       - Kind:            LF_ARGLIST
; YAML:         ArgList:
; YAML:           ArgIndices:      [  ]
; YAML:       - Kind:            LF_MFUNCTION
; YAML:         MemberFunction:
; YAML:           ReturnType:      116
; YAML:           ClassType:       4100
; YAML:           ThisType:        4101
; YAML:           CallConv:        ThisCall
; YAML:           Options:         [ None ]
; YAML:           ParameterCount:  0
; YAML:           ArgumentList:    4104
; YAML:           ThisPointerAdjustment: 0
; YAML:       - Kind:            LF_FIELDLIST
; YAML:         FieldList:
; YAML:           - Kind:            LF_MEMBER
; YAML:             DataMember:
; YAML:               Attrs:           3
; YAML:               Type:            116
; YAML:               FieldOffset:     0
; YAML:               Name:            X
; YAML:           - Kind:            LF_MEMBER
; YAML:             DataMember:
; YAML:               Attrs:           3
; YAML:               Type:            116
; YAML:               FieldOffset:     4
; YAML:               Name:            Y
; YAML:           - Kind:            LF_ONEMETHOD
; YAML:             OneMethod:
; YAML:               Type:            4103
; YAML:               Attrs:           3
; YAML:               VFTableOffset:   -1
; YAML:               Name:            Foo
; YAML:           - Kind:            LF_ONEMETHOD
; YAML:             OneMethod:
; YAML:               Type:            4105
; YAML:               Attrs:           3
; YAML:               VFTableOffset:   -1
; YAML:               Name:            method
; YAML:       - Kind:            LF_STRUCTURE
; YAML:         Class:
; YAML:           MemberCount:     4
; YAML:           Options:         [ None, HasUniqueName ]
; YAML:           FieldList:       4106
; YAML:           Name:            Foo
; YAML:           UniqueName:      '.?AUFoo@@'
; YAML:           DerivationList:  0
; YAML:           VTableShape:     0
; YAML:           Size:            8
; YAML:       - Kind:            LF_STRING_ID
; YAML:         StringId:
; YAML:           Id:              0
; YAML:           String:          'D:\src\llvmbuild\clang\Debug\x86\foo.cpp'
; YAML:       - Kind:            LF_UDT_SRC_LINE
; YAML:         UdtSourceLine:
; YAML:           UDT:             4107
; YAML:           SourceFile:      4108
; YAML:           LineNumber:      1
; YAML:       - Kind:            LF_MFUNC_ID
; YAML:         MemberFuncId:
; YAML:           ClassType:       4100
; YAML:           FunctionType:    4103
; YAML:           Name:            Foo
; YAML:       - Kind:            LF_MFUNC_ID
; YAML:         MemberFuncId:
; YAML:           ClassType:       4100
; YAML:           FunctionType:    4105
; YAML:           Name:            method
; YAML:   - Name:            '.debug$H'
; YAML:     Characteristics: [ IMAGE_SCN_CNT_INITIALIZED_DATA, IMAGE_SCN_MEM_DISCARDABLE, IMAGE_SCN_MEM_READ ]
; YAML:     Alignment:       4
; YAML:     GlobalHashes:
; YAML:       Version:         0
; YAML:       HashAlgorithm:   1
; YAML:       HashValues:
; YAML:         - 800309EE1ED8BB5B
; YAML:         - 5397319F1CC14E2C
; YAML:         - DF04AA3125BBC50E
; YAML:         - 95CEBA304A2C4493
; YAML:         - C324F82D24D22283
; YAML:         - BB039258F2425BCF
; YAML:         - DDE23757322DB7C3
; YAML:         - 44D3ED149C981B2A
; YAML:         - D341E2F9BE57A1C7
; YAML:         - DD327744BE6783A4
; YAML:         - 5B17837C70325869
; YAML:         - 375C55CDF44B4147
; YAML:         - 0634944401BCC520
; YAML:         - 6CC0AFB95FA2BFF2
; YAML:         - D016F92E5C009314
; YAML:         - 74698BE366891D3D
; YAML:         - 4470750F2E319329
; YAML:         - 0FB556FD1FAB66D7
; YAML:         - 5970EFB4874D0F3F
; YAML:         - EDB1D74C120CF44A
; ...


; ASM:      .section        .debug$H,"dr"
; ASM-NEXT: .p2align        2
; ASM-NEXT: .long   20171205                # Magic
; ASM-NEXT: .short  0                       # Section Version
; ASM-NEXT: .short  1                       # Hash Algorithm
; ASM-NEXT: .byte   0x80, 0x03, 0x09, 0xee  # 0x1000 [800309EE1ED8BB5B]
; ASM-NEXT: .byte   0x1e, 0xd8, 0xbb, 0x5b
; ASM-NEXT: .byte   0x53, 0x97, 0x31, 0x9f  # 0x1001 [5397319F1CC14E2C]
; ASM-NEXT: .byte   0x1c, 0xc1, 0x4e, 0x2c
; ASM-NEXT: .byte   0xdf, 0x04, 0xaa, 0x31  # 0x1002 [DF04AA3125BBC50E]
