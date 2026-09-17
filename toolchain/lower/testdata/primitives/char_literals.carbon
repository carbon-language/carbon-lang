// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/char_literals.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/char_literals.carbon

class CharLike {
  impl as Core.ImplicitAs(char) {
    fn Convert(unused self) -> char { return 'x'; }
  }
}

fn Discard[T: type](unused x: T) {}

fn Convert() {
  Discard('x' as u8);
  Discard('x' as u16);
  Discard('x' as u32);
  Discard('x' as i8);
  Discard('x' as i16);
  Discard('x' as i32);
}

fn ConvertConstant() {
  Discard('x' as u8);
  Discard('x' as u16);
  Discard('x' as u32);
  Discard('x' as i16);
  Discard('x' as i32);
}

fn CompareChar(b: char) {
  // TODO: Should we support CharLiteral vs CharLiteral comparisons?

  Discard('x' == b);
  Discard('x' != b);
  Discard('x' < b);
  Discard('x' <= b);
  Discard('x' > b);
  Discard('x' >= b);

  Discard(b == 'x');
  Discard(b != 'x');
  Discard(b < 'x');
  Discard(b <= 'x');
  Discard(b > 'x');
  Discard(b >= 'x');
}

fn SubCharChar(c: char) {
  Discard('x' - 'y');
  Discard(c - 'x');
  Discard('x' - c);
}

fn AddSubIntLiteral() {
  // TODO: Should we support arithmetic with non-constant integers?
  // We currently require a cast to `char` first.

  Discard('x' + 1);
  Discard(1 + 'x');
  Discard('x' - 1);
}

fn AddSubInt8() {
  // TODO: Should we support arithmetic with non-constant integers?
  // We currently require a cast to `char` first.

  Discard('x' + (1 as i8));
  Discard((1 as i8) + 'x');
  Discard('x' - (1 as i8));
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'char_literals.carbon'
// CHECK:STDOUT: source_filename = "char_literals.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %self) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i8 120, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 120), !dbg !27
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 120), !dbg !28
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 120), !dbg !31
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 120), !dbg !34
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 120), !dbg !37
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 120), !dbg !40
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertConstant.Main() #0 !dbg !44 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 120), !dbg !45
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 120), !dbg !46
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 120), !dbg !47
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 120), !dbg !48
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 120), !dbg !49
// CHECK:STDOUT:   ret void, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompareChar.Main(i8 %b) #0 !dbg !51 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call = icmp eq i8 120, %b, !dbg !55
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.EqWith.impl.Equal.call), !dbg !56
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call = icmp ne i8 120, %b, !dbg !60
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.EqWith.impl.NotEqual.call), !dbg !61
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call = icmp ult i8 120, %b, !dbg !63
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.Less.call), !dbg !64
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 120, %b, !dbg !66
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !67
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call = icmp ugt i8 120, %b, !dbg !69
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.Greater.call), !dbg !70
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 120, %b, !dbg !72
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !73
// CHECK:STDOUT:   %char.as.EqWith.impl.Equal.call = icmp eq i8 %b, 120, !dbg !74
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.Equal.call), !dbg !75
// CHECK:STDOUT:   %char.as.EqWith.impl.NotEqual.call = icmp ne i8 %b, 120, !dbg !76
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.NotEqual.call), !dbg !77
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Less.call = icmp ult i8 %b, 120, !dbg !78
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Less.call), !dbg !79
// CHECK:STDOUT:   %char.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 %b, 120, !dbg !80
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !81
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Greater.call = icmp ugt i8 %b, 120, !dbg !82
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Greater.call), !dbg !83
// CHECK:STDOUT:   %char.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 %b, 120, !dbg !84
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !86
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CSubCharChar.Main(i8 %c) #0 !dbg !88 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !91
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i32 @"_COp.Char.Core:SubWith.7c3e388e0087434a.Core.2afba0967235ad27"(i8 %c, i8 120), !dbg !92
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.SubWith.impl.Op.call), !dbg !95
// CHECK:STDOUT:   %T.as_type.as.SubWith.impl.Op.call = call i32 @"_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.2afba0967235ad27"(i8 120, i8 %c), !dbg !97
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %T.as_type.as.SubWith.impl.Op.call), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubIntLiteral.Main() #0 !dbg !101 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !102
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !103
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !104
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt8.Main() #0 !dbg !106 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !107
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !108
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !109
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.15688e19507865c6(i8 %x) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.147e0bd019a31663(i16 %x) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !119
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.eadca4a969cf6f21(i32 %x) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %x) #0 !dbg !141 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.084a2801e0f5d754({} %x) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.Char.Core:SubWith.7c3e388e0087434a.Core.2afba0967235ad27"(i8 %self, i8 %other) #0 !dbg !152 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %self), !dbg !159
// CHECK:STDOUT:   %2 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %other), !dbg !162
// CHECK:STDOUT:   %3 = sub i32 %1, %2, !dbg !163
// CHECK:STDOUT:   ret i32 %3, !dbg !164
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.2afba0967235ad27"(i8 %self, i8 %other) #0 !dbg !166 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %self), !dbg !173
// CHECK:STDOUT:   %2 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %other), !dbg !174
// CHECK:STDOUT:   %3 = sub i32 %1, %2, !dbg !175
// CHECK:STDOUT:   ret i32 %3, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_COp.Char.Core:SubWith.295cf9b677564c5c.Core"(i8, i8)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %self) #0 !dbg !183 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.de54a264bd4d5c6f"(i8 %self), !dbg !189
// CHECK:STDOUT:   ret i32 %1, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.de54a264bd4d5c6f"(i8 %self) #0 !dbg !192 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.d19fcdc7c1bc8ad3"(i8 %self), !dbg !198
// CHECK:STDOUT:   ret i32 %1, !dbg !199
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.d19fcdc7c1bc8ad3"(i8 %from) #0 !dbg !201 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !206
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !209
// CHECK:STDOUT:   ret i32 %2, !dbg !210
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %self) #0 !dbg !212 {
// CHECK:STDOUT:   ret i8 %self, !dbg !218
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.15688e19507865c6, { 2, 0, 1 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.147e0bd019a31663, { 3, 0, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.eadca4a969cf6f21, { 5, 0, 4, 1, 2, 3 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.a4c7fafe8a72cac4, { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.084a2801e0f5d754, { 6, 0, 1, 2, 3, 4, 5 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1", { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "char_literals.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !13 = !{!6, !12}
// CHECK:STDOUT: !14 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !15 = !{null}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !17 = !{null, !6}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core", scope: null, file: !4, line: 15, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !25)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !12)
// CHECK:STDOUT: !24 = !DILocation(line: 15, column: 39, scope: !19)
// CHECK:STDOUT: !25 = !{!20}
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !4, line: 21, type: !16, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !27 = !DILocation(line: 22, column: 3, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 23, column: 3, scope: !26)
// CHECK:STDOUT: !29 = !{null, !7}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 24, column: 3, scope: !26)
// CHECK:STDOUT: !32 = !{null, !8}
// CHECK:STDOUT: !33 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 25, column: 3, scope: !26)
// CHECK:STDOUT: !37 = !DILocation(line: 26, column: 3, scope: !26)
// CHECK:STDOUT: !40 = !DILocation(line: 27, column: 3, scope: !26)
// CHECK:STDOUT: !43 = !DILocation(line: 21, column: 1, scope: !26)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "ConvertConstant", linkageName: "_CConvertConstant.Main", scope: null, file: !4, line: 30, type: !16, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !45 = !DILocation(line: 31, column: 3, scope: !44)
// CHECK:STDOUT: !46 = !DILocation(line: 32, column: 3, scope: !44)
// CHECK:STDOUT: !47 = !DILocation(line: 33, column: 3, scope: !44)
// CHECK:STDOUT: !48 = !DILocation(line: 34, column: 3, scope: !44)
// CHECK:STDOUT: !49 = !DILocation(line: 35, column: 3, scope: !44)
// CHECK:STDOUT: !50 = !DILocation(line: 30, column: 1, scope: !44)
// CHECK:STDOUT: !51 = distinct !DISubprogram(name: "CompareChar", linkageName: "_CCompareChar.Main", scope: null, file: !4, line: 38, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !87)
// CHECK:STDOUT: !52 = !DILocalVariable(arg: 1, scope: !51, type: !6)
// CHECK:STDOUT: !55 = !DILocation(line: 41, column: 11, scope: !51)
// CHECK:STDOUT: !56 = !DILocation(line: 41, column: 3, scope: !51)
// CHECK:STDOUT: !57 = !{null, !12}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !60 = !DILocation(line: 42, column: 11, scope: !51)
// CHECK:STDOUT: !61 = !DILocation(line: 42, column: 3, scope: !51)
// CHECK:STDOUT: !63 = !DILocation(line: 43, column: 11, scope: !51)
// CHECK:STDOUT: !64 = !DILocation(line: 43, column: 3, scope: !51)
// CHECK:STDOUT: !66 = !DILocation(line: 44, column: 11, scope: !51)
// CHECK:STDOUT: !67 = !DILocation(line: 44, column: 3, scope: !51)
// CHECK:STDOUT: !69 = !DILocation(line: 45, column: 11, scope: !51)
// CHECK:STDOUT: !70 = !DILocation(line: 45, column: 3, scope: !51)
// CHECK:STDOUT: !72 = !DILocation(line: 46, column: 11, scope: !51)
// CHECK:STDOUT: !73 = !DILocation(line: 46, column: 3, scope: !51)
// CHECK:STDOUT: !74 = !DILocation(line: 48, column: 11, scope: !51)
// CHECK:STDOUT: !75 = !DILocation(line: 48, column: 3, scope: !51)
// CHECK:STDOUT: !76 = !DILocation(line: 49, column: 11, scope: !51)
// CHECK:STDOUT: !77 = !DILocation(line: 49, column: 3, scope: !51)
// CHECK:STDOUT: !78 = !DILocation(line: 50, column: 11, scope: !51)
// CHECK:STDOUT: !79 = !DILocation(line: 50, column: 3, scope: !51)
// CHECK:STDOUT: !80 = !DILocation(line: 51, column: 11, scope: !51)
// CHECK:STDOUT: !81 = !DILocation(line: 51, column: 3, scope: !51)
// CHECK:STDOUT: !82 = !DILocation(line: 52, column: 11, scope: !51)
// CHECK:STDOUT: !83 = !DILocation(line: 52, column: 3, scope: !51)
// CHECK:STDOUT: !84 = !DILocation(line: 53, column: 11, scope: !51)
// CHECK:STDOUT: !85 = !DILocation(line: 53, column: 3, scope: !51)
// CHECK:STDOUT: !86 = !DILocation(line: 38, column: 1, scope: !51)
// CHECK:STDOUT: !87 = !{!52}
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "SubCharChar", linkageName: "_CSubCharChar.Main", scope: null, file: !4, line: 56, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !100)
// CHECK:STDOUT: !89 = !DILocalVariable(arg: 1, scope: !88, type: !6)
// CHECK:STDOUT: !91 = !DILocation(line: 57, column: 3, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 58, column: 11, scope: !88)
// CHECK:STDOUT: !93 = !{!11, !6, !6}
// CHECK:STDOUT: !94 = !DISubroutineType(types: !93)
// CHECK:STDOUT: !95 = !DILocation(line: 58, column: 3, scope: !88)
// CHECK:STDOUT: !97 = !DILocation(line: 59, column: 11, scope: !88)
// CHECK:STDOUT: !98 = !DILocation(line: 59, column: 3, scope: !88)
// CHECK:STDOUT: !99 = !DILocation(line: 56, column: 1, scope: !88)
// CHECK:STDOUT: !100 = !{!89}
// CHECK:STDOUT: !101 = distinct !DISubprogram(name: "AddSubIntLiteral", linkageName: "_CAddSubIntLiteral.Main", scope: null, file: !4, line: 62, type: !16, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !102 = !DILocation(line: 66, column: 3, scope: !101)
// CHECK:STDOUT: !103 = !DILocation(line: 67, column: 3, scope: !101)
// CHECK:STDOUT: !104 = !DILocation(line: 68, column: 3, scope: !101)
// CHECK:STDOUT: !105 = !DILocation(line: 62, column: 1, scope: !101)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "AddSubInt8", linkageName: "_CAddSubInt8.Main", scope: null, file: !4, line: 71, type: !16, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !107 = !DILocation(line: 75, column: 3, scope: !106)
// CHECK:STDOUT: !108 = !DILocation(line: 76, column: 3, scope: !106)
// CHECK:STDOUT: !109 = !DILocation(line: 77, column: 3, scope: !106)
// CHECK:STDOUT: !110 = !DILocation(line: 71, column: 1, scope: !106)
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.15688e19507865c6", scope: null, file: !4, line: 19, type: !18, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !115)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !6)
// CHECK:STDOUT: !114 = !DILocation(line: 19, column: 1, scope: !111)
// CHECK:STDOUT: !115 = !{!112}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.147e0bd019a31663", scope: null, file: !4, line: 19, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !120)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !7)
// CHECK:STDOUT: !119 = !DILocation(line: 19, column: 1, scope: !116)
// CHECK:STDOUT: !120 = !{!117}
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.eadca4a969cf6f21", scope: null, file: !4, line: 19, type: !33, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !125)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !8)
// CHECK:STDOUT: !124 = !DILocation(line: 19, column: 1, scope: !121)
// CHECK:STDOUT: !125 = !{!122}
// CHECK:STDOUT: !141 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.a4c7fafe8a72cac4", scope: null, file: !4, line: 19, type: !58, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !145)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !141, type: !12)
// CHECK:STDOUT: !144 = !DILocation(line: 19, column: 1, scope: !141)
// CHECK:STDOUT: !145 = !{!142}
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.084a2801e0f5d754", scope: null, file: !4, line: 19, type: !58, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !150)
// CHECK:STDOUT: !147 = !DILocalVariable(arg: 1, scope: !146, type: !12)
// CHECK:STDOUT: !149 = !DILocation(line: 19, column: 1, scope: !146)
// CHECK:STDOUT: !150 = !{!147}
// CHECK:STDOUT: !151 = !DIFile(filename: "{{.*}}/prelude/types/char.carbon", directory: "")
// CHECK:STDOUT: !152 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.7c3e388e0087434a.Core.2afba0967235ad27", scope: null, file: !151, line: 129, type: !94, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !165)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !152, type: !6)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 2, scope: !152, type: !6)
// CHECK:STDOUT: !159 = !DILocation(line: 130, column: 13, scope: !152)
// CHECK:STDOUT: !160 = !{!11, !6}
// CHECK:STDOUT: !161 = !DISubroutineType(types: !160)
// CHECK:STDOUT: !162 = !DILocation(line: 130, column: 29, scope: !152)
// CHECK:STDOUT: !163 = !DILocation(line: 130, column: 12, scope: !152)
// CHECK:STDOUT: !164 = !DILocation(line: 130, column: 5, scope: !152)
// CHECK:STDOUT: !165 = !{!153, !154}
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.2afba0967235ad27", scope: null, file: !151, line: 123, type: !94, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !177)
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 1, scope: !166, type: !6)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 2, scope: !166, type: !6)
// CHECK:STDOUT: !173 = !DILocation(line: 124, column: 13, scope: !166)
// CHECK:STDOUT: !174 = !DILocation(line: 124, column: 29, scope: !166)
// CHECK:STDOUT: !175 = !DILocation(line: 124, column: 12, scope: !166)
// CHECK:STDOUT: !176 = !DILocation(line: 124, column: 5, scope: !166)
// CHECK:STDOUT: !177 = !{!167, !168}
// CHECK:STDOUT: !183 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1", scope: null, file: !151, line: 42, type: !161, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !190)
// CHECK:STDOUT: !184 = !DILocalVariable(arg: 1, scope: !183, type: !6)
// CHECK:STDOUT: !189 = !DILocation(line: 43, column: 5, scope: !183)
// CHECK:STDOUT: !190 = !{!184}
// CHECK:STDOUT: !191 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.de54a264bd4d5c6f", scope: null, file: !191, line: 88, type: !161, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !200)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !6)
// CHECK:STDOUT: !198 = !DILocation(line: 89, column: 12, scope: !192)
// CHECK:STDOUT: !199 = !DILocation(line: 89, column: 5, scope: !192)
// CHECK:STDOUT: !200 = !{!193}
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.d19fcdc7c1bc8ad3", scope: null, file: !191, line: 80, type: !161, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !211)
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !201, type: !6)
// CHECK:STDOUT: !206 = !DILocation(line: 82, column: 17, scope: !201)
// CHECK:STDOUT: !207 = !{!6, !6}
// CHECK:STDOUT: !208 = !DISubroutineType(types: !207)
// CHECK:STDOUT: !209 = !DILocation(line: 82, column: 12, scope: !201)
// CHECK:STDOUT: !210 = !DILocation(line: 82, column: 5, scope: !201)
// CHECK:STDOUT: !211 = !{!202}
// CHECK:STDOUT: !212 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", scope: null, file: !191, line: 60, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !219)
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 1, scope: !212, type: !6)
// CHECK:STDOUT: !218 = !DILocation(line: 60, column: 29, scope: !212)
// CHECK:STDOUT: !219 = !{!213}
// CHECK:STDOUT:
