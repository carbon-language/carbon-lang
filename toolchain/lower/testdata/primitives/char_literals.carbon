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

fn Discard[T:! type](unused x: T) {}

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

// CHECK:STDOUT: ; ModuleID = 'char_literals.carbon'
// CHECK:STDOUT: source_filename = "char_literals.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i8 120, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main() #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 120), !dbg !15
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 120), !dbg !16
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 120), !dbg !17
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 120), !dbg !18
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 120), !dbg !19
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 120), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertConstant.Main() #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 120), !dbg !23
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 120), !dbg !24
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 120), !dbg !25
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 120), !dbg !26
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 120), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompareChar.Main(i8 %b) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call = icmp eq i8 120, %b, !dbg !34
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.EqWith.impl.Equal.call), !dbg !35
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call = icmp ne i8 120, %b, !dbg !36
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.EqWith.impl.NotEqual.call), !dbg !37
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call = icmp ult i8 120, %b, !dbg !38
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.Less.call), !dbg !39
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 120, %b, !dbg !40
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !41
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call = icmp ugt i8 120, %b, !dbg !42
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.Greater.call), !dbg !43
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 120, %b, !dbg !44
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !45
// CHECK:STDOUT:   %char.as.EqWith.impl.Equal.call = icmp eq i8 %b, 120, !dbg !46
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.Equal.call), !dbg !47
// CHECK:STDOUT:   %char.as.EqWith.impl.NotEqual.call = icmp ne i8 %b, 120, !dbg !48
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.NotEqual.call), !dbg !49
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Less.call = icmp ult i8 %b, 120, !dbg !50
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Less.call), !dbg !51
// CHECK:STDOUT:   %char.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 %b, 120, !dbg !52
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !53
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Greater.call = icmp ugt i8 %b, 120, !dbg !54
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Greater.call), !dbg !55
// CHECK:STDOUT:   %char.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 %b, 120, !dbg !56
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CSubCharChar.Main(i8 %c) #0 !dbg !59 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !62
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i32 @"_COp.Char.Core:SubWith.7c3e388e0087434a.Core.f0dfac78fde8177c"(i8 %c, i8 120), !dbg !63
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.SubWith.impl.Op.call), !dbg !64
// CHECK:STDOUT:   %T.as_type.as.SubWith.impl.Op.call = call i32 @"_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.f0dfac78fde8177c"(i8 120, i8 %c), !dbg !65
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %T.as_type.as.SubWith.impl.Op.call), !dbg !66
// CHECK:STDOUT:   ret void, !dbg !67
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubIntLiteral.Main() #0 !dbg !68 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !69
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !70
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !71
// CHECK:STDOUT:   ret void, !dbg !72
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt8.Main() #0 !dbg !73 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !74
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !75
// CHECK:STDOUT:   call void @_CDiscard.Main.084a2801e0f5d754({} zeroinitializer), !dbg !76
// CHECK:STDOUT:   ret void, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.15688e19507865c6(i8 %x) #0 !dbg !78 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.147e0bd019a31663(i16 %x) #0 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.eadca4a969cf6f21(i32 %x) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %x) #0 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !101
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.084a2801e0f5d754({} %x) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !105
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.Char.Core:SubWith.7c3e388e0087434a.Core.f0dfac78fde8177c"(i8 %self, i8 %other) #0 !dbg !106 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %self), !dbg !114
// CHECK:STDOUT:   %2 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %other), !dbg !115
// CHECK:STDOUT:   %3 = sub i32 %1, %2, !dbg !116
// CHECK:STDOUT:   ret i32 %3, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.f0dfac78fde8177c"(i8 %self, i8 %other) #0 !dbg !118 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %self), !dbg !122
// CHECK:STDOUT:   %2 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %other), !dbg !123
// CHECK:STDOUT:   %3 = sub i32 %1, %2, !dbg !124
// CHECK:STDOUT:   ret i32 %3, !dbg !125
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_COp.Char.Core:SubWith.295cf9b677564c5c.Core"(i8, i8)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %self) #0 !dbg !126 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9"(i8 %self), !dbg !131
// CHECK:STDOUT:   ret i32 %1, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9"(i8 %self) #0 !dbg !132 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.1d53f86bcab954ee"(i8 %self), !dbg !136
// CHECK:STDOUT:   ret i32 %1, !dbg !137
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.1d53f86bcab954ee"(i8 %from) #0 !dbg !138 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !141
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !142
// CHECK:STDOUT:   ret i32 %2, !dbg !143
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %self) #0 !dbg !144 {
// CHECK:STDOUT:   ret i8 %self, !dbg !149
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.15688e19507865c6, { 2, 0, 1 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.147e0bd019a31663, { 3, 0, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.eadca4a969cf6f21, { 5, 0, 4, 1, 2, 3 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.a4c7fafe8a72cac4, { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.084a2801e0f5d754, { 6, 0, 1, 2, 3, 4, 5 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee", { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "char_literals.carbon", directory: "")
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core", scope: null, file: !1, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 39, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !1, line: 21, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null}
// CHECK:STDOUT: !15 = !DILocation(line: 22, column: 3, scope: !12)
// CHECK:STDOUT: !16 = !DILocation(line: 23, column: 3, scope: !12)
// CHECK:STDOUT: !17 = !DILocation(line: 24, column: 3, scope: !12)
// CHECK:STDOUT: !18 = !DILocation(line: 25, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 26, column: 3, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 27, column: 3, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 21, column: 1, scope: !12)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "ConvertConstant", linkageName: "_CConvertConstant.Main", scope: null, file: !1, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !23 = !DILocation(line: 31, column: 3, scope: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 32, column: 3, scope: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 33, column: 3, scope: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 34, column: 3, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 35, column: 3, scope: !22)
// CHECK:STDOUT: !28 = !DILocation(line: 30, column: 1, scope: !22)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "CompareChar", linkageName: "_CCompareChar.Main", scope: null, file: !1, line: 38, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !32)
// CHECK:STDOUT: !30 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !31 = !{null, !7}
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !29, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 41, column: 11, scope: !29)
// CHECK:STDOUT: !35 = !DILocation(line: 41, column: 3, scope: !29)
// CHECK:STDOUT: !36 = !DILocation(line: 42, column: 11, scope: !29)
// CHECK:STDOUT: !37 = !DILocation(line: 42, column: 3, scope: !29)
// CHECK:STDOUT: !38 = !DILocation(line: 43, column: 11, scope: !29)
// CHECK:STDOUT: !39 = !DILocation(line: 43, column: 3, scope: !29)
// CHECK:STDOUT: !40 = !DILocation(line: 44, column: 11, scope: !29)
// CHECK:STDOUT: !41 = !DILocation(line: 44, column: 3, scope: !29)
// CHECK:STDOUT: !42 = !DILocation(line: 45, column: 11, scope: !29)
// CHECK:STDOUT: !43 = !DILocation(line: 45, column: 3, scope: !29)
// CHECK:STDOUT: !44 = !DILocation(line: 46, column: 11, scope: !29)
// CHECK:STDOUT: !45 = !DILocation(line: 46, column: 3, scope: !29)
// CHECK:STDOUT: !46 = !DILocation(line: 48, column: 11, scope: !29)
// CHECK:STDOUT: !47 = !DILocation(line: 48, column: 3, scope: !29)
// CHECK:STDOUT: !48 = !DILocation(line: 49, column: 11, scope: !29)
// CHECK:STDOUT: !49 = !DILocation(line: 49, column: 3, scope: !29)
// CHECK:STDOUT: !50 = !DILocation(line: 50, column: 11, scope: !29)
// CHECK:STDOUT: !51 = !DILocation(line: 50, column: 3, scope: !29)
// CHECK:STDOUT: !52 = !DILocation(line: 51, column: 11, scope: !29)
// CHECK:STDOUT: !53 = !DILocation(line: 51, column: 3, scope: !29)
// CHECK:STDOUT: !54 = !DILocation(line: 52, column: 11, scope: !29)
// CHECK:STDOUT: !55 = !DILocation(line: 52, column: 3, scope: !29)
// CHECK:STDOUT: !56 = !DILocation(line: 53, column: 11, scope: !29)
// CHECK:STDOUT: !57 = !DILocation(line: 53, column: 3, scope: !29)
// CHECK:STDOUT: !58 = !DILocation(line: 38, column: 1, scope: !29)
// CHECK:STDOUT: !59 = distinct !DISubprogram(name: "SubCharChar", linkageName: "_CSubCharChar.Main", scope: null, file: !1, line: 56, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !60)
// CHECK:STDOUT: !60 = !{!61}
// CHECK:STDOUT: !61 = !DILocalVariable(arg: 1, scope: !59, type: !7)
// CHECK:STDOUT: !62 = !DILocation(line: 57, column: 3, scope: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 58, column: 11, scope: !59)
// CHECK:STDOUT: !64 = !DILocation(line: 58, column: 3, scope: !59)
// CHECK:STDOUT: !65 = !DILocation(line: 59, column: 11, scope: !59)
// CHECK:STDOUT: !66 = !DILocation(line: 59, column: 3, scope: !59)
// CHECK:STDOUT: !67 = !DILocation(line: 56, column: 1, scope: !59)
// CHECK:STDOUT: !68 = distinct !DISubprogram(name: "AddSubIntLiteral", linkageName: "_CAddSubIntLiteral.Main", scope: null, file: !1, line: 62, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !69 = !DILocation(line: 66, column: 3, scope: !68)
// CHECK:STDOUT: !70 = !DILocation(line: 67, column: 3, scope: !68)
// CHECK:STDOUT: !71 = !DILocation(line: 68, column: 3, scope: !68)
// CHECK:STDOUT: !72 = !DILocation(line: 62, column: 1, scope: !68)
// CHECK:STDOUT: !73 = distinct !DISubprogram(name: "AddSubInt8", linkageName: "_CAddSubInt8.Main", scope: null, file: !1, line: 71, type: !13, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !74 = !DILocation(line: 75, column: 3, scope: !73)
// CHECK:STDOUT: !75 = !DILocation(line: 76, column: 3, scope: !73)
// CHECK:STDOUT: !76 = !DILocation(line: 77, column: 3, scope: !73)
// CHECK:STDOUT: !77 = !DILocation(line: 71, column: 1, scope: !73)
// CHECK:STDOUT: !78 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.15688e19507865c6", scope: null, file: !1, line: 19, type: !30, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !79)
// CHECK:STDOUT: !79 = !{!80}
// CHECK:STDOUT: !80 = !DILocalVariable(arg: 1, scope: !78, type: !7)
// CHECK:STDOUT: !81 = !DILocation(line: 19, column: 1, scope: !78)
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.147e0bd019a31663", scope: null, file: !1, line: 19, type: !83, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !86)
// CHECK:STDOUT: !83 = !DISubroutineType(types: !84)
// CHECK:STDOUT: !84 = !{null, !85}
// CHECK:STDOUT: !85 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !86 = !{!87}
// CHECK:STDOUT: !87 = !DILocalVariable(arg: 1, scope: !82, type: !85)
// CHECK:STDOUT: !88 = !DILocation(line: 19, column: 1, scope: !82)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.eadca4a969cf6f21", scope: null, file: !1, line: 19, type: !90, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !93)
// CHECK:STDOUT: !90 = !DISubroutineType(types: !91)
// CHECK:STDOUT: !91 = !{null, !92}
// CHECK:STDOUT: !92 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !93 = !{!94}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !89, type: !92)
// CHECK:STDOUT: !95 = !DILocation(line: 19, column: 1, scope: !89)
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.a4c7fafe8a72cac4", scope: null, file: !1, line: 19, type: !97, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !99)
// CHECK:STDOUT: !97 = !DISubroutineType(types: !98)
// CHECK:STDOUT: !98 = !{null, !8}
// CHECK:STDOUT: !99 = !{!100}
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 1, scope: !96, type: !8)
// CHECK:STDOUT: !101 = !DILocation(line: 19, column: 1, scope: !96)
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.084a2801e0f5d754", scope: null, file: !1, line: 19, type: !97, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !103)
// CHECK:STDOUT: !103 = !{!104}
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 1, scope: !102, type: !8)
// CHECK:STDOUT: !105 = !DILocation(line: 19, column: 1, scope: !102)
// CHECK:STDOUT: !106 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.7c3e388e0087434a.Core.f0dfac78fde8177c", scope: null, file: !107, line: 129, type: !108, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !111)
// CHECK:STDOUT: !107 = !DIFile(filename: "{{.*}}/prelude/types/char.carbon", directory: "")
// CHECK:STDOUT: !108 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !109 = !{!110, !7, !7}
// CHECK:STDOUT: !110 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !111 = !{!112, !113}
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !106, type: !7)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 2, scope: !106, type: !7)
// CHECK:STDOUT: !114 = !DILocation(line: 130, column: 13, scope: !106)
// CHECK:STDOUT: !115 = !DILocation(line: 130, column: 29, scope: !106)
// CHECK:STDOUT: !116 = !DILocation(line: 130, column: 12, scope: !106)
// CHECK:STDOUT: !117 = !DILocation(line: 130, column: 5, scope: !106)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.f0dfac78fde8177c", scope: null, file: !107, line: 123, type: !108, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !119)
// CHECK:STDOUT: !119 = !{!120, !121}
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !118, type: !7)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 2, scope: !118, type: !7)
// CHECK:STDOUT: !122 = !DILocation(line: 124, column: 13, scope: !118)
// CHECK:STDOUT: !123 = !DILocation(line: 124, column: 29, scope: !118)
// CHECK:STDOUT: !124 = !DILocation(line: 124, column: 12, scope: !118)
// CHECK:STDOUT: !125 = !DILocation(line: 124, column: 5, scope: !118)
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee", scope: null, file: !107, line: 42, type: !127, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !129)
// CHECK:STDOUT: !127 = !DISubroutineType(types: !128)
// CHECK:STDOUT: !128 = !{!110, !7}
// CHECK:STDOUT: !129 = !{!130}
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !126, type: !7)
// CHECK:STDOUT: !131 = !DILocation(line: 43, column: 5, scope: !126)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9", scope: null, file: !133, line: 88, type: !127, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !134)
// CHECK:STDOUT: !133 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !134 = !{!135}
// CHECK:STDOUT: !135 = !DILocalVariable(arg: 1, scope: !132, type: !7)
// CHECK:STDOUT: !136 = !DILocation(line: 89, column: 12, scope: !132)
// CHECK:STDOUT: !137 = !DILocation(line: 89, column: 5, scope: !132)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.1d53f86bcab954ee", scope: null, file: !133, line: 80, type: !127, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !139)
// CHECK:STDOUT: !139 = !{!140}
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 1, scope: !138, type: !7)
// CHECK:STDOUT: !141 = !DILocation(line: 82, column: 17, scope: !138)
// CHECK:STDOUT: !142 = !DILocation(line: 82, column: 12, scope: !138)
// CHECK:STDOUT: !143 = !DILocation(line: 82, column: 5, scope: !138)
// CHECK:STDOUT: !144 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", scope: null, file: !133, line: 60, type: !145, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !147)
// CHECK:STDOUT: !145 = !DISubroutineType(types: !146)
// CHECK:STDOUT: !146 = !{!7, !7}
// CHECK:STDOUT: !147 = !{!148}
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !144, type: !7)
// CHECK:STDOUT: !149 = !DILocation(line: 60, column: 29, scope: !144)
