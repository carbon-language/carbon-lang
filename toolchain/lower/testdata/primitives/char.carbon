// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/primitives/char.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/primitives/char.carbon

class CharLike {
  impl as Core.ImplicitAs(char) {
    fn Convert(unused self) -> char { return 'x'; }
  }
}

fn Discard[T: type](unused x: T) {}

fn Convert(c: char, u: u8) {
  Discard(c as u8);
  Discard(c as u16);
  Discard(c as u32);
  // Note, no conversion to i8 as u8 doesn't implicitly convert to i8.
  Discard(c as i16);
  Discard(c as i32);
  Discard(u as char);
}

fn ConvertConstant() {
  Discard(('x' as char) as u8);
  Discard(('x' as char) as u16);
  Discard(('x' as char) as u32);
  Discard(('x' as char) as i16);
  Discard(('x' as char) as i32);
  Discard((123 as u8) as char);
}

fn CompareCharChar(a: char, b: char) {
  Discard(a == b);
  Discard(a != b);
  Discard(a < b);
  Discard(a <= b);
  Discard(a > b);
  Discard(a >= b);
}

fn CompareCharCharLike(a: char, b: CharLike) {
  Discard(a == b);
  Discard(a != b);
  Discard(a < b);
  Discard(a <= b);
  Discard(a > b);
  Discard(a >= b);

  Discard(b == a);
  Discard(b != a);
  Discard(b < a);
  Discard(b <= a);
  Discard(b > a);
  Discard(b >= a);
}

fn SubCharChar(c: char, d: CharLike) {
  Discard(c - c);
  Discard(c - d);
  Discard(d - c);
}

fn AddSubInt8(c: char, i: i8) {
  Discard(c + i);
  Discard(i + c);
  Discard(c - i);
}

fn AddSubInt16(c: char, i: i16) {
  Discard(c + i);
  Discard(i + c);
  Discard(c - i);
}

fn AddSubInt32(c: char, i: i32) {
  Discard(c + i);
  Discard(i + c);
  Discard(c - i);
}

fn AddSubUInt8(c: char, i: u8) {
  Discard(c + i);
  Discard(i + c);
  Discard(c - i);
}

fn AddSubUInt16(c: char, i: u16) {
  Discard(c + i);
  Discard(i + c);
  Discard(c - i);
}

fn AddSubUInt32(c: char, i: u32) {
  Discard(c + i);
  Discard(i + c);
  Discard(c - i);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'char.carbon'
// CHECK:STDOUT: source_filename = "char.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i8 120, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(i8 %c, i8 %u) #0 !dbg !40 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %c), !dbg !46
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc23 = call i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.d183f33d898732ee"(i8 %c), !dbg !49
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 %char.as.As.impl.Convert.call.loc23), !dbg !52
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc24 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.b3e492be1929cf8f"(i8 %c), !dbg !55
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.As.impl.Convert.call.loc24), !dbg !58
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc26 = call i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.05fbd7e2dbdd4666"(i8 %c), !dbg !61
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 %char.as.As.impl.Convert.call.loc26), !dbg !64
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc27 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %c), !dbg !67
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.As.impl.Convert.call.loc27), !dbg !70
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %u), !dbg !74
// CHECK:STDOUT:   ret void, !dbg !75
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertConstant.Main() #0 !dbg !77 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 120), !dbg !78
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc33 = call i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.d183f33d898732ee"(i8 120), !dbg !79
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 %char.as.As.impl.Convert.call.loc33), !dbg !80
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc34 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.b3e492be1929cf8f"(i8 120), !dbg !81
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.As.impl.Convert.call.loc34), !dbg !82
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc35 = call i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.05fbd7e2dbdd4666"(i8 120), !dbg !83
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 %char.as.As.impl.Convert.call.loc35), !dbg !84
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc36 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 120), !dbg !85
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.As.impl.Convert.call.loc36), !dbg !86
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 123), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !88
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompareCharChar.Main(i8 %a, i8 %b) #0 !dbg !89 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.EqWith.impl.Equal.call = icmp eq i8 %a, %b, !dbg !94
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.Equal.call), !dbg !96
// CHECK:STDOUT:   %char.as.EqWith.impl.NotEqual.call = icmp ne i8 %a, %b, !dbg !99
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.NotEqual.call), !dbg !101
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Less.call = icmp ult i8 %a, %b, !dbg !102
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Less.call), !dbg !104
// CHECK:STDOUT:   %char.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 %a, %b, !dbg !105
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !107
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Greater.call = icmp ugt i8 %a, %b, !dbg !108
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Greater.call), !dbg !110
// CHECK:STDOUT:   %char.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 %a, %b, !dbg !111
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !113
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompareCharCharLike.Main(i8 %a, ptr %b) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc50 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !122
// CHECK:STDOUT:   %char.as.EqWith.impl.Equal.call = icmp eq i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc50, !dbg !121
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.Equal.call), !dbg !123
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc51 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !125
// CHECK:STDOUT:   %char.as.EqWith.impl.NotEqual.call = icmp ne i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc51, !dbg !124
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.NotEqual.call), !dbg !126
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc52 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !128
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Less.call = icmp ult i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc52, !dbg !127
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Less.call), !dbg !129
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc53 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !131
// CHECK:STDOUT:   %char.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc53, !dbg !130
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !132
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc54 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !134
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Greater.call = icmp ugt i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc54, !dbg !133
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Greater.call), !dbg !135
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc55 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !137
// CHECK:STDOUT:   %char.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc55, !dbg !136
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !138
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc57 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !139
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call = icmp eq i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc57, %a, !dbg !139
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.EqWith.impl.Equal.call), !dbg !141
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc58 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !142
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call = icmp ne i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc58, %a, !dbg !142
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.EqWith.impl.NotEqual.call), !dbg !144
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc59 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !145
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call = icmp ult i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc59, %a, !dbg !145
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.Less.call), !dbg !147
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc60 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !148
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc60, %a, !dbg !148
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !150
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc61 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !151
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call = icmp ugt i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc61, %a, !dbg !151
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.Greater.call), !dbg !153
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc62 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !154
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc62, %a, !dbg !154
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !156
// CHECK:STDOUT:   ret void, !dbg !157
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CSubCharChar.Main(i8 %c, ptr %d) #0 !dbg !159 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call.loc66 = call i32 @"_COp.Char.Core:SubWith.295cf9b677564c5c.Core"(i8 %c, i8 %c), !dbg !164
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.SubWith.impl.Op.call.loc66), !dbg !166
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc67 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %d), !dbg !168
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call.loc67 = call i32 @"_COp.Char.Core:SubWith.7c3e388e0087434a.Core.9188b823847778f0"(i8 %c, i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc67), !dbg !167
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.SubWith.impl.Op.call.loc67), !dbg !169
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc68 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %d), !dbg !170
// CHECK:STDOUT:   %T.as_type.as.SubWith.impl.Op.call = call i32 @"_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.9188b823847778f0"(i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc68, i8 %c), !dbg !170
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %T.as_type.as.SubWith.impl.Op.call), !dbg !172
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_COp.Char.Core:SubWith.295cf9b677564c5c.Core"(i8, i8)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt8.Main(i8 %c, i8 %i) #0 !dbg !175 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.be6db155159bf0e6"(i8 %c, i8 %i), !dbg !180
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !184
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.be6db155159bf0e6"(i8 %i, i8 %c), !dbg !185
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !189
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.be6db155159bf0e6"(i8 %c, i8 %i), !dbg !190
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !192
// CHECK:STDOUT:   ret void, !dbg !193
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt16.Main(i8 %c, i16 %i) #0 !dbg !195 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.b2ec6933266ee2f0"(i8 %c, i16 %i), !dbg !200
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !204
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.b2ec6933266ee2f0"(i16 %i, i8 %c), !dbg !205
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !209
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.b2ec6933266ee2f0"(i8 %c, i16 %i), !dbg !210
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !212
// CHECK:STDOUT:   ret void, !dbg !213
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt32.Main(i8 %c, i32 %i) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.3c06bceea9ba44a7"(i8 %c, i32 %i), !dbg !220
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !224
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.3c06bceea9ba44a7"(i32 %i, i8 %c), !dbg !225
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !229
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.3c06bceea9ba44a7"(i8 %c, i32 %i), !dbg !230
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !232
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubUInt8.Main(i8 %c, i8 %i) #0 !dbg !235 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.c8c7baf2b5955a72"(i8 %c, i8 %i), !dbg !240
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !244
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.c8c7baf2b5955a72"(i8 %i, i8 %c), !dbg !245
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !247
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.c8c7baf2b5955a72"(i8 %c, i8 %i), !dbg !248
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !250
// CHECK:STDOUT:   ret void, !dbg !251
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubUInt16.Main(i8 %c, i16 %i) #0 !dbg !253 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.61bb220507f77095"(i8 %c, i16 %i), !dbg !258
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !262
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.61bb220507f77095"(i16 %i, i8 %c), !dbg !263
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !267
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.61bb220507f77095"(i8 %c, i16 %i), !dbg !268
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !270
// CHECK:STDOUT:   ret void, !dbg !271
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubUInt32.Main(i8 %c, i32 %i) #0 !dbg !273 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.4494a819d074492e"(i8 %c, i32 %i), !dbg !278
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !282
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.4494a819d074492e"(i32 %i, i8 %c), !dbg !283
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !287
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.4494a819d074492e"(i8 %c, i32 %i), !dbg !288
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !290
// CHECK:STDOUT:   ret void, !dbg !291
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.15688e19507865c6(i8 %x) #0 !dbg !293 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !296
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.d183f33d898732ee"(i8 %self) #0 !dbg !299 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.335e322ca15c76a1"(i8 %self), !dbg !305
// CHECK:STDOUT:   ret i16 %1, !dbg !305
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.147e0bd019a31663(i16 %x) #0 !dbg !307 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !310
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.b3e492be1929cf8f"(i8 %self) #0 !dbg !312 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.9b7e0edf5c8a8a84"(i8 %self), !dbg !318
// CHECK:STDOUT:   ret i32 %1, !dbg !318
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.eadca4a969cf6f21(i32 %x) #0 !dbg !320 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !323
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.05fbd7e2dbdd4666"(i8 %self) #0 !dbg !325 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.90fccc2ec10748e3"(i8 %self), !dbg !331
// CHECK:STDOUT:   ret i16 %1, !dbg !331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %self) #0 !dbg !338 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.de54a264bd4d5c6f"(i8 %self), !dbg !344
// CHECK:STDOUT:   ret i32 %1, !dbg !344
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %x) #0 !dbg !356 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !359
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.Char.Core:SubWith.7c3e388e0087434a.Core.9188b823847778f0"(i8 %self, i8 %other) #0 !dbg !361 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %self), !dbg !368
// CHECK:STDOUT:   %2 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %other), !dbg !369
// CHECK:STDOUT:   %3 = sub i32 %1, %2, !dbg !370
// CHECK:STDOUT:   ret i32 %3, !dbg !371
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.9188b823847778f0"(i8 %self, i8 %other) #0 !dbg !373 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %self), !dbg !380
// CHECK:STDOUT:   %2 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1"(i8 %other), !dbg !381
// CHECK:STDOUT:   %3 = sub i32 %1, %2, !dbg !382
// CHECK:STDOUT:   ret i32 %3, !dbg !383
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.be6db155159bf0e6"(i8 %self, i8 %other) #0 !dbg !385 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !394
// CHECK:STDOUT:   ret i8 %1, !dbg !396
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.be6db155159bf0e6"(i8 %self, i8 %other) #0 !dbg !398 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !407
// CHECK:STDOUT:   ret i8 %1, !dbg !409
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.be6db155159bf0e6"(i8 %self, i8 %other) #0 !dbg !411 {
// CHECK:STDOUT:   %1 = sub i8 %self, %other, !dbg !420
// CHECK:STDOUT:   ret i8 %1, !dbg !422
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.b2ec6933266ee2f0"(i8 %self, i16 %other) #0 !dbg !424 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !432
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !433
// CHECK:STDOUT:   ret i8 %2, !dbg !435
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.b2ec6933266ee2f0"(i16 %self, i8 %other) #0 !dbg !437 {
// CHECK:STDOUT:   %1 = trunc i16 %self to i8, !dbg !444
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !446
// CHECK:STDOUT:   ret i8 %2, !dbg !448
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.b2ec6933266ee2f0"(i8 %self, i16 %other) #0 !dbg !450 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !458
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !459
// CHECK:STDOUT:   ret i8 %2, !dbg !461
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.3c06bceea9ba44a7"(i8 %self, i32 %other) #0 !dbg !463 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !471
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !472
// CHECK:STDOUT:   ret i8 %2, !dbg !474
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.3c06bceea9ba44a7"(i32 %self, i8 %other) #0 !dbg !476 {
// CHECK:STDOUT:   %1 = trunc i32 %self to i8, !dbg !483
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !485
// CHECK:STDOUT:   ret i8 %2, !dbg !487
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.3c06bceea9ba44a7"(i8 %self, i32 %other) #0 !dbg !489 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !497
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !498
// CHECK:STDOUT:   ret i8 %2, !dbg !500
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.c8c7baf2b5955a72"(i8 %self, i8 %other) #0 !dbg !502 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !511
// CHECK:STDOUT:   ret i8 %1, !dbg !513
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.c8c7baf2b5955a72"(i8 %self, i8 %other) #0 !dbg !515 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !524
// CHECK:STDOUT:   ret i8 %1, !dbg !526
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.c8c7baf2b5955a72"(i8 %self, i8 %other) #0 !dbg !528 {
// CHECK:STDOUT:   %1 = sub i8 %self, %other, !dbg !537
// CHECK:STDOUT:   ret i8 %1, !dbg !539
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.61bb220507f77095"(i8 %self, i16 %other) #0 !dbg !541 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !549
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !550
// CHECK:STDOUT:   ret i8 %2, !dbg !552
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.61bb220507f77095"(i16 %self, i8 %other) #0 !dbg !554 {
// CHECK:STDOUT:   %1 = trunc i16 %self to i8, !dbg !561
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !563
// CHECK:STDOUT:   ret i8 %2, !dbg !565
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.61bb220507f77095"(i8 %self, i16 %other) #0 !dbg !567 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !575
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !576
// CHECK:STDOUT:   ret i8 %2, !dbg !578
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.4494a819d074492e"(i8 %self, i32 %other) #0 !dbg !580 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !588
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !589
// CHECK:STDOUT:   ret i8 %2, !dbg !591
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.4494a819d074492e"(i32 %self, i8 %other) #0 !dbg !593 {
// CHECK:STDOUT:   %1 = trunc i32 %self to i8, !dbg !600
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !602
// CHECK:STDOUT:   ret i8 %2, !dbg !604
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.4494a819d074492e"(i8 %self, i32 %other) #0 !dbg !606 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !614
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !615
// CHECK:STDOUT:   ret i8 %2, !dbg !617
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.335e322ca15c76a1"(i8 %self) #0 !dbg !620 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.e4ef44314f014f83"(i8 %self), !dbg !626
// CHECK:STDOUT:   ret i16 %1, !dbg !627
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.9b7e0edf5c8a8a84"(i8 %self) #0 !dbg !629 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.62c2f9adcd5bc12d"(i8 %self), !dbg !635
// CHECK:STDOUT:   ret i32 %1, !dbg !636
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.90fccc2ec10748e3"(i8 %self) #0 !dbg !638 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.4f3746a8c9265ebe"(i8 %self), !dbg !644
// CHECK:STDOUT:   ret i16 %1, !dbg !645
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.de54a264bd4d5c6f"(i8 %self) #0 !dbg !647 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.d19fcdc7c1bc8ad3"(i8 %self), !dbg !653
// CHECK:STDOUT:   ret i32 %1, !dbg !654
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.e4ef44314f014f83"(i8 %from) #0 !dbg !656 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !661
// CHECK:STDOUT:   %2 = zext i8 %1 to i16, !dbg !664
// CHECK:STDOUT:   ret i16 %2, !dbg !665
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.62c2f9adcd5bc12d"(i8 %from) #0 !dbg !667 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !672
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !673
// CHECK:STDOUT:   ret i32 %2, !dbg !674
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.4f3746a8c9265ebe"(i8 %from) #0 !dbg !676 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !681
// CHECK:STDOUT:   %2 = zext i8 %1 to i16, !dbg !682
// CHECK:STDOUT:   ret i16 %2, !dbg !683
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.d19fcdc7c1bc8ad3"(i8 %from) #0 !dbg !685 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !690
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !691
// CHECK:STDOUT:   ret i32 %2, !dbg !692
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %self) #0 !dbg !694 {
// CHECK:STDOUT:   ret i8 %self, !dbg !700
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.15688e19507865c6, { 21, 0, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.d183f33d898732ee", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.147e0bd019a31663, { 3, 0, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.b3e492be1929cf8f", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.eadca4a969cf6f21, { 6, 0, 5, 1, 2, 3, 4 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.05fbd7e2dbdd4666", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1", { 0, 1, 2, 3, 5, 4 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.a4c7fafe8a72cac4, { 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "char.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !9 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !11 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !12 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !13 = !{!6, !12}
// CHECK:STDOUT: !14 = !DISubroutineType(types: !13)
// CHECK:STDOUT: !15 = !{null, !6, !6}
// CHECK:STDOUT: !16 = !DISubroutineType(types: !15)
// CHECK:STDOUT: !17 = !{null}
// CHECK:STDOUT: !18 = !DISubroutineType(types: !17)
// CHECK:STDOUT: !19 = !{null, !6, !12}
// CHECK:STDOUT: !20 = !DISubroutineType(types: !19)
// CHECK:STDOUT: !21 = !{!10, !6, !6}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !{null, !6, !11}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !{null, !6, !9}
// CHECK:STDOUT: !26 = !DISubroutineType(types: !25)
// CHECK:STDOUT: !27 = !{null, !6, !10}
// CHECK:STDOUT: !28 = !DISubroutineType(types: !27)
// CHECK:STDOUT: !29 = !{null, !6, !7}
// CHECK:STDOUT: !30 = !DISubroutineType(types: !29)
// CHECK:STDOUT: !31 = !{null, !6, !8}
// CHECK:STDOUT: !32 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core", scope: null, file: !4, line: 15, type: !14, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !39)
// CHECK:STDOUT: !34 = !DILocalVariable(arg: 1, scope: !33, type: !12)
// CHECK:STDOUT: !38 = !DILocation(line: 15, column: 39, scope: !33)
// CHECK:STDOUT: !39 = !{!34}
// CHECK:STDOUT: !40 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !4, line: 21, type: !16, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !76)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 1, scope: !40, type: !6)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 2, scope: !40, type: !6)
// CHECK:STDOUT: !46 = !DILocation(line: 22, column: 3, scope: !40)
// CHECK:STDOUT: !47 = !{null, !6}
// CHECK:STDOUT: !48 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !49 = !DILocation(line: 23, column: 11, scope: !40)
// CHECK:STDOUT: !50 = !{!7, !6}
// CHECK:STDOUT: !51 = !DISubroutineType(types: !50)
// CHECK:STDOUT: !52 = !DILocation(line: 23, column: 3, scope: !40)
// CHECK:STDOUT: !53 = !{null, !7}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !55 = !DILocation(line: 24, column: 11, scope: !40)
// CHECK:STDOUT: !56 = !{!8, !6}
// CHECK:STDOUT: !57 = !DISubroutineType(types: !56)
// CHECK:STDOUT: !58 = !DILocation(line: 24, column: 3, scope: !40)
// CHECK:STDOUT: !59 = !{null, !8}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = !DILocation(line: 26, column: 11, scope: !40)
// CHECK:STDOUT: !62 = !{!9, !6}
// CHECK:STDOUT: !63 = !DISubroutineType(types: !62)
// CHECK:STDOUT: !64 = !DILocation(line: 26, column: 3, scope: !40)
// CHECK:STDOUT: !67 = !DILocation(line: 27, column: 11, scope: !40)
// CHECK:STDOUT: !68 = !{!10, !6}
// CHECK:STDOUT: !69 = !DISubroutineType(types: !68)
// CHECK:STDOUT: !70 = !DILocation(line: 27, column: 3, scope: !40)
// CHECK:STDOUT: !74 = !DILocation(line: 28, column: 3, scope: !40)
// CHECK:STDOUT: !75 = !DILocation(line: 21, column: 1, scope: !40)
// CHECK:STDOUT: !76 = !{!41, !42}
// CHECK:STDOUT: !77 = distinct !DISubprogram(name: "ConvertConstant", linkageName: "_CConvertConstant.Main", scope: null, file: !4, line: 31, type: !18, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !78 = !DILocation(line: 32, column: 3, scope: !77)
// CHECK:STDOUT: !79 = !DILocation(line: 33, column: 11, scope: !77)
// CHECK:STDOUT: !80 = !DILocation(line: 33, column: 3, scope: !77)
// CHECK:STDOUT: !81 = !DILocation(line: 34, column: 11, scope: !77)
// CHECK:STDOUT: !82 = !DILocation(line: 34, column: 3, scope: !77)
// CHECK:STDOUT: !83 = !DILocation(line: 35, column: 11, scope: !77)
// CHECK:STDOUT: !84 = !DILocation(line: 35, column: 3, scope: !77)
// CHECK:STDOUT: !85 = !DILocation(line: 36, column: 11, scope: !77)
// CHECK:STDOUT: !86 = !DILocation(line: 36, column: 3, scope: !77)
// CHECK:STDOUT: !87 = !DILocation(line: 37, column: 3, scope: !77)
// CHECK:STDOUT: !88 = !DILocation(line: 31, column: 1, scope: !77)
// CHECK:STDOUT: !89 = distinct !DISubprogram(name: "CompareCharChar", linkageName: "_CCompareCharChar.Main", scope: null, file: !4, line: 40, type: !16, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !115)
// CHECK:STDOUT: !90 = !DILocalVariable(arg: 1, scope: !89, type: !6)
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 2, scope: !89, type: !6)
// CHECK:STDOUT: !94 = !DILocation(line: 41, column: 11, scope: !89)
// CHECK:STDOUT: !96 = !DILocation(line: 41, column: 3, scope: !89)
// CHECK:STDOUT: !97 = !{null, !12}
// CHECK:STDOUT: !98 = !DISubroutineType(types: !97)
// CHECK:STDOUT: !99 = !DILocation(line: 42, column: 11, scope: !89)
// CHECK:STDOUT: !101 = !DILocation(line: 42, column: 3, scope: !89)
// CHECK:STDOUT: !102 = !DILocation(line: 43, column: 11, scope: !89)
// CHECK:STDOUT: !104 = !DILocation(line: 43, column: 3, scope: !89)
// CHECK:STDOUT: !105 = !DILocation(line: 44, column: 11, scope: !89)
// CHECK:STDOUT: !107 = !DILocation(line: 44, column: 3, scope: !89)
// CHECK:STDOUT: !108 = !DILocation(line: 45, column: 11, scope: !89)
// CHECK:STDOUT: !110 = !DILocation(line: 45, column: 3, scope: !89)
// CHECK:STDOUT: !111 = !DILocation(line: 46, column: 11, scope: !89)
// CHECK:STDOUT: !113 = !DILocation(line: 46, column: 3, scope: !89)
// CHECK:STDOUT: !114 = !DILocation(line: 40, column: 1, scope: !89)
// CHECK:STDOUT: !115 = !{!90, !91}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "CompareCharCharLike", linkageName: "_CCompareCharCharLike.Main", scope: null, file: !4, line: 49, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !158)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !6)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 2, scope: !116, type: !12)
// CHECK:STDOUT: !121 = !DILocation(line: 50, column: 11, scope: !116)
// CHECK:STDOUT: !122 = !DILocation(line: 50, column: 16, scope: !116)
// CHECK:STDOUT: !123 = !DILocation(line: 50, column: 3, scope: !116)
// CHECK:STDOUT: !124 = !DILocation(line: 51, column: 11, scope: !116)
// CHECK:STDOUT: !125 = !DILocation(line: 51, column: 16, scope: !116)
// CHECK:STDOUT: !126 = !DILocation(line: 51, column: 3, scope: !116)
// CHECK:STDOUT: !127 = !DILocation(line: 52, column: 11, scope: !116)
// CHECK:STDOUT: !128 = !DILocation(line: 52, column: 15, scope: !116)
// CHECK:STDOUT: !129 = !DILocation(line: 52, column: 3, scope: !116)
// CHECK:STDOUT: !130 = !DILocation(line: 53, column: 11, scope: !116)
// CHECK:STDOUT: !131 = !DILocation(line: 53, column: 16, scope: !116)
// CHECK:STDOUT: !132 = !DILocation(line: 53, column: 3, scope: !116)
// CHECK:STDOUT: !133 = !DILocation(line: 54, column: 11, scope: !116)
// CHECK:STDOUT: !134 = !DILocation(line: 54, column: 15, scope: !116)
// CHECK:STDOUT: !135 = !DILocation(line: 54, column: 3, scope: !116)
// CHECK:STDOUT: !136 = !DILocation(line: 55, column: 11, scope: !116)
// CHECK:STDOUT: !137 = !DILocation(line: 55, column: 16, scope: !116)
// CHECK:STDOUT: !138 = !DILocation(line: 55, column: 3, scope: !116)
// CHECK:STDOUT: !139 = !DILocation(line: 57, column: 11, scope: !116)
// CHECK:STDOUT: !141 = !DILocation(line: 57, column: 3, scope: !116)
// CHECK:STDOUT: !142 = !DILocation(line: 58, column: 11, scope: !116)
// CHECK:STDOUT: !144 = !DILocation(line: 58, column: 3, scope: !116)
// CHECK:STDOUT: !145 = !DILocation(line: 59, column: 11, scope: !116)
// CHECK:STDOUT: !147 = !DILocation(line: 59, column: 3, scope: !116)
// CHECK:STDOUT: !148 = !DILocation(line: 60, column: 11, scope: !116)
// CHECK:STDOUT: !150 = !DILocation(line: 60, column: 3, scope: !116)
// CHECK:STDOUT: !151 = !DILocation(line: 61, column: 11, scope: !116)
// CHECK:STDOUT: !153 = !DILocation(line: 61, column: 3, scope: !116)
// CHECK:STDOUT: !154 = !DILocation(line: 62, column: 11, scope: !116)
// CHECK:STDOUT: !156 = !DILocation(line: 62, column: 3, scope: !116)
// CHECK:STDOUT: !157 = !DILocation(line: 49, column: 1, scope: !116)
// CHECK:STDOUT: !158 = !{!117, !118}
// CHECK:STDOUT: !159 = distinct !DISubprogram(name: "SubCharChar", linkageName: "_CSubCharChar.Main", scope: null, file: !4, line: 65, type: !20, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !174)
// CHECK:STDOUT: !160 = !DILocalVariable(arg: 1, scope: !159, type: !6)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 2, scope: !159, type: !12)
// CHECK:STDOUT: !164 = !DILocation(line: 66, column: 11, scope: !159)
// CHECK:STDOUT: !166 = !DILocation(line: 66, column: 3, scope: !159)
// CHECK:STDOUT: !167 = !DILocation(line: 67, column: 11, scope: !159)
// CHECK:STDOUT: !168 = !DILocation(line: 67, column: 15, scope: !159)
// CHECK:STDOUT: !169 = !DILocation(line: 67, column: 3, scope: !159)
// CHECK:STDOUT: !170 = !DILocation(line: 68, column: 11, scope: !159)
// CHECK:STDOUT: !172 = !DILocation(line: 68, column: 3, scope: !159)
// CHECK:STDOUT: !173 = !DILocation(line: 65, column: 1, scope: !159)
// CHECK:STDOUT: !174 = !{!160, !161}
// CHECK:STDOUT: !175 = distinct !DISubprogram(name: "AddSubInt8", linkageName: "_CAddSubInt8.Main", scope: null, file: !4, line: 71, type: !24, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !194)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !175, type: !6)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 2, scope: !175, type: !11)
// CHECK:STDOUT: !180 = !DILocation(line: 72, column: 11, scope: !175)
// CHECK:STDOUT: !182 = !{!6, !6, !11}
// CHECK:STDOUT: !183 = !DISubroutineType(types: !182)
// CHECK:STDOUT: !184 = !DILocation(line: 72, column: 3, scope: !175)
// CHECK:STDOUT: !185 = !DILocation(line: 73, column: 11, scope: !175)
// CHECK:STDOUT: !187 = !{!6, !11, !6}
// CHECK:STDOUT: !188 = !DISubroutineType(types: !187)
// CHECK:STDOUT: !189 = !DILocation(line: 73, column: 3, scope: !175)
// CHECK:STDOUT: !190 = !DILocation(line: 74, column: 11, scope: !175)
// CHECK:STDOUT: !192 = !DILocation(line: 74, column: 3, scope: !175)
// CHECK:STDOUT: !193 = !DILocation(line: 71, column: 1, scope: !175)
// CHECK:STDOUT: !194 = !{!176, !177}
// CHECK:STDOUT: !195 = distinct !DISubprogram(name: "AddSubInt16", linkageName: "_CAddSubInt16.Main", scope: null, file: !4, line: 77, type: !26, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !214)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !195, type: !6)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 2, scope: !195, type: !9)
// CHECK:STDOUT: !200 = !DILocation(line: 78, column: 11, scope: !195)
// CHECK:STDOUT: !202 = !{!6, !6, !9}
// CHECK:STDOUT: !203 = !DISubroutineType(types: !202)
// CHECK:STDOUT: !204 = !DILocation(line: 78, column: 3, scope: !195)
// CHECK:STDOUT: !205 = !DILocation(line: 79, column: 11, scope: !195)
// CHECK:STDOUT: !207 = !{!6, !9, !6}
// CHECK:STDOUT: !208 = !DISubroutineType(types: !207)
// CHECK:STDOUT: !209 = !DILocation(line: 79, column: 3, scope: !195)
// CHECK:STDOUT: !210 = !DILocation(line: 80, column: 11, scope: !195)
// CHECK:STDOUT: !212 = !DILocation(line: 80, column: 3, scope: !195)
// CHECK:STDOUT: !213 = !DILocation(line: 77, column: 1, scope: !195)
// CHECK:STDOUT: !214 = !{!196, !197}
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "AddSubInt32", linkageName: "_CAddSubInt32.Main", scope: null, file: !4, line: 83, type: !28, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !234)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 1, scope: !215, type: !6)
// CHECK:STDOUT: !217 = !DILocalVariable(arg: 2, scope: !215, type: !10)
// CHECK:STDOUT: !220 = !DILocation(line: 84, column: 11, scope: !215)
// CHECK:STDOUT: !222 = !{!6, !6, !10}
// CHECK:STDOUT: !223 = !DISubroutineType(types: !222)
// CHECK:STDOUT: !224 = !DILocation(line: 84, column: 3, scope: !215)
// CHECK:STDOUT: !225 = !DILocation(line: 85, column: 11, scope: !215)
// CHECK:STDOUT: !227 = !{!6, !10, !6}
// CHECK:STDOUT: !228 = !DISubroutineType(types: !227)
// CHECK:STDOUT: !229 = !DILocation(line: 85, column: 3, scope: !215)
// CHECK:STDOUT: !230 = !DILocation(line: 86, column: 11, scope: !215)
// CHECK:STDOUT: !232 = !DILocation(line: 86, column: 3, scope: !215)
// CHECK:STDOUT: !233 = !DILocation(line: 83, column: 1, scope: !215)
// CHECK:STDOUT: !234 = !{!216, !217}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "AddSubUInt8", linkageName: "_CAddSubUInt8.Main", scope: null, file: !4, line: 89, type: !16, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !252)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !6)
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 2, scope: !235, type: !6)
// CHECK:STDOUT: !240 = !DILocation(line: 90, column: 11, scope: !235)
// CHECK:STDOUT: !242 = !{!6, !6, !6}
// CHECK:STDOUT: !243 = !DISubroutineType(types: !242)
// CHECK:STDOUT: !244 = !DILocation(line: 90, column: 3, scope: !235)
// CHECK:STDOUT: !245 = !DILocation(line: 91, column: 11, scope: !235)
// CHECK:STDOUT: !247 = !DILocation(line: 91, column: 3, scope: !235)
// CHECK:STDOUT: !248 = !DILocation(line: 92, column: 11, scope: !235)
// CHECK:STDOUT: !250 = !DILocation(line: 92, column: 3, scope: !235)
// CHECK:STDOUT: !251 = !DILocation(line: 89, column: 1, scope: !235)
// CHECK:STDOUT: !252 = !{!236, !237}
// CHECK:STDOUT: !253 = distinct !DISubprogram(name: "AddSubUInt16", linkageName: "_CAddSubUInt16.Main", scope: null, file: !4, line: 95, type: !30, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !272)
// CHECK:STDOUT: !254 = !DILocalVariable(arg: 1, scope: !253, type: !6)
// CHECK:STDOUT: !255 = !DILocalVariable(arg: 2, scope: !253, type: !7)
// CHECK:STDOUT: !258 = !DILocation(line: 96, column: 11, scope: !253)
// CHECK:STDOUT: !260 = !{!6, !6, !7}
// CHECK:STDOUT: !261 = !DISubroutineType(types: !260)
// CHECK:STDOUT: !262 = !DILocation(line: 96, column: 3, scope: !253)
// CHECK:STDOUT: !263 = !DILocation(line: 97, column: 11, scope: !253)
// CHECK:STDOUT: !265 = !{!6, !7, !6}
// CHECK:STDOUT: !266 = !DISubroutineType(types: !265)
// CHECK:STDOUT: !267 = !DILocation(line: 97, column: 3, scope: !253)
// CHECK:STDOUT: !268 = !DILocation(line: 98, column: 11, scope: !253)
// CHECK:STDOUT: !270 = !DILocation(line: 98, column: 3, scope: !253)
// CHECK:STDOUT: !271 = !DILocation(line: 95, column: 1, scope: !253)
// CHECK:STDOUT: !272 = !{!254, !255}
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "AddSubUInt32", linkageName: "_CAddSubUInt32.Main", scope: null, file: !4, line: 101, type: !32, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !292)
// CHECK:STDOUT: !274 = !DILocalVariable(arg: 1, scope: !273, type: !6)
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 2, scope: !273, type: !8)
// CHECK:STDOUT: !278 = !DILocation(line: 102, column: 11, scope: !273)
// CHECK:STDOUT: !280 = !{!6, !6, !8}
// CHECK:STDOUT: !281 = !DISubroutineType(types: !280)
// CHECK:STDOUT: !282 = !DILocation(line: 102, column: 3, scope: !273)
// CHECK:STDOUT: !283 = !DILocation(line: 103, column: 11, scope: !273)
// CHECK:STDOUT: !285 = !{!6, !8, !6}
// CHECK:STDOUT: !286 = !DISubroutineType(types: !285)
// CHECK:STDOUT: !287 = !DILocation(line: 103, column: 3, scope: !273)
// CHECK:STDOUT: !288 = !DILocation(line: 104, column: 11, scope: !273)
// CHECK:STDOUT: !290 = !DILocation(line: 104, column: 3, scope: !273)
// CHECK:STDOUT: !291 = !DILocation(line: 101, column: 1, scope: !273)
// CHECK:STDOUT: !292 = !{!274, !275}
// CHECK:STDOUT: !293 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.15688e19507865c6", scope: null, file: !4, line: 19, type: !48, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !297)
// CHECK:STDOUT: !294 = !DILocalVariable(arg: 1, scope: !293, type: !6)
// CHECK:STDOUT: !296 = !DILocation(line: 19, column: 1, scope: !293)
// CHECK:STDOUT: !297 = !{!294}
// CHECK:STDOUT: !298 = !DIFile(filename: "{{.*}}/prelude/types/char.carbon", directory: "")
// CHECK:STDOUT: !299 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.d183f33d898732ee", scope: null, file: !298, line: 42, type: !51, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !306)
// CHECK:STDOUT: !300 = !DILocalVariable(arg: 1, scope: !299, type: !6)
// CHECK:STDOUT: !305 = !DILocation(line: 43, column: 5, scope: !299)
// CHECK:STDOUT: !306 = !{!300}
// CHECK:STDOUT: !307 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.147e0bd019a31663", scope: null, file: !4, line: 19, type: !54, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !311)
// CHECK:STDOUT: !308 = !DILocalVariable(arg: 1, scope: !307, type: !7)
// CHECK:STDOUT: !310 = !DILocation(line: 19, column: 1, scope: !307)
// CHECK:STDOUT: !311 = !{!308}
// CHECK:STDOUT: !312 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.b3e492be1929cf8f", scope: null, file: !298, line: 42, type: !57, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !319)
// CHECK:STDOUT: !313 = !DILocalVariable(arg: 1, scope: !312, type: !6)
// CHECK:STDOUT: !318 = !DILocation(line: 43, column: 5, scope: !312)
// CHECK:STDOUT: !319 = !{!313}
// CHECK:STDOUT: !320 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.eadca4a969cf6f21", scope: null, file: !4, line: 19, type: !60, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !324)
// CHECK:STDOUT: !321 = !DILocalVariable(arg: 1, scope: !320, type: !8)
// CHECK:STDOUT: !323 = !DILocation(line: 19, column: 1, scope: !320)
// CHECK:STDOUT: !324 = !{!321}
// CHECK:STDOUT: !325 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.05fbd7e2dbdd4666", scope: null, file: !298, line: 42, type: !63, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !332)
// CHECK:STDOUT: !326 = !DILocalVariable(arg: 1, scope: !325, type: !6)
// CHECK:STDOUT: !331 = !DILocation(line: 43, column: 5, scope: !325)
// CHECK:STDOUT: !332 = !{!326}
// CHECK:STDOUT: !338 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.da068cc98c408be1", scope: null, file: !298, line: 42, type: !69, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !345)
// CHECK:STDOUT: !339 = !DILocalVariable(arg: 1, scope: !338, type: !6)
// CHECK:STDOUT: !344 = !DILocation(line: 43, column: 5, scope: !338)
// CHECK:STDOUT: !345 = !{!339}
// CHECK:STDOUT: !356 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.a4c7fafe8a72cac4", scope: null, file: !4, line: 19, type: !98, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !360)
// CHECK:STDOUT: !357 = !DILocalVariable(arg: 1, scope: !356, type: !12)
// CHECK:STDOUT: !359 = !DILocation(line: 19, column: 1, scope: !356)
// CHECK:STDOUT: !360 = !{!357}
// CHECK:STDOUT: !361 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.7c3e388e0087434a.Core.9188b823847778f0", scope: null, file: !298, line: 129, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !372)
// CHECK:STDOUT: !362 = !DILocalVariable(arg: 1, scope: !361, type: !6)
// CHECK:STDOUT: !363 = !DILocalVariable(arg: 2, scope: !361, type: !6)
// CHECK:STDOUT: !368 = !DILocation(line: 130, column: 13, scope: !361)
// CHECK:STDOUT: !369 = !DILocation(line: 130, column: 29, scope: !361)
// CHECK:STDOUT: !370 = !DILocation(line: 130, column: 12, scope: !361)
// CHECK:STDOUT: !371 = !DILocation(line: 130, column: 5, scope: !361)
// CHECK:STDOUT: !372 = !{!362, !363}
// CHECK:STDOUT: !373 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.9188b823847778f0", scope: null, file: !298, line: 123, type: !22, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !384)
// CHECK:STDOUT: !374 = !DILocalVariable(arg: 1, scope: !373, type: !6)
// CHECK:STDOUT: !375 = !DILocalVariable(arg: 2, scope: !373, type: !6)
// CHECK:STDOUT: !380 = !DILocation(line: 124, column: 13, scope: !373)
// CHECK:STDOUT: !381 = !DILocation(line: 124, column: 29, scope: !373)
// CHECK:STDOUT: !382 = !DILocation(line: 124, column: 12, scope: !373)
// CHECK:STDOUT: !383 = !DILocation(line: 124, column: 5, scope: !373)
// CHECK:STDOUT: !384 = !{!374, !375}
// CHECK:STDOUT: !385 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.be6db155159bf0e6", scope: null, file: !298, line: 94, type: !183, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !397)
// CHECK:STDOUT: !386 = !DILocalVariable(arg: 1, scope: !385, type: !6)
// CHECK:STDOUT: !387 = !DILocalVariable(arg: 2, scope: !385, type: !11)
// CHECK:STDOUT: !394 = !DILocation(line: 96, column: 13, scope: !385)
// CHECK:STDOUT: !396 = !DILocation(line: 96, column: 5, scope: !385)
// CHECK:STDOUT: !397 = !{!386, !387}
// CHECK:STDOUT: !398 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.be6db155159bf0e6", scope: null, file: !298, line: 102, type: !188, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !410)
// CHECK:STDOUT: !399 = !DILocalVariable(arg: 1, scope: !398, type: !11)
// CHECK:STDOUT: !400 = !DILocalVariable(arg: 2, scope: !398, type: !6)
// CHECK:STDOUT: !407 = !DILocation(line: 104, column: 13, scope: !398)
// CHECK:STDOUT: !409 = !DILocation(line: 104, column: 5, scope: !398)
// CHECK:STDOUT: !410 = !{!399, !400}
// CHECK:STDOUT: !411 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.be6db155159bf0e6", scope: null, file: !298, line: 110, type: !183, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !423)
// CHECK:STDOUT: !412 = !DILocalVariable(arg: 1, scope: !411, type: !6)
// CHECK:STDOUT: !413 = !DILocalVariable(arg: 2, scope: !411, type: !11)
// CHECK:STDOUT: !420 = !DILocation(line: 112, column: 13, scope: !411)
// CHECK:STDOUT: !422 = !DILocation(line: 112, column: 5, scope: !411)
// CHECK:STDOUT: !423 = !{!412, !413}
// CHECK:STDOUT: !424 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.b2ec6933266ee2f0", scope: null, file: !298, line: 94, type: !203, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !436)
// CHECK:STDOUT: !425 = !DILocalVariable(arg: 1, scope: !424, type: !6)
// CHECK:STDOUT: !426 = !DILocalVariable(arg: 2, scope: !424, type: !9)
// CHECK:STDOUT: !432 = !DILocation(line: 96, column: 29, scope: !424)
// CHECK:STDOUT: !433 = !DILocation(line: 96, column: 13, scope: !424)
// CHECK:STDOUT: !435 = !DILocation(line: 96, column: 5, scope: !424)
// CHECK:STDOUT: !436 = !{!425, !426}
// CHECK:STDOUT: !437 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.b2ec6933266ee2f0", scope: null, file: !298, line: 102, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !449)
// CHECK:STDOUT: !438 = !DILocalVariable(arg: 1, scope: !437, type: !9)
// CHECK:STDOUT: !439 = !DILocalVariable(arg: 2, scope: !437, type: !6)
// CHECK:STDOUT: !444 = !DILocation(line: 104, column: 14, scope: !437)
// CHECK:STDOUT: !446 = !DILocation(line: 104, column: 13, scope: !437)
// CHECK:STDOUT: !448 = !DILocation(line: 104, column: 5, scope: !437)
// CHECK:STDOUT: !449 = !{!438, !439}
// CHECK:STDOUT: !450 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.b2ec6933266ee2f0", scope: null, file: !298, line: 110, type: !203, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !462)
// CHECK:STDOUT: !451 = !DILocalVariable(arg: 1, scope: !450, type: !6)
// CHECK:STDOUT: !452 = !DILocalVariable(arg: 2, scope: !450, type: !9)
// CHECK:STDOUT: !458 = !DILocation(line: 112, column: 29, scope: !450)
// CHECK:STDOUT: !459 = !DILocation(line: 112, column: 13, scope: !450)
// CHECK:STDOUT: !461 = !DILocation(line: 112, column: 5, scope: !450)
// CHECK:STDOUT: !462 = !{!451, !452}
// CHECK:STDOUT: !463 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.3c06bceea9ba44a7", scope: null, file: !298, line: 94, type: !223, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !475)
// CHECK:STDOUT: !464 = !DILocalVariable(arg: 1, scope: !463, type: !6)
// CHECK:STDOUT: !465 = !DILocalVariable(arg: 2, scope: !463, type: !10)
// CHECK:STDOUT: !471 = !DILocation(line: 96, column: 29, scope: !463)
// CHECK:STDOUT: !472 = !DILocation(line: 96, column: 13, scope: !463)
// CHECK:STDOUT: !474 = !DILocation(line: 96, column: 5, scope: !463)
// CHECK:STDOUT: !475 = !{!464, !465}
// CHECK:STDOUT: !476 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.3c06bceea9ba44a7", scope: null, file: !298, line: 102, type: !228, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !488)
// CHECK:STDOUT: !477 = !DILocalVariable(arg: 1, scope: !476, type: !10)
// CHECK:STDOUT: !478 = !DILocalVariable(arg: 2, scope: !476, type: !6)
// CHECK:STDOUT: !483 = !DILocation(line: 104, column: 14, scope: !476)
// CHECK:STDOUT: !485 = !DILocation(line: 104, column: 13, scope: !476)
// CHECK:STDOUT: !487 = !DILocation(line: 104, column: 5, scope: !476)
// CHECK:STDOUT: !488 = !{!477, !478}
// CHECK:STDOUT: !489 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.3c06bceea9ba44a7", scope: null, file: !298, line: 110, type: !223, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !501)
// CHECK:STDOUT: !490 = !DILocalVariable(arg: 1, scope: !489, type: !6)
// CHECK:STDOUT: !491 = !DILocalVariable(arg: 2, scope: !489, type: !10)
// CHECK:STDOUT: !497 = !DILocation(line: 112, column: 29, scope: !489)
// CHECK:STDOUT: !498 = !DILocation(line: 112, column: 13, scope: !489)
// CHECK:STDOUT: !500 = !DILocation(line: 112, column: 5, scope: !489)
// CHECK:STDOUT: !501 = !{!490, !491}
// CHECK:STDOUT: !502 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.c8c7baf2b5955a72", scope: null, file: !298, line: 94, type: !243, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !514)
// CHECK:STDOUT: !503 = !DILocalVariable(arg: 1, scope: !502, type: !6)
// CHECK:STDOUT: !504 = !DILocalVariable(arg: 2, scope: !502, type: !6)
// CHECK:STDOUT: !511 = !DILocation(line: 96, column: 13, scope: !502)
// CHECK:STDOUT: !513 = !DILocation(line: 96, column: 5, scope: !502)
// CHECK:STDOUT: !514 = !{!503, !504}
// CHECK:STDOUT: !515 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.c8c7baf2b5955a72", scope: null, file: !298, line: 102, type: !243, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !527)
// CHECK:STDOUT: !516 = !DILocalVariable(arg: 1, scope: !515, type: !6)
// CHECK:STDOUT: !517 = !DILocalVariable(arg: 2, scope: !515, type: !6)
// CHECK:STDOUT: !524 = !DILocation(line: 104, column: 13, scope: !515)
// CHECK:STDOUT: !526 = !DILocation(line: 104, column: 5, scope: !515)
// CHECK:STDOUT: !527 = !{!516, !517}
// CHECK:STDOUT: !528 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.c8c7baf2b5955a72", scope: null, file: !298, line: 110, type: !243, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !540)
// CHECK:STDOUT: !529 = !DILocalVariable(arg: 1, scope: !528, type: !6)
// CHECK:STDOUT: !530 = !DILocalVariable(arg: 2, scope: !528, type: !6)
// CHECK:STDOUT: !537 = !DILocation(line: 112, column: 13, scope: !528)
// CHECK:STDOUT: !539 = !DILocation(line: 112, column: 5, scope: !528)
// CHECK:STDOUT: !540 = !{!529, !530}
// CHECK:STDOUT: !541 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.61bb220507f77095", scope: null, file: !298, line: 94, type: !261, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !553)
// CHECK:STDOUT: !542 = !DILocalVariable(arg: 1, scope: !541, type: !6)
// CHECK:STDOUT: !543 = !DILocalVariable(arg: 2, scope: !541, type: !7)
// CHECK:STDOUT: !549 = !DILocation(line: 96, column: 29, scope: !541)
// CHECK:STDOUT: !550 = !DILocation(line: 96, column: 13, scope: !541)
// CHECK:STDOUT: !552 = !DILocation(line: 96, column: 5, scope: !541)
// CHECK:STDOUT: !553 = !{!542, !543}
// CHECK:STDOUT: !554 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.61bb220507f77095", scope: null, file: !298, line: 102, type: !266, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !566)
// CHECK:STDOUT: !555 = !DILocalVariable(arg: 1, scope: !554, type: !7)
// CHECK:STDOUT: !556 = !DILocalVariable(arg: 2, scope: !554, type: !6)
// CHECK:STDOUT: !561 = !DILocation(line: 104, column: 14, scope: !554)
// CHECK:STDOUT: !563 = !DILocation(line: 104, column: 13, scope: !554)
// CHECK:STDOUT: !565 = !DILocation(line: 104, column: 5, scope: !554)
// CHECK:STDOUT: !566 = !{!555, !556}
// CHECK:STDOUT: !567 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.61bb220507f77095", scope: null, file: !298, line: 110, type: !261, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !579)
// CHECK:STDOUT: !568 = !DILocalVariable(arg: 1, scope: !567, type: !6)
// CHECK:STDOUT: !569 = !DILocalVariable(arg: 2, scope: !567, type: !7)
// CHECK:STDOUT: !575 = !DILocation(line: 112, column: 29, scope: !567)
// CHECK:STDOUT: !576 = !DILocation(line: 112, column: 13, scope: !567)
// CHECK:STDOUT: !578 = !DILocation(line: 112, column: 5, scope: !567)
// CHECK:STDOUT: !579 = !{!568, !569}
// CHECK:STDOUT: !580 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.4494a819d074492e", scope: null, file: !298, line: 94, type: !281, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !592)
// CHECK:STDOUT: !581 = !DILocalVariable(arg: 1, scope: !580, type: !6)
// CHECK:STDOUT: !582 = !DILocalVariable(arg: 2, scope: !580, type: !8)
// CHECK:STDOUT: !588 = !DILocation(line: 96, column: 29, scope: !580)
// CHECK:STDOUT: !589 = !DILocation(line: 96, column: 13, scope: !580)
// CHECK:STDOUT: !591 = !DILocation(line: 96, column: 5, scope: !580)
// CHECK:STDOUT: !592 = !{!581, !582}
// CHECK:STDOUT: !593 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.4494a819d074492e", scope: null, file: !298, line: 102, type: !286, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !605)
// CHECK:STDOUT: !594 = !DILocalVariable(arg: 1, scope: !593, type: !8)
// CHECK:STDOUT: !595 = !DILocalVariable(arg: 2, scope: !593, type: !6)
// CHECK:STDOUT: !600 = !DILocation(line: 104, column: 14, scope: !593)
// CHECK:STDOUT: !602 = !DILocation(line: 104, column: 13, scope: !593)
// CHECK:STDOUT: !604 = !DILocation(line: 104, column: 5, scope: !593)
// CHECK:STDOUT: !605 = !{!594, !595}
// CHECK:STDOUT: !606 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.4494a819d074492e", scope: null, file: !298, line: 110, type: !281, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !618)
// CHECK:STDOUT: !607 = !DILocalVariable(arg: 1, scope: !606, type: !6)
// CHECK:STDOUT: !608 = !DILocalVariable(arg: 2, scope: !606, type: !8)
// CHECK:STDOUT: !614 = !DILocation(line: 112, column: 29, scope: !606)
// CHECK:STDOUT: !615 = !DILocation(line: 112, column: 13, scope: !606)
// CHECK:STDOUT: !617 = !DILocation(line: 112, column: 5, scope: !606)
// CHECK:STDOUT: !618 = !{!607, !608}
// CHECK:STDOUT: !619 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !620 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.335e322ca15c76a1", scope: null, file: !619, line: 88, type: !51, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !628)
// CHECK:STDOUT: !621 = !DILocalVariable(arg: 1, scope: !620, type: !6)
// CHECK:STDOUT: !626 = !DILocation(line: 89, column: 12, scope: !620)
// CHECK:STDOUT: !627 = !DILocation(line: 89, column: 5, scope: !620)
// CHECK:STDOUT: !628 = !{!621}
// CHECK:STDOUT: !629 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.9b7e0edf5c8a8a84", scope: null, file: !619, line: 88, type: !57, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !637)
// CHECK:STDOUT: !630 = !DILocalVariable(arg: 1, scope: !629, type: !6)
// CHECK:STDOUT: !635 = !DILocation(line: 89, column: 12, scope: !629)
// CHECK:STDOUT: !636 = !DILocation(line: 89, column: 5, scope: !629)
// CHECK:STDOUT: !637 = !{!630}
// CHECK:STDOUT: !638 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.90fccc2ec10748e3", scope: null, file: !619, line: 88, type: !63, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !646)
// CHECK:STDOUT: !639 = !DILocalVariable(arg: 1, scope: !638, type: !6)
// CHECK:STDOUT: !644 = !DILocation(line: 89, column: 12, scope: !638)
// CHECK:STDOUT: !645 = !DILocation(line: 89, column: 5, scope: !638)
// CHECK:STDOUT: !646 = !{!639}
// CHECK:STDOUT: !647 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.de54a264bd4d5c6f", scope: null, file: !619, line: 88, type: !69, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !655)
// CHECK:STDOUT: !648 = !DILocalVariable(arg: 1, scope: !647, type: !6)
// CHECK:STDOUT: !653 = !DILocation(line: 89, column: 12, scope: !647)
// CHECK:STDOUT: !654 = !DILocation(line: 89, column: 5, scope: !647)
// CHECK:STDOUT: !655 = !{!648}
// CHECK:STDOUT: !656 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.e4ef44314f014f83", scope: null, file: !619, line: 72, type: !51, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !666)
// CHECK:STDOUT: !657 = !DILocalVariable(arg: 1, scope: !656, type: !6)
// CHECK:STDOUT: !661 = !DILocation(line: 74, column: 17, scope: !656)
// CHECK:STDOUT: !662 = !{!6, !6}
// CHECK:STDOUT: !663 = !DISubroutineType(types: !662)
// CHECK:STDOUT: !664 = !DILocation(line: 74, column: 12, scope: !656)
// CHECK:STDOUT: !665 = !DILocation(line: 74, column: 5, scope: !656)
// CHECK:STDOUT: !666 = !{!657}
// CHECK:STDOUT: !667 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.62c2f9adcd5bc12d", scope: null, file: !619, line: 72, type: !57, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !675)
// CHECK:STDOUT: !668 = !DILocalVariable(arg: 1, scope: !667, type: !6)
// CHECK:STDOUT: !672 = !DILocation(line: 74, column: 17, scope: !667)
// CHECK:STDOUT: !673 = !DILocation(line: 74, column: 12, scope: !667)
// CHECK:STDOUT: !674 = !DILocation(line: 74, column: 5, scope: !667)
// CHECK:STDOUT: !675 = !{!668}
// CHECK:STDOUT: !676 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.4f3746a8c9265ebe", scope: null, file: !619, line: 80, type: !63, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !684)
// CHECK:STDOUT: !677 = !DILocalVariable(arg: 1, scope: !676, type: !6)
// CHECK:STDOUT: !681 = !DILocation(line: 82, column: 17, scope: !676)
// CHECK:STDOUT: !682 = !DILocation(line: 82, column: 12, scope: !676)
// CHECK:STDOUT: !683 = !DILocation(line: 82, column: 5, scope: !676)
// CHECK:STDOUT: !684 = !{!677}
// CHECK:STDOUT: !685 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.d19fcdc7c1bc8ad3", scope: null, file: !619, line: 80, type: !69, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !693)
// CHECK:STDOUT: !686 = !DILocalVariable(arg: 1, scope: !685, type: !6)
// CHECK:STDOUT: !690 = !DILocation(line: 82, column: 17, scope: !685)
// CHECK:STDOUT: !691 = !DILocation(line: 82, column: 12, scope: !685)
// CHECK:STDOUT: !692 = !DILocation(line: 82, column: 5, scope: !685)
// CHECK:STDOUT: !693 = !{!686}
// CHECK:STDOUT: !694 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", scope: null, file: !619, line: 60, type: !663, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !701)
// CHECK:STDOUT: !695 = !DILocalVariable(arg: 1, scope: !694, type: !6)
// CHECK:STDOUT: !700 = !DILocation(line: 60, column: 29, scope: !694)
// CHECK:STDOUT: !701 = !{!695}
// CHECK:STDOUT:
