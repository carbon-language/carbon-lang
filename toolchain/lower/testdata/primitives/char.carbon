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

fn Discard[T:! type](unused x: T) {}

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

// CHECK:STDOUT: ; ModuleID = 'char.carbon'
// CHECK:STDOUT: source_filename = "char.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %self) #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret i8 120, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvert.Main(i8 %c, i8 %u) #0 !dbg !12 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %c), !dbg !18
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc23 = call i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.dbe0a954f4d47a23"(i8 %c), !dbg !19
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 %char.as.As.impl.Convert.call.loc23), !dbg !20
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc24 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.691b43fb97812586"(i8 %c), !dbg !21
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.As.impl.Convert.call.loc24), !dbg !22
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc26 = call i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.48e83ee0f3ef2ee8"(i8 %c), !dbg !23
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 %char.as.As.impl.Convert.call.loc26), !dbg !24
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc27 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.8da8ddbe24cfd39a"(i8 %c), !dbg !25
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.As.impl.Convert.call.loc27), !dbg !26
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %u), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertConstant.Main() #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 120), !dbg !32
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc33 = call i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.dbe0a954f4d47a23"(i8 120), !dbg !33
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 %char.as.As.impl.Convert.call.loc33), !dbg !34
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc34 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.691b43fb97812586"(i8 120), !dbg !35
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.As.impl.Convert.call.loc34), !dbg !36
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc35 = call i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.48e83ee0f3ef2ee8"(i8 120), !dbg !37
// CHECK:STDOUT:   call void @_CDiscard.Main.147e0bd019a31663(i16 %char.as.As.impl.Convert.call.loc35), !dbg !38
// CHECK:STDOUT:   %char.as.As.impl.Convert.call.loc36 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.8da8ddbe24cfd39a"(i8 120), !dbg !39
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.As.impl.Convert.call.loc36), !dbg !40
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 123), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompareCharChar.Main(i8 %a, i8 %b) #0 !dbg !43 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.EqWith.impl.Equal.call = icmp eq i8 %a, %b, !dbg !47
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.Equal.call), !dbg !48
// CHECK:STDOUT:   %char.as.EqWith.impl.NotEqual.call = icmp ne i8 %a, %b, !dbg !49
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.NotEqual.call), !dbg !50
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Less.call = icmp ult i8 %a, %b, !dbg !51
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Less.call), !dbg !52
// CHECK:STDOUT:   %char.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 %a, %b, !dbg !53
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !54
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Greater.call = icmp ugt i8 %a, %b, !dbg !55
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Greater.call), !dbg !56
// CHECK:STDOUT:   %char.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 %a, %b, !dbg !57
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !59
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCompareCharCharLike.Main(i8 %a, ptr %b) #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc50 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %char.as.EqWith.impl.Equal.call = icmp eq i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc50, !dbg !67
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.Equal.call), !dbg !68
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc51 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !69
// CHECK:STDOUT:   %char.as.EqWith.impl.NotEqual.call = icmp ne i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc51, !dbg !70
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.NotEqual.call), !dbg !71
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc52 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !72
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Less.call = icmp ult i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc52, !dbg !73
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Less.call), !dbg !74
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc53 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !75
// CHECK:STDOUT:   %char.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc53, !dbg !76
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !77
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc54 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !78
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Greater.call = icmp ugt i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc54, !dbg !79
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Greater.call), !dbg !80
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc55 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !81
// CHECK:STDOUT:   %char.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.loc55, !dbg !82
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !83
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc57 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !84
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call = icmp eq i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc57, %a, !dbg !84
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.EqWith.impl.Equal.call), !dbg !85
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc58 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !86
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call = icmp ne i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc58, %a, !dbg !86
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.EqWith.impl.NotEqual.call), !dbg !87
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc59 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !88
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call = icmp ult i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc59, %a, !dbg !88
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.Less.call), !dbg !89
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc60 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !90
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc60, %a, !dbg !90
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !91
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc61 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !92
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call = icmp ugt i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc61, %a, !dbg !92
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.Greater.call), !dbg !93
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc62 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !94
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc62, %a, !dbg !94
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !95
// CHECK:STDOUT:   ret void, !dbg !96
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CSubCharChar.Main(i8 %c, ptr %d) #0 !dbg !97 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call.loc66 = call i32 @"_COp.Char.Core:SubWith.295cf9b677564c5c.Core"(i8 %c, i8 %c), !dbg !101
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.SubWith.impl.Op.call.loc66), !dbg !102
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc67 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %d), !dbg !103
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call.loc67 = call i32 @"_COp.Char.Core:SubWith.7c3e388e0087434a.Core.ab8531ca840d1501"(i8 %c, i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc67), !dbg !104
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.SubWith.impl.Op.call.loc67), !dbg !105
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.loc68 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %d), !dbg !106
// CHECK:STDOUT:   %T.as_type.as.SubWith.impl.Op.call = call i32 @"_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.ab8531ca840d1501"(i8 %CharLike.as.ImplicitAs.impl.Convert.call.loc68, i8 %c), !dbg !106
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %T.as_type.as.SubWith.impl.Op.call), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !108
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_COp.Char.Core:SubWith.295cf9b677564c5c.Core"(i8, i8)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt8.Main(i8 %c, i8 %i) #0 !dbg !109 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.852ed75bdc952b41"(i8 %c, i8 %i), !dbg !116
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !117
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.852ed75bdc952b41"(i8 %i, i8 %c), !dbg !118
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !119
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.852ed75bdc952b41"(i8 %c, i8 %i), !dbg !120
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt16.Main(i8 %c, i16 %i) #0 !dbg !123 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.4de33f895b808f69"(i8 %c, i16 %i), !dbg !130
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !131
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.4de33f895b808f69"(i16 %i, i8 %c), !dbg !132
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !133
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.4de33f895b808f69"(i8 %c, i16 %i), !dbg !134
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !135
// CHECK:STDOUT:   ret void, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt32.Main(i8 %c, i32 %i) #0 !dbg !137 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6"(i8 %c, i32 %i), !dbg !144
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !145
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.ae42bf297a1ee9b6"(i32 %i, i8 %c), !dbg !146
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !147
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6"(i8 %c, i32 %i), !dbg !148
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !149
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubUInt8.Main(i8 %c, i8 %i) #0 !dbg !151 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f"(i8 %c, i8 %i), !dbg !155
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !156
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.3ff23939bf07ad5f"(i8 %i, i8 %c), !dbg !157
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !158
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f"(i8 %c, i8 %i), !dbg !159
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !160
// CHECK:STDOUT:   ret void, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubUInt16.Main(i8 %c, i16 %i) #0 !dbg !162 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.a374c478e4368fa7"(i8 %c, i16 %i), !dbg !169
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !170
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.a374c478e4368fa7"(i16 %i, i8 %c), !dbg !171
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !172
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.a374c478e4368fa7"(i8 %c, i16 %i), !dbg !173
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !174
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubUInt32.Main(i8 %c, i32 %i) #0 !dbg !176 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.bca5dfe49de51c76"(i8 %c, i32 %i), !dbg !183
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !184
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.bca5dfe49de51c76"(i32 %i, i8 %c), !dbg !185
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !186
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.bca5dfe49de51c76"(i8 %c, i32 %i), !dbg !187
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !188
// CHECK:STDOUT:   ret void, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.15688e19507865c6(i8 %x) #0 !dbg !190 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !195
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.dbe0a954f4d47a23"(i8 %self) #0 !dbg !196 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b906c5940adce8f0"(i8 %self), !dbg !202
// CHECK:STDOUT:   ret i16 %1, !dbg !202
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.147e0bd019a31663(i16 %x) #0 !dbg !203 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.691b43fb97812586"(i8 %self) #0 !dbg !209 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dbd6eca082a732b7"(i8 %self), !dbg !214
// CHECK:STDOUT:   ret i32 %1, !dbg !214
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.eadca4a969cf6f21(i32 %x) #0 !dbg !215 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !220
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.48e83ee0f3ef2ee8"(i8 %self) #0 !dbg !221 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dc14f658d346b3e8"(i8 %self), !dbg !226
// CHECK:STDOUT:   ret i16 %1, !dbg !226
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.8da8ddbe24cfd39a"(i8 %self) #0 !dbg !227 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9"(i8 %self), !dbg !232
// CHECK:STDOUT:   ret i32 %1, !dbg !232
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %x) #0 !dbg !233 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !238
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.Char.Core:SubWith.7c3e388e0087434a.Core.ab8531ca840d1501"(i8 %self, i8 %other) #0 !dbg !239 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %self), !dbg !245
// CHECK:STDOUT:   %2 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %other), !dbg !246
// CHECK:STDOUT:   %3 = sub i32 %1, %2, !dbg !247
// CHECK:STDOUT:   ret i32 %3, !dbg !248
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.ab8531ca840d1501"(i8 %self, i8 %other) #0 !dbg !249 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %self), !dbg !253
// CHECK:STDOUT:   %2 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %other), !dbg !254
// CHECK:STDOUT:   %3 = sub i32 %1, %2, !dbg !255
// CHECK:STDOUT:   ret i32 %3, !dbg !256
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.852ed75bdc952b41"(i8 %self, i8 %other) #0 !dbg !257 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !263
// CHECK:STDOUT:   ret i8 %1, !dbg !264
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.852ed75bdc952b41"(i8 %self, i8 %other) #0 !dbg !265 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !271
// CHECK:STDOUT:   ret i8 %1, !dbg !272
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.852ed75bdc952b41"(i8 %self, i8 %other) #0 !dbg !273 {
// CHECK:STDOUT:   %1 = sub i8 %self, %other, !dbg !277
// CHECK:STDOUT:   ret i8 %1, !dbg !278
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.4de33f895b808f69"(i8 %self, i16 %other) #0 !dbg !279 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !285
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !286
// CHECK:STDOUT:   ret i8 %2, !dbg !287
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.4de33f895b808f69"(i16 %self, i8 %other) #0 !dbg !288 {
// CHECK:STDOUT:   %1 = trunc i16 %self to i8, !dbg !294
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !295
// CHECK:STDOUT:   ret i8 %2, !dbg !296
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.4de33f895b808f69"(i8 %self, i16 %other) #0 !dbg !297 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !301
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !302
// CHECK:STDOUT:   ret i8 %2, !dbg !303
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6"(i8 %self, i32 %other) #0 !dbg !304 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !310
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !311
// CHECK:STDOUT:   ret i8 %2, !dbg !312
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.ae42bf297a1ee9b6"(i32 %self, i8 %other) #0 !dbg !313 {
// CHECK:STDOUT:   %1 = trunc i32 %self to i8, !dbg !319
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !320
// CHECK:STDOUT:   ret i8 %2, !dbg !321
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6"(i8 %self, i32 %other) #0 !dbg !322 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !326
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !327
// CHECK:STDOUT:   ret i8 %2, !dbg !328
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f"(i8 %self, i8 %other) #0 !dbg !329 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !335
// CHECK:STDOUT:   ret i8 %1, !dbg !336
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.3ff23939bf07ad5f"(i8 %self, i8 %other) #0 !dbg !337 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !341
// CHECK:STDOUT:   ret i8 %1, !dbg !342
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f"(i8 %self, i8 %other) #0 !dbg !343 {
// CHECK:STDOUT:   %1 = sub i8 %self, %other, !dbg !347
// CHECK:STDOUT:   ret i8 %1, !dbg !348
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.a374c478e4368fa7"(i8 %self, i16 %other) #0 !dbg !349 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !355
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !356
// CHECK:STDOUT:   ret i8 %2, !dbg !357
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.a374c478e4368fa7"(i16 %self, i8 %other) #0 !dbg !358 {
// CHECK:STDOUT:   %1 = trunc i16 %self to i8, !dbg !364
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !365
// CHECK:STDOUT:   ret i8 %2, !dbg !366
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.a374c478e4368fa7"(i8 %self, i16 %other) #0 !dbg !367 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !371
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !372
// CHECK:STDOUT:   ret i8 %2, !dbg !373
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.bca5dfe49de51c76"(i8 %self, i32 %other) #0 !dbg !374 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !380
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !381
// CHECK:STDOUT:   ret i8 %2, !dbg !382
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.bca5dfe49de51c76"(i32 %self, i8 %other) #0 !dbg !383 {
// CHECK:STDOUT:   %1 = trunc i32 %self to i8, !dbg !389
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !390
// CHECK:STDOUT:   ret i8 %2, !dbg !391
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.bca5dfe49de51c76"(i8 %self, i32 %other) #0 !dbg !392 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !396
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !397
// CHECK:STDOUT:   ret i8 %2, !dbg !398
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b906c5940adce8f0"(i8 %self) #0 !dbg !399 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.c3acaf94968d88d2"(i8 %self), !dbg !403
// CHECK:STDOUT:   ret i16 %1, !dbg !404
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dbd6eca082a732b7"(i8 %self) #0 !dbg !405 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.d93106ca8c7912cd"(i8 %self), !dbg !408
// CHECK:STDOUT:   ret i32 %1, !dbg !409
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dc14f658d346b3e8"(i8 %self) #0 !dbg !410 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.8d84eea89906cfec"(i8 %self), !dbg !413
// CHECK:STDOUT:   ret i16 %1, !dbg !414
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9"(i8 %self) #0 !dbg !415 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.1d53f86bcab954ee"(i8 %self), !dbg !418
// CHECK:STDOUT:   ret i32 %1, !dbg !419
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %self) #0 !dbg !420 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9"(i8 %self), !dbg !423
// CHECK:STDOUT:   ret i32 %1, !dbg !423
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.c3acaf94968d88d2"(i8 %from) #0 !dbg !424 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !427
// CHECK:STDOUT:   %2 = zext i8 %1 to i16, !dbg !428
// CHECK:STDOUT:   ret i16 %2, !dbg !429
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.d93106ca8c7912cd"(i8 %from) #0 !dbg !430 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !433
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !434
// CHECK:STDOUT:   ret i32 %2, !dbg !435
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.8d84eea89906cfec"(i8 %from) #0 !dbg !436 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !439
// CHECK:STDOUT:   %2 = zext i8 %1 to i16, !dbg !440
// CHECK:STDOUT:   ret i16 %2, !dbg !441
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.1d53f86bcab954ee"(i8 %from) #0 !dbg !442 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !445
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !446
// CHECK:STDOUT:   ret i32 %2, !dbg !447
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %self) #0 !dbg !448 {
// CHECK:STDOUT:   ret i8 %self, !dbg !453
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.15688e19507865c6, { 21, 0, 20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.dbe0a954f4d47a23", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.147e0bd019a31663, { 3, 0, 2, 1 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.691b43fb97812586", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.eadca4a969cf6f21, { 6, 0, 5, 1, 2, 3, 4 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.48e83ee0f3ef2ee8", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.8da8ddbe24cfd39a", { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CDiscard.Main.a4c7fafe8a72cac4, { 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!0}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !1 = !DIFile(filename: "char.carbon", directory: "")
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
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !1, line: 21, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !15)
// CHECK:STDOUT: !13 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !14 = !{null, !7, !7}
// CHECK:STDOUT: !15 = !{!16, !17}
// CHECK:STDOUT: !16 = !DILocalVariable(arg: 1, scope: !12, type: !7)
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 2, scope: !12, type: !7)
// CHECK:STDOUT: !18 = !DILocation(line: 22, column: 3, scope: !12)
// CHECK:STDOUT: !19 = !DILocation(line: 23, column: 11, scope: !12)
// CHECK:STDOUT: !20 = !DILocation(line: 23, column: 3, scope: !12)
// CHECK:STDOUT: !21 = !DILocation(line: 24, column: 11, scope: !12)
// CHECK:STDOUT: !22 = !DILocation(line: 24, column: 3, scope: !12)
// CHECK:STDOUT: !23 = !DILocation(line: 26, column: 11, scope: !12)
// CHECK:STDOUT: !24 = !DILocation(line: 26, column: 3, scope: !12)
// CHECK:STDOUT: !25 = !DILocation(line: 27, column: 11, scope: !12)
// CHECK:STDOUT: !26 = !DILocation(line: 27, column: 3, scope: !12)
// CHECK:STDOUT: !27 = !DILocation(line: 28, column: 3, scope: !12)
// CHECK:STDOUT: !28 = !DILocation(line: 21, column: 1, scope: !12)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "ConvertConstant", linkageName: "_CConvertConstant.Main", scope: null, file: !1, line: 31, type: !30, spFlags: DISPFlagDefinition, unit: !0)
// CHECK:STDOUT: !30 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !31 = !{null}
// CHECK:STDOUT: !32 = !DILocation(line: 32, column: 3, scope: !29)
// CHECK:STDOUT: !33 = !DILocation(line: 33, column: 11, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 33, column: 3, scope: !29)
// CHECK:STDOUT: !35 = !DILocation(line: 34, column: 11, scope: !29)
// CHECK:STDOUT: !36 = !DILocation(line: 34, column: 3, scope: !29)
// CHECK:STDOUT: !37 = !DILocation(line: 35, column: 11, scope: !29)
// CHECK:STDOUT: !38 = !DILocation(line: 35, column: 3, scope: !29)
// CHECK:STDOUT: !39 = !DILocation(line: 36, column: 11, scope: !29)
// CHECK:STDOUT: !40 = !DILocation(line: 36, column: 3, scope: !29)
// CHECK:STDOUT: !41 = !DILocation(line: 37, column: 3, scope: !29)
// CHECK:STDOUT: !42 = !DILocation(line: 31, column: 1, scope: !29)
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "CompareCharChar", linkageName: "_CCompareCharChar.Main", scope: null, file: !1, line: 40, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !44)
// CHECK:STDOUT: !44 = !{!45, !46}
// CHECK:STDOUT: !45 = !DILocalVariable(arg: 1, scope: !43, type: !7)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 2, scope: !43, type: !7)
// CHECK:STDOUT: !47 = !DILocation(line: 41, column: 11, scope: !43)
// CHECK:STDOUT: !48 = !DILocation(line: 41, column: 3, scope: !43)
// CHECK:STDOUT: !49 = !DILocation(line: 42, column: 11, scope: !43)
// CHECK:STDOUT: !50 = !DILocation(line: 42, column: 3, scope: !43)
// CHECK:STDOUT: !51 = !DILocation(line: 43, column: 11, scope: !43)
// CHECK:STDOUT: !52 = !DILocation(line: 43, column: 3, scope: !43)
// CHECK:STDOUT: !53 = !DILocation(line: 44, column: 11, scope: !43)
// CHECK:STDOUT: !54 = !DILocation(line: 44, column: 3, scope: !43)
// CHECK:STDOUT: !55 = !DILocation(line: 45, column: 11, scope: !43)
// CHECK:STDOUT: !56 = !DILocation(line: 45, column: 3, scope: !43)
// CHECK:STDOUT: !57 = !DILocation(line: 46, column: 11, scope: !43)
// CHECK:STDOUT: !58 = !DILocation(line: 46, column: 3, scope: !43)
// CHECK:STDOUT: !59 = !DILocation(line: 40, column: 1, scope: !43)
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "CompareCharCharLike", linkageName: "_CCompareCharCharLike.Main", scope: null, file: !1, line: 49, type: !61, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DISubroutineType(types: !62)
// CHECK:STDOUT: !62 = !{null, !7, !8}
// CHECK:STDOUT: !63 = !{!64, !65}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !60, type: !7)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 2, scope: !60, type: !8)
// CHECK:STDOUT: !66 = !DILocation(line: 50, column: 16, scope: !60)
// CHECK:STDOUT: !67 = !DILocation(line: 50, column: 11, scope: !60)
// CHECK:STDOUT: !68 = !DILocation(line: 50, column: 3, scope: !60)
// CHECK:STDOUT: !69 = !DILocation(line: 51, column: 16, scope: !60)
// CHECK:STDOUT: !70 = !DILocation(line: 51, column: 11, scope: !60)
// CHECK:STDOUT: !71 = !DILocation(line: 51, column: 3, scope: !60)
// CHECK:STDOUT: !72 = !DILocation(line: 52, column: 15, scope: !60)
// CHECK:STDOUT: !73 = !DILocation(line: 52, column: 11, scope: !60)
// CHECK:STDOUT: !74 = !DILocation(line: 52, column: 3, scope: !60)
// CHECK:STDOUT: !75 = !DILocation(line: 53, column: 16, scope: !60)
// CHECK:STDOUT: !76 = !DILocation(line: 53, column: 11, scope: !60)
// CHECK:STDOUT: !77 = !DILocation(line: 53, column: 3, scope: !60)
// CHECK:STDOUT: !78 = !DILocation(line: 54, column: 15, scope: !60)
// CHECK:STDOUT: !79 = !DILocation(line: 54, column: 11, scope: !60)
// CHECK:STDOUT: !80 = !DILocation(line: 54, column: 3, scope: !60)
// CHECK:STDOUT: !81 = !DILocation(line: 55, column: 16, scope: !60)
// CHECK:STDOUT: !82 = !DILocation(line: 55, column: 11, scope: !60)
// CHECK:STDOUT: !83 = !DILocation(line: 55, column: 3, scope: !60)
// CHECK:STDOUT: !84 = !DILocation(line: 57, column: 11, scope: !60)
// CHECK:STDOUT: !85 = !DILocation(line: 57, column: 3, scope: !60)
// CHECK:STDOUT: !86 = !DILocation(line: 58, column: 11, scope: !60)
// CHECK:STDOUT: !87 = !DILocation(line: 58, column: 3, scope: !60)
// CHECK:STDOUT: !88 = !DILocation(line: 59, column: 11, scope: !60)
// CHECK:STDOUT: !89 = !DILocation(line: 59, column: 3, scope: !60)
// CHECK:STDOUT: !90 = !DILocation(line: 60, column: 11, scope: !60)
// CHECK:STDOUT: !91 = !DILocation(line: 60, column: 3, scope: !60)
// CHECK:STDOUT: !92 = !DILocation(line: 61, column: 11, scope: !60)
// CHECK:STDOUT: !93 = !DILocation(line: 61, column: 3, scope: !60)
// CHECK:STDOUT: !94 = !DILocation(line: 62, column: 11, scope: !60)
// CHECK:STDOUT: !95 = !DILocation(line: 62, column: 3, scope: !60)
// CHECK:STDOUT: !96 = !DILocation(line: 49, column: 1, scope: !60)
// CHECK:STDOUT: !97 = distinct !DISubprogram(name: "SubCharChar", linkageName: "_CSubCharChar.Main", scope: null, file: !1, line: 65, type: !61, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !98)
// CHECK:STDOUT: !98 = !{!99, !100}
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 1, scope: !97, type: !7)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 2, scope: !97, type: !8)
// CHECK:STDOUT: !101 = !DILocation(line: 66, column: 11, scope: !97)
// CHECK:STDOUT: !102 = !DILocation(line: 66, column: 3, scope: !97)
// CHECK:STDOUT: !103 = !DILocation(line: 67, column: 15, scope: !97)
// CHECK:STDOUT: !104 = !DILocation(line: 67, column: 11, scope: !97)
// CHECK:STDOUT: !105 = !DILocation(line: 67, column: 3, scope: !97)
// CHECK:STDOUT: !106 = !DILocation(line: 68, column: 11, scope: !97)
// CHECK:STDOUT: !107 = !DILocation(line: 68, column: 3, scope: !97)
// CHECK:STDOUT: !108 = !DILocation(line: 65, column: 1, scope: !97)
// CHECK:STDOUT: !109 = distinct !DISubprogram(name: "AddSubInt8", linkageName: "_CAddSubInt8.Main", scope: null, file: !1, line: 71, type: !110, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !113)
// CHECK:STDOUT: !110 = !DISubroutineType(types: !111)
// CHECK:STDOUT: !111 = !{null, !7, !112}
// CHECK:STDOUT: !112 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !113 = !{!114, !115}
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 1, scope: !109, type: !7)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 2, scope: !109, type: !112)
// CHECK:STDOUT: !116 = !DILocation(line: 72, column: 11, scope: !109)
// CHECK:STDOUT: !117 = !DILocation(line: 72, column: 3, scope: !109)
// CHECK:STDOUT: !118 = !DILocation(line: 73, column: 11, scope: !109)
// CHECK:STDOUT: !119 = !DILocation(line: 73, column: 3, scope: !109)
// CHECK:STDOUT: !120 = !DILocation(line: 74, column: 11, scope: !109)
// CHECK:STDOUT: !121 = !DILocation(line: 74, column: 3, scope: !109)
// CHECK:STDOUT: !122 = !DILocation(line: 71, column: 1, scope: !109)
// CHECK:STDOUT: !123 = distinct !DISubprogram(name: "AddSubInt16", linkageName: "_CAddSubInt16.Main", scope: null, file: !1, line: 77, type: !124, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !127)
// CHECK:STDOUT: !124 = !DISubroutineType(types: !125)
// CHECK:STDOUT: !125 = !{null, !7, !126}
// CHECK:STDOUT: !126 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !127 = !{!128, !129}
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 1, scope: !123, type: !7)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 2, scope: !123, type: !126)
// CHECK:STDOUT: !130 = !DILocation(line: 78, column: 11, scope: !123)
// CHECK:STDOUT: !131 = !DILocation(line: 78, column: 3, scope: !123)
// CHECK:STDOUT: !132 = !DILocation(line: 79, column: 11, scope: !123)
// CHECK:STDOUT: !133 = !DILocation(line: 79, column: 3, scope: !123)
// CHECK:STDOUT: !134 = !DILocation(line: 80, column: 11, scope: !123)
// CHECK:STDOUT: !135 = !DILocation(line: 80, column: 3, scope: !123)
// CHECK:STDOUT: !136 = !DILocation(line: 77, column: 1, scope: !123)
// CHECK:STDOUT: !137 = distinct !DISubprogram(name: "AddSubInt32", linkageName: "_CAddSubInt32.Main", scope: null, file: !1, line: 83, type: !138, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !141)
// CHECK:STDOUT: !138 = !DISubroutineType(types: !139)
// CHECK:STDOUT: !139 = !{null, !7, !140}
// CHECK:STDOUT: !140 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !141 = !{!142, !143}
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 1, scope: !137, type: !7)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 2, scope: !137, type: !140)
// CHECK:STDOUT: !144 = !DILocation(line: 84, column: 11, scope: !137)
// CHECK:STDOUT: !145 = !DILocation(line: 84, column: 3, scope: !137)
// CHECK:STDOUT: !146 = !DILocation(line: 85, column: 11, scope: !137)
// CHECK:STDOUT: !147 = !DILocation(line: 85, column: 3, scope: !137)
// CHECK:STDOUT: !148 = !DILocation(line: 86, column: 11, scope: !137)
// CHECK:STDOUT: !149 = !DILocation(line: 86, column: 3, scope: !137)
// CHECK:STDOUT: !150 = !DILocation(line: 83, column: 1, scope: !137)
// CHECK:STDOUT: !151 = distinct !DISubprogram(name: "AddSubUInt8", linkageName: "_CAddSubUInt8.Main", scope: null, file: !1, line: 89, type: !13, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !152)
// CHECK:STDOUT: !152 = !{!153, !154}
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 1, scope: !151, type: !7)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 2, scope: !151, type: !7)
// CHECK:STDOUT: !155 = !DILocation(line: 90, column: 11, scope: !151)
// CHECK:STDOUT: !156 = !DILocation(line: 90, column: 3, scope: !151)
// CHECK:STDOUT: !157 = !DILocation(line: 91, column: 11, scope: !151)
// CHECK:STDOUT: !158 = !DILocation(line: 91, column: 3, scope: !151)
// CHECK:STDOUT: !159 = !DILocation(line: 92, column: 11, scope: !151)
// CHECK:STDOUT: !160 = !DILocation(line: 92, column: 3, scope: !151)
// CHECK:STDOUT: !161 = !DILocation(line: 89, column: 1, scope: !151)
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "AddSubUInt16", linkageName: "_CAddSubUInt16.Main", scope: null, file: !1, line: 95, type: !163, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !166)
// CHECK:STDOUT: !163 = !DISubroutineType(types: !164)
// CHECK:STDOUT: !164 = !{null, !7, !165}
// CHECK:STDOUT: !165 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !166 = !{!167, !168}
// CHECK:STDOUT: !167 = !DILocalVariable(arg: 1, scope: !162, type: !7)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 2, scope: !162, type: !165)
// CHECK:STDOUT: !169 = !DILocation(line: 96, column: 11, scope: !162)
// CHECK:STDOUT: !170 = !DILocation(line: 96, column: 3, scope: !162)
// CHECK:STDOUT: !171 = !DILocation(line: 97, column: 11, scope: !162)
// CHECK:STDOUT: !172 = !DILocation(line: 97, column: 3, scope: !162)
// CHECK:STDOUT: !173 = !DILocation(line: 98, column: 11, scope: !162)
// CHECK:STDOUT: !174 = !DILocation(line: 98, column: 3, scope: !162)
// CHECK:STDOUT: !175 = !DILocation(line: 95, column: 1, scope: !162)
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "AddSubUInt32", linkageName: "_CAddSubUInt32.Main", scope: null, file: !1, line: 101, type: !177, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !180)
// CHECK:STDOUT: !177 = !DISubroutineType(types: !178)
// CHECK:STDOUT: !178 = !{null, !7, !179}
// CHECK:STDOUT: !179 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !180 = !{!181, !182}
// CHECK:STDOUT: !181 = !DILocalVariable(arg: 1, scope: !176, type: !7)
// CHECK:STDOUT: !182 = !DILocalVariable(arg: 2, scope: !176, type: !179)
// CHECK:STDOUT: !183 = !DILocation(line: 102, column: 11, scope: !176)
// CHECK:STDOUT: !184 = !DILocation(line: 102, column: 3, scope: !176)
// CHECK:STDOUT: !185 = !DILocation(line: 103, column: 11, scope: !176)
// CHECK:STDOUT: !186 = !DILocation(line: 103, column: 3, scope: !176)
// CHECK:STDOUT: !187 = !DILocation(line: 104, column: 11, scope: !176)
// CHECK:STDOUT: !188 = !DILocation(line: 104, column: 3, scope: !176)
// CHECK:STDOUT: !189 = !DILocation(line: 101, column: 1, scope: !176)
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.15688e19507865c6", scope: null, file: !1, line: 19, type: !191, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !193)
// CHECK:STDOUT: !191 = !DISubroutineType(types: !192)
// CHECK:STDOUT: !192 = !{null, !7}
// CHECK:STDOUT: !193 = !{!194}
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 1, scope: !190, type: !7)
// CHECK:STDOUT: !195 = !DILocation(line: 19, column: 1, scope: !190)
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.dbe0a954f4d47a23", scope: null, file: !197, line: 42, type: !198, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !200)
// CHECK:STDOUT: !197 = !DIFile(filename: "{{.*}}/prelude/types/char.carbon", directory: "")
// CHECK:STDOUT: !198 = !DISubroutineType(types: !199)
// CHECK:STDOUT: !199 = !{!165, !7}
// CHECK:STDOUT: !200 = !{!201}
// CHECK:STDOUT: !201 = !DILocalVariable(arg: 1, scope: !196, type: !7)
// CHECK:STDOUT: !202 = !DILocation(line: 43, column: 5, scope: !196)
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.147e0bd019a31663", scope: null, file: !1, line: 19, type: !204, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !206)
// CHECK:STDOUT: !204 = !DISubroutineType(types: !205)
// CHECK:STDOUT: !205 = !{null, !165}
// CHECK:STDOUT: !206 = !{!207}
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !203, type: !165)
// CHECK:STDOUT: !208 = !DILocation(line: 19, column: 1, scope: !203)
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.691b43fb97812586", scope: null, file: !197, line: 42, type: !210, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !212)
// CHECK:STDOUT: !210 = !DISubroutineType(types: !211)
// CHECK:STDOUT: !211 = !{!179, !7}
// CHECK:STDOUT: !212 = !{!213}
// CHECK:STDOUT: !213 = !DILocalVariable(arg: 1, scope: !209, type: !7)
// CHECK:STDOUT: !214 = !DILocation(line: 43, column: 5, scope: !209)
// CHECK:STDOUT: !215 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.eadca4a969cf6f21", scope: null, file: !1, line: 19, type: !216, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !218)
// CHECK:STDOUT: !216 = !DISubroutineType(types: !217)
// CHECK:STDOUT: !217 = !{null, !179}
// CHECK:STDOUT: !218 = !{!219}
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !215, type: !179)
// CHECK:STDOUT: !220 = !DILocation(line: 19, column: 1, scope: !215)
// CHECK:STDOUT: !221 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.48e83ee0f3ef2ee8", scope: null, file: !197, line: 42, type: !222, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !224)
// CHECK:STDOUT: !222 = !DISubroutineType(types: !223)
// CHECK:STDOUT: !223 = !{!126, !7}
// CHECK:STDOUT: !224 = !{!225}
// CHECK:STDOUT: !225 = !DILocalVariable(arg: 1, scope: !221, type: !7)
// CHECK:STDOUT: !226 = !DILocation(line: 43, column: 5, scope: !221)
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.8da8ddbe24cfd39a", scope: null, file: !197, line: 42, type: !228, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !230)
// CHECK:STDOUT: !228 = !DISubroutineType(types: !229)
// CHECK:STDOUT: !229 = !{!140, !7}
// CHECK:STDOUT: !230 = !{!231}
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !227, type: !7)
// CHECK:STDOUT: !232 = !DILocation(line: 43, column: 5, scope: !227)
// CHECK:STDOUT: !233 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.a4c7fafe8a72cac4", scope: null, file: !1, line: 19, type: !234, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !236)
// CHECK:STDOUT: !234 = !DISubroutineType(types: !235)
// CHECK:STDOUT: !235 = !{null, !8}
// CHECK:STDOUT: !236 = !{!237}
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 1, scope: !233, type: !8)
// CHECK:STDOUT: !238 = !DILocation(line: 19, column: 1, scope: !233)
// CHECK:STDOUT: !239 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.7c3e388e0087434a.Core.ab8531ca840d1501", scope: null, file: !197, line: 129, type: !240, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !242)
// CHECK:STDOUT: !240 = !DISubroutineType(types: !241)
// CHECK:STDOUT: !241 = !{!140, !7, !7}
// CHECK:STDOUT: !242 = !{!243, !244}
// CHECK:STDOUT: !243 = !DILocalVariable(arg: 1, scope: !239, type: !7)
// CHECK:STDOUT: !244 = !DILocalVariable(arg: 2, scope: !239, type: !7)
// CHECK:STDOUT: !245 = !DILocation(line: 130, column: 13, scope: !239)
// CHECK:STDOUT: !246 = !DILocation(line: 130, column: 29, scope: !239)
// CHECK:STDOUT: !247 = !DILocation(line: 130, column: 12, scope: !239)
// CHECK:STDOUT: !248 = !DILocation(line: 130, column: 5, scope: !239)
// CHECK:STDOUT: !249 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.ab8531ca840d1501", scope: null, file: !197, line: 123, type: !240, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !250)
// CHECK:STDOUT: !250 = !{!251, !252}
// CHECK:STDOUT: !251 = !DILocalVariable(arg: 1, scope: !249, type: !7)
// CHECK:STDOUT: !252 = !DILocalVariable(arg: 2, scope: !249, type: !7)
// CHECK:STDOUT: !253 = !DILocation(line: 124, column: 13, scope: !249)
// CHECK:STDOUT: !254 = !DILocation(line: 124, column: 29, scope: !249)
// CHECK:STDOUT: !255 = !DILocation(line: 124, column: 12, scope: !249)
// CHECK:STDOUT: !256 = !DILocation(line: 124, column: 5, scope: !249)
// CHECK:STDOUT: !257 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.852ed75bdc952b41", scope: null, file: !197, line: 94, type: !258, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !260)
// CHECK:STDOUT: !258 = !DISubroutineType(types: !259)
// CHECK:STDOUT: !259 = !{!7, !7, !112}
// CHECK:STDOUT: !260 = !{!261, !262}
// CHECK:STDOUT: !261 = !DILocalVariable(arg: 1, scope: !257, type: !7)
// CHECK:STDOUT: !262 = !DILocalVariable(arg: 2, scope: !257, type: !112)
// CHECK:STDOUT: !263 = !DILocation(line: 96, column: 13, scope: !257)
// CHECK:STDOUT: !264 = !DILocation(line: 96, column: 5, scope: !257)
// CHECK:STDOUT: !265 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.852ed75bdc952b41", scope: null, file: !197, line: 102, type: !266, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !268)
// CHECK:STDOUT: !266 = !DISubroutineType(types: !267)
// CHECK:STDOUT: !267 = !{!7, !112, !7}
// CHECK:STDOUT: !268 = !{!269, !270}
// CHECK:STDOUT: !269 = !DILocalVariable(arg: 1, scope: !265, type: !112)
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 2, scope: !265, type: !7)
// CHECK:STDOUT: !271 = !DILocation(line: 104, column: 13, scope: !265)
// CHECK:STDOUT: !272 = !DILocation(line: 104, column: 5, scope: !265)
// CHECK:STDOUT: !273 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.852ed75bdc952b41", scope: null, file: !197, line: 110, type: !258, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !274)
// CHECK:STDOUT: !274 = !{!275, !276}
// CHECK:STDOUT: !275 = !DILocalVariable(arg: 1, scope: !273, type: !7)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 2, scope: !273, type: !112)
// CHECK:STDOUT: !277 = !DILocation(line: 112, column: 13, scope: !273)
// CHECK:STDOUT: !278 = !DILocation(line: 112, column: 5, scope: !273)
// CHECK:STDOUT: !279 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.4de33f895b808f69", scope: null, file: !197, line: 94, type: !280, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !282)
// CHECK:STDOUT: !280 = !DISubroutineType(types: !281)
// CHECK:STDOUT: !281 = !{!7, !7, !126}
// CHECK:STDOUT: !282 = !{!283, !284}
// CHECK:STDOUT: !283 = !DILocalVariable(arg: 1, scope: !279, type: !7)
// CHECK:STDOUT: !284 = !DILocalVariable(arg: 2, scope: !279, type: !126)
// CHECK:STDOUT: !285 = !DILocation(line: 96, column: 29, scope: !279)
// CHECK:STDOUT: !286 = !DILocation(line: 96, column: 13, scope: !279)
// CHECK:STDOUT: !287 = !DILocation(line: 96, column: 5, scope: !279)
// CHECK:STDOUT: !288 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.4de33f895b808f69", scope: null, file: !197, line: 102, type: !289, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !291)
// CHECK:STDOUT: !289 = !DISubroutineType(types: !290)
// CHECK:STDOUT: !290 = !{!7, !126, !7}
// CHECK:STDOUT: !291 = !{!292, !293}
// CHECK:STDOUT: !292 = !DILocalVariable(arg: 1, scope: !288, type: !126)
// CHECK:STDOUT: !293 = !DILocalVariable(arg: 2, scope: !288, type: !7)
// CHECK:STDOUT: !294 = !DILocation(line: 104, column: 14, scope: !288)
// CHECK:STDOUT: !295 = !DILocation(line: 104, column: 13, scope: !288)
// CHECK:STDOUT: !296 = !DILocation(line: 104, column: 5, scope: !288)
// CHECK:STDOUT: !297 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.4de33f895b808f69", scope: null, file: !197, line: 110, type: !280, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !298)
// CHECK:STDOUT: !298 = !{!299, !300}
// CHECK:STDOUT: !299 = !DILocalVariable(arg: 1, scope: !297, type: !7)
// CHECK:STDOUT: !300 = !DILocalVariable(arg: 2, scope: !297, type: !126)
// CHECK:STDOUT: !301 = !DILocation(line: 112, column: 29, scope: !297)
// CHECK:STDOUT: !302 = !DILocation(line: 112, column: 13, scope: !297)
// CHECK:STDOUT: !303 = !DILocation(line: 112, column: 5, scope: !297)
// CHECK:STDOUT: !304 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6", scope: null, file: !197, line: 94, type: !305, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !307)
// CHECK:STDOUT: !305 = !DISubroutineType(types: !306)
// CHECK:STDOUT: !306 = !{!7, !7, !140}
// CHECK:STDOUT: !307 = !{!308, !309}
// CHECK:STDOUT: !308 = !DILocalVariable(arg: 1, scope: !304, type: !7)
// CHECK:STDOUT: !309 = !DILocalVariable(arg: 2, scope: !304, type: !140)
// CHECK:STDOUT: !310 = !DILocation(line: 96, column: 29, scope: !304)
// CHECK:STDOUT: !311 = !DILocation(line: 96, column: 13, scope: !304)
// CHECK:STDOUT: !312 = !DILocation(line: 96, column: 5, scope: !304)
// CHECK:STDOUT: !313 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.ae42bf297a1ee9b6", scope: null, file: !197, line: 102, type: !314, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !316)
// CHECK:STDOUT: !314 = !DISubroutineType(types: !315)
// CHECK:STDOUT: !315 = !{!7, !140, !7}
// CHECK:STDOUT: !316 = !{!317, !318}
// CHECK:STDOUT: !317 = !DILocalVariable(arg: 1, scope: !313, type: !140)
// CHECK:STDOUT: !318 = !DILocalVariable(arg: 2, scope: !313, type: !7)
// CHECK:STDOUT: !319 = !DILocation(line: 104, column: 14, scope: !313)
// CHECK:STDOUT: !320 = !DILocation(line: 104, column: 13, scope: !313)
// CHECK:STDOUT: !321 = !DILocation(line: 104, column: 5, scope: !313)
// CHECK:STDOUT: !322 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6", scope: null, file: !197, line: 110, type: !305, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !323)
// CHECK:STDOUT: !323 = !{!324, !325}
// CHECK:STDOUT: !324 = !DILocalVariable(arg: 1, scope: !322, type: !7)
// CHECK:STDOUT: !325 = !DILocalVariable(arg: 2, scope: !322, type: !140)
// CHECK:STDOUT: !326 = !DILocation(line: 112, column: 29, scope: !322)
// CHECK:STDOUT: !327 = !DILocation(line: 112, column: 13, scope: !322)
// CHECK:STDOUT: !328 = !DILocation(line: 112, column: 5, scope: !322)
// CHECK:STDOUT: !329 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f", scope: null, file: !197, line: 94, type: !330, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !332)
// CHECK:STDOUT: !330 = !DISubroutineType(types: !331)
// CHECK:STDOUT: !331 = !{!7, !7, !7}
// CHECK:STDOUT: !332 = !{!333, !334}
// CHECK:STDOUT: !333 = !DILocalVariable(arg: 1, scope: !329, type: !7)
// CHECK:STDOUT: !334 = !DILocalVariable(arg: 2, scope: !329, type: !7)
// CHECK:STDOUT: !335 = !DILocation(line: 96, column: 13, scope: !329)
// CHECK:STDOUT: !336 = !DILocation(line: 96, column: 5, scope: !329)
// CHECK:STDOUT: !337 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.3ff23939bf07ad5f", scope: null, file: !197, line: 102, type: !330, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !338)
// CHECK:STDOUT: !338 = !{!339, !340}
// CHECK:STDOUT: !339 = !DILocalVariable(arg: 1, scope: !337, type: !7)
// CHECK:STDOUT: !340 = !DILocalVariable(arg: 2, scope: !337, type: !7)
// CHECK:STDOUT: !341 = !DILocation(line: 104, column: 13, scope: !337)
// CHECK:STDOUT: !342 = !DILocation(line: 104, column: 5, scope: !337)
// CHECK:STDOUT: !343 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f", scope: null, file: !197, line: 110, type: !330, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !344)
// CHECK:STDOUT: !344 = !{!345, !346}
// CHECK:STDOUT: !345 = !DILocalVariable(arg: 1, scope: !343, type: !7)
// CHECK:STDOUT: !346 = !DILocalVariable(arg: 2, scope: !343, type: !7)
// CHECK:STDOUT: !347 = !DILocation(line: 112, column: 13, scope: !343)
// CHECK:STDOUT: !348 = !DILocation(line: 112, column: 5, scope: !343)
// CHECK:STDOUT: !349 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.a374c478e4368fa7", scope: null, file: !197, line: 94, type: !350, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !352)
// CHECK:STDOUT: !350 = !DISubroutineType(types: !351)
// CHECK:STDOUT: !351 = !{!7, !7, !165}
// CHECK:STDOUT: !352 = !{!353, !354}
// CHECK:STDOUT: !353 = !DILocalVariable(arg: 1, scope: !349, type: !7)
// CHECK:STDOUT: !354 = !DILocalVariable(arg: 2, scope: !349, type: !165)
// CHECK:STDOUT: !355 = !DILocation(line: 96, column: 29, scope: !349)
// CHECK:STDOUT: !356 = !DILocation(line: 96, column: 13, scope: !349)
// CHECK:STDOUT: !357 = !DILocation(line: 96, column: 5, scope: !349)
// CHECK:STDOUT: !358 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.a374c478e4368fa7", scope: null, file: !197, line: 102, type: !359, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !361)
// CHECK:STDOUT: !359 = !DISubroutineType(types: !360)
// CHECK:STDOUT: !360 = !{!7, !165, !7}
// CHECK:STDOUT: !361 = !{!362, !363}
// CHECK:STDOUT: !362 = !DILocalVariable(arg: 1, scope: !358, type: !165)
// CHECK:STDOUT: !363 = !DILocalVariable(arg: 2, scope: !358, type: !7)
// CHECK:STDOUT: !364 = !DILocation(line: 104, column: 14, scope: !358)
// CHECK:STDOUT: !365 = !DILocation(line: 104, column: 13, scope: !358)
// CHECK:STDOUT: !366 = !DILocation(line: 104, column: 5, scope: !358)
// CHECK:STDOUT: !367 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.a374c478e4368fa7", scope: null, file: !197, line: 110, type: !350, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !368)
// CHECK:STDOUT: !368 = !{!369, !370}
// CHECK:STDOUT: !369 = !DILocalVariable(arg: 1, scope: !367, type: !7)
// CHECK:STDOUT: !370 = !DILocalVariable(arg: 2, scope: !367, type: !165)
// CHECK:STDOUT: !371 = !DILocation(line: 112, column: 29, scope: !367)
// CHECK:STDOUT: !372 = !DILocation(line: 112, column: 13, scope: !367)
// CHECK:STDOUT: !373 = !DILocation(line: 112, column: 5, scope: !367)
// CHECK:STDOUT: !374 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.bca5dfe49de51c76", scope: null, file: !197, line: 94, type: !375, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !377)
// CHECK:STDOUT: !375 = !DISubroutineType(types: !376)
// CHECK:STDOUT: !376 = !{!7, !7, !179}
// CHECK:STDOUT: !377 = !{!378, !379}
// CHECK:STDOUT: !378 = !DILocalVariable(arg: 1, scope: !374, type: !7)
// CHECK:STDOUT: !379 = !DILocalVariable(arg: 2, scope: !374, type: !179)
// CHECK:STDOUT: !380 = !DILocation(line: 96, column: 29, scope: !374)
// CHECK:STDOUT: !381 = !DILocation(line: 96, column: 13, scope: !374)
// CHECK:STDOUT: !382 = !DILocation(line: 96, column: 5, scope: !374)
// CHECK:STDOUT: !383 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.bca5dfe49de51c76", scope: null, file: !197, line: 102, type: !384, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !386)
// CHECK:STDOUT: !384 = !DISubroutineType(types: !385)
// CHECK:STDOUT: !385 = !{!7, !179, !7}
// CHECK:STDOUT: !386 = !{!387, !388}
// CHECK:STDOUT: !387 = !DILocalVariable(arg: 1, scope: !383, type: !179)
// CHECK:STDOUT: !388 = !DILocalVariable(arg: 2, scope: !383, type: !7)
// CHECK:STDOUT: !389 = !DILocation(line: 104, column: 14, scope: !383)
// CHECK:STDOUT: !390 = !DILocation(line: 104, column: 13, scope: !383)
// CHECK:STDOUT: !391 = !DILocation(line: 104, column: 5, scope: !383)
// CHECK:STDOUT: !392 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.bca5dfe49de51c76", scope: null, file: !197, line: 110, type: !375, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !393)
// CHECK:STDOUT: !393 = !{!394, !395}
// CHECK:STDOUT: !394 = !DILocalVariable(arg: 1, scope: !392, type: !7)
// CHECK:STDOUT: !395 = !DILocalVariable(arg: 2, scope: !392, type: !179)
// CHECK:STDOUT: !396 = !DILocation(line: 112, column: 29, scope: !392)
// CHECK:STDOUT: !397 = !DILocation(line: 112, column: 13, scope: !392)
// CHECK:STDOUT: !398 = !DILocation(line: 112, column: 5, scope: !392)
// CHECK:STDOUT: !399 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b906c5940adce8f0", scope: null, file: !400, line: 88, type: !198, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !401)
// CHECK:STDOUT: !400 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !401 = !{!402}
// CHECK:STDOUT: !402 = !DILocalVariable(arg: 1, scope: !399, type: !7)
// CHECK:STDOUT: !403 = !DILocation(line: 89, column: 12, scope: !399)
// CHECK:STDOUT: !404 = !DILocation(line: 89, column: 5, scope: !399)
// CHECK:STDOUT: !405 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dbd6eca082a732b7", scope: null, file: !400, line: 88, type: !210, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !406)
// CHECK:STDOUT: !406 = !{!407}
// CHECK:STDOUT: !407 = !DILocalVariable(arg: 1, scope: !405, type: !7)
// CHECK:STDOUT: !408 = !DILocation(line: 89, column: 12, scope: !405)
// CHECK:STDOUT: !409 = !DILocation(line: 89, column: 5, scope: !405)
// CHECK:STDOUT: !410 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dc14f658d346b3e8", scope: null, file: !400, line: 88, type: !222, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !411)
// CHECK:STDOUT: !411 = !{!412}
// CHECK:STDOUT: !412 = !DILocalVariable(arg: 1, scope: !410, type: !7)
// CHECK:STDOUT: !413 = !DILocation(line: 89, column: 12, scope: !410)
// CHECK:STDOUT: !414 = !DILocation(line: 89, column: 5, scope: !410)
// CHECK:STDOUT: !415 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9", scope: null, file: !400, line: 88, type: !228, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !416)
// CHECK:STDOUT: !416 = !{!417}
// CHECK:STDOUT: !417 = !DILocalVariable(arg: 1, scope: !415, type: !7)
// CHECK:STDOUT: !418 = !DILocation(line: 89, column: 12, scope: !415)
// CHECK:STDOUT: !419 = !DILocation(line: 89, column: 5, scope: !415)
// CHECK:STDOUT: !420 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee", scope: null, file: !197, line: 42, type: !228, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !421)
// CHECK:STDOUT: !421 = !{!422}
// CHECK:STDOUT: !422 = !DILocalVariable(arg: 1, scope: !420, type: !7)
// CHECK:STDOUT: !423 = !DILocation(line: 43, column: 5, scope: !420)
// CHECK:STDOUT: !424 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.c3acaf94968d88d2", scope: null, file: !400, line: 72, type: !198, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !425)
// CHECK:STDOUT: !425 = !{!426}
// CHECK:STDOUT: !426 = !DILocalVariable(arg: 1, scope: !424, type: !7)
// CHECK:STDOUT: !427 = !DILocation(line: 74, column: 17, scope: !424)
// CHECK:STDOUT: !428 = !DILocation(line: 74, column: 12, scope: !424)
// CHECK:STDOUT: !429 = !DILocation(line: 74, column: 5, scope: !424)
// CHECK:STDOUT: !430 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.d93106ca8c7912cd", scope: null, file: !400, line: 72, type: !210, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !431)
// CHECK:STDOUT: !431 = !{!432}
// CHECK:STDOUT: !432 = !DILocalVariable(arg: 1, scope: !430, type: !7)
// CHECK:STDOUT: !433 = !DILocation(line: 74, column: 17, scope: !430)
// CHECK:STDOUT: !434 = !DILocation(line: 74, column: 12, scope: !430)
// CHECK:STDOUT: !435 = !DILocation(line: 74, column: 5, scope: !430)
// CHECK:STDOUT: !436 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.8d84eea89906cfec", scope: null, file: !400, line: 80, type: !222, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !437)
// CHECK:STDOUT: !437 = !{!438}
// CHECK:STDOUT: !438 = !DILocalVariable(arg: 1, scope: !436, type: !7)
// CHECK:STDOUT: !439 = !DILocation(line: 82, column: 17, scope: !436)
// CHECK:STDOUT: !440 = !DILocation(line: 82, column: 12, scope: !436)
// CHECK:STDOUT: !441 = !DILocation(line: 82, column: 5, scope: !436)
// CHECK:STDOUT: !442 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.1d53f86bcab954ee", scope: null, file: !400, line: 80, type: !228, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !443)
// CHECK:STDOUT: !443 = !{!444}
// CHECK:STDOUT: !444 = !DILocalVariable(arg: 1, scope: !442, type: !7)
// CHECK:STDOUT: !445 = !DILocation(line: 82, column: 17, scope: !442)
// CHECK:STDOUT: !446 = !DILocation(line: 82, column: 12, scope: !442)
// CHECK:STDOUT: !447 = !DILocation(line: 82, column: 5, scope: !442)
// CHECK:STDOUT: !448 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", scope: null, file: !400, line: 60, type: !449, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !451)
// CHECK:STDOUT: !449 = !DISubroutineType(types: !450)
// CHECK:STDOUT: !450 = !{!7, !7}
// CHECK:STDOUT: !451 = !{!452}
// CHECK:STDOUT: !452 = !DILocalVariable(arg: 1, scope: !448, type: !7)
// CHECK:STDOUT: !453 = !DILocation(line: 60, column: 29, scope: !448)
