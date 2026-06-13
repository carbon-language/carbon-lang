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
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.1 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %char.as.EqWith.impl.Equal.call = icmp eq i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.1, !dbg !67
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.Equal.call), !dbg !68
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.2 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %char.as.EqWith.impl.NotEqual.call = icmp ne i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.2, !dbg !69
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.EqWith.impl.NotEqual.call), !dbg !70
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.3 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Less.call = icmp ult i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.3, !dbg !71
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Less.call), !dbg !72
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.4 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %char.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.4, !dbg !73
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !74
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.5 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %char.as.OrderedWith.impl.Greater.call = icmp ugt i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.5, !dbg !75
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.Greater.call), !dbg !76
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.6 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %char.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 %a, %CharLike.as.ImplicitAs.impl.Convert.call.6, !dbg !77
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %char.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !78
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.7 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.Equal.call = icmp eq i8 %CharLike.as.ImplicitAs.impl.Convert.call.7, %a, !dbg !79
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.EqWith.impl.Equal.call), !dbg !80
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.8 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %T.as_type.as.EqWith.impl.NotEqual.call = icmp ne i8 %CharLike.as.ImplicitAs.impl.Convert.call.8, %a, !dbg !81
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.EqWith.impl.NotEqual.call), !dbg !82
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.9 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Less.call = icmp ult i8 %CharLike.as.ImplicitAs.impl.Convert.call.9, %a, !dbg !83
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.Less.call), !dbg !84
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.10 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call = icmp ule i8 %CharLike.as.ImplicitAs.impl.Convert.call.10, %a, !dbg !85
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.LessOrEquivalent.call), !dbg !86
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.11 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.Greater.call = icmp ugt i8 %CharLike.as.ImplicitAs.impl.Convert.call.11, %a, !dbg !87
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.Greater.call), !dbg !88
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.12 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %b), !dbg !66
// CHECK:STDOUT:   %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call = icmp uge i8 %CharLike.as.ImplicitAs.impl.Convert.call.12, %a, !dbg !89
// CHECK:STDOUT:   call void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %T.as_type.as.OrderedWith.impl.GreaterOrEquivalent.call), !dbg !90
// CHECK:STDOUT:   ret void, !dbg !91
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CSubCharChar.Main(i8 %c, ptr %d) #0 !dbg !92 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call.loc66 = call i32 @"_COp.Char.Core:SubWith.295cf9b677564c5c.Core"(i8 %c, i8 %c), !dbg !96
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.SubWith.impl.Op.call.loc66), !dbg !97
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.1 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %d), !dbg !98
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call.loc67 = call i32 @"_COp.Char.Core:SubWith.7c3e388e0087434a.Core.ab8531ca840d1501"(i8 %c, i8 %CharLike.as.ImplicitAs.impl.Convert.call.1), !dbg !99
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %char.as.SubWith.impl.Op.call.loc67), !dbg !100
// CHECK:STDOUT:   %CharLike.as.ImplicitAs.impl.Convert.call.2 = call i8 @"_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core"(ptr %d), !dbg !98
// CHECK:STDOUT:   %T.as_type.as.SubWith.impl.Op.call = call i32 @"_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.ab8531ca840d1501"(i8 %CharLike.as.ImplicitAs.impl.Convert.call.2, i8 %c), !dbg !101
// CHECK:STDOUT:   call void @_CDiscard.Main.eadca4a969cf6f21(i32 %T.as_type.as.SubWith.impl.Op.call), !dbg !102
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @"_COp.Char.Core:SubWith.295cf9b677564c5c.Core"(i8, i8)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt8.Main(i8 %c, i8 %i) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.852ed75bdc952b41"(i8 %c, i8 %i), !dbg !111
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !112
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.852ed75bdc952b41"(i8 %i, i8 %c), !dbg !113
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !114
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.852ed75bdc952b41"(i8 %c, i8 %i), !dbg !115
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !116
// CHECK:STDOUT:   ret void, !dbg !117
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt16.Main(i8 %c, i16 %i) #0 !dbg !118 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.4de33f895b808f69"(i8 %c, i16 %i), !dbg !125
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !126
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.4de33f895b808f69"(i16 %i, i8 %c), !dbg !127
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !128
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.4de33f895b808f69"(i8 %c, i16 %i), !dbg !129
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !130
// CHECK:STDOUT:   ret void, !dbg !131
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubInt32.Main(i8 %c, i32 %i) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6"(i8 %c, i32 %i), !dbg !139
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !140
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.ae42bf297a1ee9b6"(i32 %i, i8 %c), !dbg !141
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !142
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6"(i8 %c, i32 %i), !dbg !143
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubUInt8.Main(i8 %c, i8 %i) #0 !dbg !146 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f"(i8 %c, i8 %i), !dbg !150
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !151
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.3ff23939bf07ad5f"(i8 %i, i8 %c), !dbg !152
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !153
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f"(i8 %c, i8 %i), !dbg !154
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !155
// CHECK:STDOUT:   ret void, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubUInt16.Main(i8 %c, i16 %i) #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.a374c478e4368fa7"(i8 %c, i16 %i), !dbg !164
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !165
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.a374c478e4368fa7"(i16 %i, i8 %c), !dbg !166
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !167
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.a374c478e4368fa7"(i8 %c, i16 %i), !dbg !168
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !169
// CHECK:STDOUT:   ret void, !dbg !170
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CAddSubUInt32.Main(i8 %c, i32 %i) #0 !dbg !171 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %char.as.AddWith.impl.Op.call = call i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.bca5dfe49de51c76"(i8 %c, i32 %i), !dbg !178
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.AddWith.impl.Op.call), !dbg !179
// CHECK:STDOUT:   %T.as_type.as.AddWith.impl.Op.call = call i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.bca5dfe49de51c76"(i32 %i, i8 %c), !dbg !180
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %T.as_type.as.AddWith.impl.Op.call), !dbg !181
// CHECK:STDOUT:   %char.as.SubWith.impl.Op.call = call i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.bca5dfe49de51c76"(i8 %c, i32 %i), !dbg !182
// CHECK:STDOUT:   call void @_CDiscard.Main.15688e19507865c6(i8 %char.as.SubWith.impl.Op.call), !dbg !183
// CHECK:STDOUT:   ret void, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.15688e19507865c6(i8 %x) #0 !dbg !185 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.dbe0a954f4d47a23"(i8 %self) #0 !dbg !191 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b906c5940adce8f0"(i8 %self), !dbg !197
// CHECK:STDOUT:   ret i16 %1, !dbg !197
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.147e0bd019a31663(i16 %x) #0 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !203
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.691b43fb97812586"(i8 %self) #0 !dbg !204 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dbd6eca082a732b7"(i8 %self), !dbg !209
// CHECK:STDOUT:   ret i32 %1, !dbg !209
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.eadca4a969cf6f21(i32 %x) #0 !dbg !210 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !215
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.48e83ee0f3ef2ee8"(i8 %self) #0 !dbg !216 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dc14f658d346b3e8"(i8 %self), !dbg !221
// CHECK:STDOUT:   ret i16 %1, !dbg !221
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.8da8ddbe24cfd39a"(i8 %self) #0 !dbg !222 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9"(i8 %self), !dbg !227
// CHECK:STDOUT:   ret i32 %1, !dbg !227
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CDiscard.Main.a4c7fafe8a72cac4(i1 %x) #0 !dbg !228 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.Char.Core:SubWith.7c3e388e0087434a.Core.ab8531ca840d1501"(i8 %self, i8 %other) #0 !dbg !234 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %self), !dbg !240
// CHECK:STDOUT:   %2 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %other), !dbg !241
// CHECK:STDOUT:   %3 = sub i32 %1, %2, !dbg !242
// CHECK:STDOUT:   ret i32 %3, !dbg !243
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.ab8531ca840d1501"(i8 %self, i8 %other) #0 !dbg !244 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %self), !dbg !248
// CHECK:STDOUT:   %2 = call i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %other), !dbg !249
// CHECK:STDOUT:   %3 = sub i32 %1, %2, !dbg !250
// CHECK:STDOUT:   ret i32 %3, !dbg !251
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.852ed75bdc952b41"(i8 %self, i8 %other) #0 !dbg !252 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !258
// CHECK:STDOUT:   ret i8 %1, !dbg !259
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.852ed75bdc952b41"(i8 %self, i8 %other) #0 !dbg !260 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !266
// CHECK:STDOUT:   ret i8 %1, !dbg !267
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.852ed75bdc952b41"(i8 %self, i8 %other) #0 !dbg !268 {
// CHECK:STDOUT:   %1 = sub i8 %self, %other, !dbg !272
// CHECK:STDOUT:   ret i8 %1, !dbg !273
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.4de33f895b808f69"(i8 %self, i16 %other) #0 !dbg !274 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !280
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !281
// CHECK:STDOUT:   ret i8 %2, !dbg !282
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.4de33f895b808f69"(i16 %self, i8 %other) #0 !dbg !283 {
// CHECK:STDOUT:   %1 = trunc i16 %self to i8, !dbg !289
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !290
// CHECK:STDOUT:   ret i8 %2, !dbg !291
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.4de33f895b808f69"(i8 %self, i16 %other) #0 !dbg !292 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !296
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !297
// CHECK:STDOUT:   ret i8 %2, !dbg !298
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6"(i8 %self, i32 %other) #0 !dbg !299 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !305
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !306
// CHECK:STDOUT:   ret i8 %2, !dbg !307
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.ae42bf297a1ee9b6"(i32 %self, i8 %other) #0 !dbg !308 {
// CHECK:STDOUT:   %1 = trunc i32 %self to i8, !dbg !314
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !315
// CHECK:STDOUT:   ret i8 %2, !dbg !316
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6"(i8 %self, i32 %other) #0 !dbg !317 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !321
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !322
// CHECK:STDOUT:   ret i8 %2, !dbg !323
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f"(i8 %self, i8 %other) #0 !dbg !324 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !330
// CHECK:STDOUT:   ret i8 %1, !dbg !331
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.3ff23939bf07ad5f"(i8 %self, i8 %other) #0 !dbg !332 {
// CHECK:STDOUT:   %1 = add i8 %self, %other, !dbg !336
// CHECK:STDOUT:   ret i8 %1, !dbg !337
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f"(i8 %self, i8 %other) #0 !dbg !338 {
// CHECK:STDOUT:   %1 = sub i8 %self, %other, !dbg !342
// CHECK:STDOUT:   ret i8 %1, !dbg !343
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.a374c478e4368fa7"(i8 %self, i16 %other) #0 !dbg !344 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !350
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !351
// CHECK:STDOUT:   ret i8 %2, !dbg !352
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.a374c478e4368fa7"(i16 %self, i8 %other) #0 !dbg !353 {
// CHECK:STDOUT:   %1 = trunc i16 %self to i8, !dbg !359
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !360
// CHECK:STDOUT:   ret i8 %2, !dbg !361
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.a374c478e4368fa7"(i8 %self, i16 %other) #0 !dbg !362 {
// CHECK:STDOUT:   %1 = trunc i16 %other to i8, !dbg !366
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !367
// CHECK:STDOUT:   ret i8 %2, !dbg !368
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.bca5dfe49de51c76"(i8 %self, i32 %other) #0 !dbg !369 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !375
// CHECK:STDOUT:   %2 = add i8 %self, %1, !dbg !376
// CHECK:STDOUT:   ret i8 %2, !dbg !377
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.bca5dfe49de51c76"(i32 %self, i8 %other) #0 !dbg !378 {
// CHECK:STDOUT:   %1 = trunc i32 %self to i8, !dbg !384
// CHECK:STDOUT:   %2 = add i8 %1, %other, !dbg !385
// CHECK:STDOUT:   ret i8 %2, !dbg !386
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.bca5dfe49de51c76"(i8 %self, i32 %other) #0 !dbg !387 {
// CHECK:STDOUT:   %1 = trunc i32 %other to i8, !dbg !391
// CHECK:STDOUT:   %2 = sub i8 %self, %1, !dbg !392
// CHECK:STDOUT:   ret i8 %2, !dbg !393
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b906c5940adce8f0"(i8 %self) #0 !dbg !394 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.c3acaf94968d88d2"(i8 %self), !dbg !398
// CHECK:STDOUT:   ret i16 %1, !dbg !399
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dbd6eca082a732b7"(i8 %self) #0 !dbg !400 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.d93106ca8c7912cd"(i8 %self), !dbg !403
// CHECK:STDOUT:   ret i32 %1, !dbg !404
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dc14f658d346b3e8"(i8 %self) #0 !dbg !405 {
// CHECK:STDOUT:   %1 = call i16 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.8d84eea89906cfec"(i8 %self), !dbg !408
// CHECK:STDOUT:   ret i16 %1, !dbg !409
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9"(i8 %self) #0 !dbg !410 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.1d53f86bcab954ee"(i8 %self), !dbg !413
// CHECK:STDOUT:   ret i32 %1, !dbg !414
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee"(i8 %self) #0 !dbg !415 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9"(i8 %self), !dbg !418
// CHECK:STDOUT:   ret i32 %1, !dbg !418
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.c3acaf94968d88d2"(i8 %from) #0 !dbg !419 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !422
// CHECK:STDOUT:   %2 = zext i8 %1 to i16, !dbg !423
// CHECK:STDOUT:   ret i16 %2, !dbg !424
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.d93106ca8c7912cd"(i8 %from) #0 !dbg !425 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !428
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !429
// CHECK:STDOUT:   ret i32 %2, !dbg !430
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i16 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.8d84eea89906cfec"(i8 %from) #0 !dbg !431 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !434
// CHECK:STDOUT:   %2 = zext i8 %1 to i16, !dbg !435
// CHECK:STDOUT:   ret i16 %2, !dbg !436
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.1d53f86bcab954ee"(i8 %from) #0 !dbg !437 {
// CHECK:STDOUT:   %1 = call i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %from), !dbg !440
// CHECK:STDOUT:   %2 = zext i8 %1 to i32, !dbg !441
// CHECK:STDOUT:   ret i32 %2, !dbg !442
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i8 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8"(i8 %self) #0 !dbg !443 {
// CHECK:STDOUT:   ret i8 %self, !dbg !448
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
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "char.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.CharLike.Main:ImplicitAs.295cf9b677564c5c.Core", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7, !8}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !9 = !{!10}
// CHECK:STDOUT: !10 = !DILocalVariable(arg: 1, scope: !4, type: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 15, column: 39, scope: !4)
// CHECK:STDOUT: !12 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Main", scope: null, file: !3, line: 21, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !15)
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
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "ConvertConstant", linkageName: "_CConvertConstant.Main", scope: null, file: !3, line: 31, type: !30, spFlags: DISPFlagDefinition, unit: !2)
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
// CHECK:STDOUT: !43 = distinct !DISubprogram(name: "CompareCharChar", linkageName: "_CCompareCharChar.Main", scope: null, file: !3, line: 40, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !44)
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
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "CompareCharCharLike", linkageName: "_CCompareCharCharLike.Main", scope: null, file: !3, line: 49, type: !61, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !63)
// CHECK:STDOUT: !61 = !DISubroutineType(types: !62)
// CHECK:STDOUT: !62 = !{null, !7, !8}
// CHECK:STDOUT: !63 = !{!64, !65}
// CHECK:STDOUT: !64 = !DILocalVariable(arg: 1, scope: !60, type: !7)
// CHECK:STDOUT: !65 = !DILocalVariable(arg: 2, scope: !60, type: !8)
// CHECK:STDOUT: !66 = !DILocation(line: 0, scope: !60)
// CHECK:STDOUT: !67 = !DILocation(line: 50, column: 11, scope: !60)
// CHECK:STDOUT: !68 = !DILocation(line: 50, column: 3, scope: !60)
// CHECK:STDOUT: !69 = !DILocation(line: 51, column: 11, scope: !60)
// CHECK:STDOUT: !70 = !DILocation(line: 51, column: 3, scope: !60)
// CHECK:STDOUT: !71 = !DILocation(line: 52, column: 11, scope: !60)
// CHECK:STDOUT: !72 = !DILocation(line: 52, column: 3, scope: !60)
// CHECK:STDOUT: !73 = !DILocation(line: 53, column: 11, scope: !60)
// CHECK:STDOUT: !74 = !DILocation(line: 53, column: 3, scope: !60)
// CHECK:STDOUT: !75 = !DILocation(line: 54, column: 11, scope: !60)
// CHECK:STDOUT: !76 = !DILocation(line: 54, column: 3, scope: !60)
// CHECK:STDOUT: !77 = !DILocation(line: 55, column: 11, scope: !60)
// CHECK:STDOUT: !78 = !DILocation(line: 55, column: 3, scope: !60)
// CHECK:STDOUT: !79 = !DILocation(line: 57, column: 11, scope: !60)
// CHECK:STDOUT: !80 = !DILocation(line: 57, column: 3, scope: !60)
// CHECK:STDOUT: !81 = !DILocation(line: 58, column: 11, scope: !60)
// CHECK:STDOUT: !82 = !DILocation(line: 58, column: 3, scope: !60)
// CHECK:STDOUT: !83 = !DILocation(line: 59, column: 11, scope: !60)
// CHECK:STDOUT: !84 = !DILocation(line: 59, column: 3, scope: !60)
// CHECK:STDOUT: !85 = !DILocation(line: 60, column: 11, scope: !60)
// CHECK:STDOUT: !86 = !DILocation(line: 60, column: 3, scope: !60)
// CHECK:STDOUT: !87 = !DILocation(line: 61, column: 11, scope: !60)
// CHECK:STDOUT: !88 = !DILocation(line: 61, column: 3, scope: !60)
// CHECK:STDOUT: !89 = !DILocation(line: 62, column: 11, scope: !60)
// CHECK:STDOUT: !90 = !DILocation(line: 62, column: 3, scope: !60)
// CHECK:STDOUT: !91 = !DILocation(line: 49, column: 1, scope: !60)
// CHECK:STDOUT: !92 = distinct !DISubprogram(name: "SubCharChar", linkageName: "_CSubCharChar.Main", scope: null, file: !3, line: 65, type: !61, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !93)
// CHECK:STDOUT: !93 = !{!94, !95}
// CHECK:STDOUT: !94 = !DILocalVariable(arg: 1, scope: !92, type: !7)
// CHECK:STDOUT: !95 = !DILocalVariable(arg: 2, scope: !92, type: !8)
// CHECK:STDOUT: !96 = !DILocation(line: 66, column: 11, scope: !92)
// CHECK:STDOUT: !97 = !DILocation(line: 66, column: 3, scope: !92)
// CHECK:STDOUT: !98 = !DILocation(line: 0, scope: !92)
// CHECK:STDOUT: !99 = !DILocation(line: 67, column: 11, scope: !92)
// CHECK:STDOUT: !100 = !DILocation(line: 67, column: 3, scope: !92)
// CHECK:STDOUT: !101 = !DILocation(line: 68, column: 11, scope: !92)
// CHECK:STDOUT: !102 = !DILocation(line: 68, column: 3, scope: !92)
// CHECK:STDOUT: !103 = !DILocation(line: 65, column: 1, scope: !92)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "AddSubInt8", linkageName: "_CAddSubInt8.Main", scope: null, file: !3, line: 71, type: !105, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !108)
// CHECK:STDOUT: !105 = !DISubroutineType(types: !106)
// CHECK:STDOUT: !106 = !{null, !7, !107}
// CHECK:STDOUT: !107 = !DIBasicType(name: "int", size: 8, encoding: DW_ATE_signed)
// CHECK:STDOUT: !108 = !{!109, !110}
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !104, type: !7)
// CHECK:STDOUT: !110 = !DILocalVariable(arg: 2, scope: !104, type: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 72, column: 11, scope: !104)
// CHECK:STDOUT: !112 = !DILocation(line: 72, column: 3, scope: !104)
// CHECK:STDOUT: !113 = !DILocation(line: 73, column: 11, scope: !104)
// CHECK:STDOUT: !114 = !DILocation(line: 73, column: 3, scope: !104)
// CHECK:STDOUT: !115 = !DILocation(line: 74, column: 11, scope: !104)
// CHECK:STDOUT: !116 = !DILocation(line: 74, column: 3, scope: !104)
// CHECK:STDOUT: !117 = !DILocation(line: 71, column: 1, scope: !104)
// CHECK:STDOUT: !118 = distinct !DISubprogram(name: "AddSubInt16", linkageName: "_CAddSubInt16.Main", scope: null, file: !3, line: 77, type: !119, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !122)
// CHECK:STDOUT: !119 = !DISubroutineType(types: !120)
// CHECK:STDOUT: !120 = !{null, !7, !121}
// CHECK:STDOUT: !121 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !122 = !{!123, !124}
// CHECK:STDOUT: !123 = !DILocalVariable(arg: 1, scope: !118, type: !7)
// CHECK:STDOUT: !124 = !DILocalVariable(arg: 2, scope: !118, type: !121)
// CHECK:STDOUT: !125 = !DILocation(line: 78, column: 11, scope: !118)
// CHECK:STDOUT: !126 = !DILocation(line: 78, column: 3, scope: !118)
// CHECK:STDOUT: !127 = !DILocation(line: 79, column: 11, scope: !118)
// CHECK:STDOUT: !128 = !DILocation(line: 79, column: 3, scope: !118)
// CHECK:STDOUT: !129 = !DILocation(line: 80, column: 11, scope: !118)
// CHECK:STDOUT: !130 = !DILocation(line: 80, column: 3, scope: !118)
// CHECK:STDOUT: !131 = !DILocation(line: 77, column: 1, scope: !118)
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "AddSubInt32", linkageName: "_CAddSubInt32.Main", scope: null, file: !3, line: 83, type: !133, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !136)
// CHECK:STDOUT: !133 = !DISubroutineType(types: !134)
// CHECK:STDOUT: !134 = !{null, !7, !135}
// CHECK:STDOUT: !135 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !136 = !{!137, !138}
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !132, type: !7)
// CHECK:STDOUT: !138 = !DILocalVariable(arg: 2, scope: !132, type: !135)
// CHECK:STDOUT: !139 = !DILocation(line: 84, column: 11, scope: !132)
// CHECK:STDOUT: !140 = !DILocation(line: 84, column: 3, scope: !132)
// CHECK:STDOUT: !141 = !DILocation(line: 85, column: 11, scope: !132)
// CHECK:STDOUT: !142 = !DILocation(line: 85, column: 3, scope: !132)
// CHECK:STDOUT: !143 = !DILocation(line: 86, column: 11, scope: !132)
// CHECK:STDOUT: !144 = !DILocation(line: 86, column: 3, scope: !132)
// CHECK:STDOUT: !145 = !DILocation(line: 83, column: 1, scope: !132)
// CHECK:STDOUT: !146 = distinct !DISubprogram(name: "AddSubUInt8", linkageName: "_CAddSubUInt8.Main", scope: null, file: !3, line: 89, type: !13, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !147)
// CHECK:STDOUT: !147 = !{!148, !149}
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !146, type: !7)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 2, scope: !146, type: !7)
// CHECK:STDOUT: !150 = !DILocation(line: 90, column: 11, scope: !146)
// CHECK:STDOUT: !151 = !DILocation(line: 90, column: 3, scope: !146)
// CHECK:STDOUT: !152 = !DILocation(line: 91, column: 11, scope: !146)
// CHECK:STDOUT: !153 = !DILocation(line: 91, column: 3, scope: !146)
// CHECK:STDOUT: !154 = !DILocation(line: 92, column: 11, scope: !146)
// CHECK:STDOUT: !155 = !DILocation(line: 92, column: 3, scope: !146)
// CHECK:STDOUT: !156 = !DILocation(line: 89, column: 1, scope: !146)
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "AddSubUInt16", linkageName: "_CAddSubUInt16.Main", scope: null, file: !3, line: 95, type: !158, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !161)
// CHECK:STDOUT: !158 = !DISubroutineType(types: !159)
// CHECK:STDOUT: !159 = !{null, !7, !160}
// CHECK:STDOUT: !160 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !161 = !{!162, !163}
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !157, type: !7)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 2, scope: !157, type: !160)
// CHECK:STDOUT: !164 = !DILocation(line: 96, column: 11, scope: !157)
// CHECK:STDOUT: !165 = !DILocation(line: 96, column: 3, scope: !157)
// CHECK:STDOUT: !166 = !DILocation(line: 97, column: 11, scope: !157)
// CHECK:STDOUT: !167 = !DILocation(line: 97, column: 3, scope: !157)
// CHECK:STDOUT: !168 = !DILocation(line: 98, column: 11, scope: !157)
// CHECK:STDOUT: !169 = !DILocation(line: 98, column: 3, scope: !157)
// CHECK:STDOUT: !170 = !DILocation(line: 95, column: 1, scope: !157)
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "AddSubUInt32", linkageName: "_CAddSubUInt32.Main", scope: null, file: !3, line: 101, type: !172, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !175)
// CHECK:STDOUT: !172 = !DISubroutineType(types: !173)
// CHECK:STDOUT: !173 = !{null, !7, !174}
// CHECK:STDOUT: !174 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !175 = !{!176, !177}
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 1, scope: !171, type: !7)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 2, scope: !171, type: !174)
// CHECK:STDOUT: !178 = !DILocation(line: 102, column: 11, scope: !171)
// CHECK:STDOUT: !179 = !DILocation(line: 102, column: 3, scope: !171)
// CHECK:STDOUT: !180 = !DILocation(line: 103, column: 11, scope: !171)
// CHECK:STDOUT: !181 = !DILocation(line: 103, column: 3, scope: !171)
// CHECK:STDOUT: !182 = !DILocation(line: 104, column: 11, scope: !171)
// CHECK:STDOUT: !183 = !DILocation(line: 104, column: 3, scope: !171)
// CHECK:STDOUT: !184 = !DILocation(line: 101, column: 1, scope: !171)
// CHECK:STDOUT: !185 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.15688e19507865c6", scope: null, file: !3, line: 19, type: !186, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !188)
// CHECK:STDOUT: !186 = !DISubroutineType(types: !187)
// CHECK:STDOUT: !187 = !{null, !7}
// CHECK:STDOUT: !188 = !{!189}
// CHECK:STDOUT: !189 = !DILocalVariable(arg: 1, scope: !185, type: !7)
// CHECK:STDOUT: !190 = !DILocation(line: 19, column: 1, scope: !185)
// CHECK:STDOUT: !191 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.dbe0a954f4d47a23", scope: null, file: !192, line: 42, type: !193, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !195)
// CHECK:STDOUT: !192 = !DIFile(filename: "{{.*}}/prelude/types/char.carbon", directory: "")
// CHECK:STDOUT: !193 = !DISubroutineType(types: !194)
// CHECK:STDOUT: !194 = !{!160, !7}
// CHECK:STDOUT: !195 = !{!196}
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 1, scope: !191, type: !7)
// CHECK:STDOUT: !197 = !DILocation(line: 43, column: 5, scope: !191)
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.147e0bd019a31663", scope: null, file: !3, line: 19, type: !199, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !201)
// CHECK:STDOUT: !199 = !DISubroutineType(types: !200)
// CHECK:STDOUT: !200 = !{null, !160}
// CHECK:STDOUT: !201 = !{!202}
// CHECK:STDOUT: !202 = !DILocalVariable(arg: 1, scope: !198, type: !160)
// CHECK:STDOUT: !203 = !DILocation(line: 19, column: 1, scope: !198)
// CHECK:STDOUT: !204 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.691b43fb97812586", scope: null, file: !192, line: 42, type: !205, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !207)
// CHECK:STDOUT: !205 = !DISubroutineType(types: !206)
// CHECK:STDOUT: !206 = !{!174, !7}
// CHECK:STDOUT: !207 = !{!208}
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 1, scope: !204, type: !7)
// CHECK:STDOUT: !209 = !DILocation(line: 43, column: 5, scope: !204)
// CHECK:STDOUT: !210 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.eadca4a969cf6f21", scope: null, file: !3, line: 19, type: !211, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !213)
// CHECK:STDOUT: !211 = !DISubroutineType(types: !212)
// CHECK:STDOUT: !212 = !{null, !174}
// CHECK:STDOUT: !213 = !{!214}
// CHECK:STDOUT: !214 = !DILocalVariable(arg: 1, scope: !210, type: !174)
// CHECK:STDOUT: !215 = !DILocation(line: 19, column: 1, scope: !210)
// CHECK:STDOUT: !216 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.48e83ee0f3ef2ee8", scope: null, file: !192, line: 42, type: !217, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !219)
// CHECK:STDOUT: !217 = !DISubroutineType(types: !218)
// CHECK:STDOUT: !218 = !{!121, !7}
// CHECK:STDOUT: !219 = !{!220}
// CHECK:STDOUT: !220 = !DILocalVariable(arg: 1, scope: !216, type: !7)
// CHECK:STDOUT: !221 = !DILocation(line: 43, column: 5, scope: !216)
// CHECK:STDOUT: !222 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.8da8ddbe24cfd39a", scope: null, file: !192, line: 42, type: !223, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !225)
// CHECK:STDOUT: !223 = !DISubroutineType(types: !224)
// CHECK:STDOUT: !224 = !{!135, !7}
// CHECK:STDOUT: !225 = !{!226}
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !222, type: !7)
// CHECK:STDOUT: !227 = !DILocation(line: 43, column: 5, scope: !222)
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "Discard", linkageName: "_CDiscard.Main.a4c7fafe8a72cac4", scope: null, file: !3, line: 19, type: !229, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !231)
// CHECK:STDOUT: !229 = !DISubroutineType(types: !230)
// CHECK:STDOUT: !230 = !{null, !8}
// CHECK:STDOUT: !231 = !{!232}
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 1, scope: !228, type: !8)
// CHECK:STDOUT: !233 = !DILocation(line: 19, column: 1, scope: !228)
// CHECK:STDOUT: !234 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.7c3e388e0087434a.Core.ab8531ca840d1501", scope: null, file: !192, line: 129, type: !235, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !237)
// CHECK:STDOUT: !235 = !DISubroutineType(types: !236)
// CHECK:STDOUT: !236 = !{!135, !7, !7}
// CHECK:STDOUT: !237 = !{!238, !239}
// CHECK:STDOUT: !238 = !DILocalVariable(arg: 1, scope: !234, type: !7)
// CHECK:STDOUT: !239 = !DILocalVariable(arg: 2, scope: !234, type: !7)
// CHECK:STDOUT: !240 = !DILocation(line: 130, column: 13, scope: !234)
// CHECK:STDOUT: !241 = !DILocation(line: 130, column: 29, scope: !234)
// CHECK:STDOUT: !242 = !DILocation(line: 130, column: 12, scope: !234)
// CHECK:STDOUT: !243 = !DILocation(line: 130, column: 5, scope: !234)
// CHECK:STDOUT: !244 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b27934686504f96c:SubWith.295cf9b677564c5c.Core.ab8531ca840d1501", scope: null, file: !192, line: 123, type: !235, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !245)
// CHECK:STDOUT: !245 = !{!246, !247}
// CHECK:STDOUT: !246 = !DILocalVariable(arg: 1, scope: !244, type: !7)
// CHECK:STDOUT: !247 = !DILocalVariable(arg: 2, scope: !244, type: !7)
// CHECK:STDOUT: !248 = !DILocation(line: 124, column: 13, scope: !244)
// CHECK:STDOUT: !249 = !DILocation(line: 124, column: 29, scope: !244)
// CHECK:STDOUT: !250 = !DILocation(line: 124, column: 12, scope: !244)
// CHECK:STDOUT: !251 = !DILocation(line: 124, column: 5, scope: !244)
// CHECK:STDOUT: !252 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.852ed75bdc952b41", scope: null, file: !192, line: 94, type: !253, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !255)
// CHECK:STDOUT: !253 = !DISubroutineType(types: !254)
// CHECK:STDOUT: !254 = !{!7, !7, !107}
// CHECK:STDOUT: !255 = !{!256, !257}
// CHECK:STDOUT: !256 = !DILocalVariable(arg: 1, scope: !252, type: !7)
// CHECK:STDOUT: !257 = !DILocalVariable(arg: 2, scope: !252, type: !107)
// CHECK:STDOUT: !258 = !DILocation(line: 96, column: 13, scope: !252)
// CHECK:STDOUT: !259 = !DILocation(line: 96, column: 5, scope: !252)
// CHECK:STDOUT: !260 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.852ed75bdc952b41", scope: null, file: !192, line: 102, type: !261, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !263)
// CHECK:STDOUT: !261 = !DISubroutineType(types: !262)
// CHECK:STDOUT: !262 = !{!7, !107, !7}
// CHECK:STDOUT: !263 = !{!264, !265}
// CHECK:STDOUT: !264 = !DILocalVariable(arg: 1, scope: !260, type: !107)
// CHECK:STDOUT: !265 = !DILocalVariable(arg: 2, scope: !260, type: !7)
// CHECK:STDOUT: !266 = !DILocation(line: 104, column: 13, scope: !260)
// CHECK:STDOUT: !267 = !DILocation(line: 104, column: 5, scope: !260)
// CHECK:STDOUT: !268 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.852ed75bdc952b41", scope: null, file: !192, line: 110, type: !253, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !269)
// CHECK:STDOUT: !269 = !{!270, !271}
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 1, scope: !268, type: !7)
// CHECK:STDOUT: !271 = !DILocalVariable(arg: 2, scope: !268, type: !107)
// CHECK:STDOUT: !272 = !DILocation(line: 112, column: 13, scope: !268)
// CHECK:STDOUT: !273 = !DILocation(line: 112, column: 5, scope: !268)
// CHECK:STDOUT: !274 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.4de33f895b808f69", scope: null, file: !192, line: 94, type: !275, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !277)
// CHECK:STDOUT: !275 = !DISubroutineType(types: !276)
// CHECK:STDOUT: !276 = !{!7, !7, !121}
// CHECK:STDOUT: !277 = !{!278, !279}
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 1, scope: !274, type: !7)
// CHECK:STDOUT: !279 = !DILocalVariable(arg: 2, scope: !274, type: !121)
// CHECK:STDOUT: !280 = !DILocation(line: 96, column: 29, scope: !274)
// CHECK:STDOUT: !281 = !DILocation(line: 96, column: 13, scope: !274)
// CHECK:STDOUT: !282 = !DILocation(line: 96, column: 5, scope: !274)
// CHECK:STDOUT: !283 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.4de33f895b808f69", scope: null, file: !192, line: 102, type: !284, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !286)
// CHECK:STDOUT: !284 = !DISubroutineType(types: !285)
// CHECK:STDOUT: !285 = !{!7, !121, !7}
// CHECK:STDOUT: !286 = !{!287, !288}
// CHECK:STDOUT: !287 = !DILocalVariable(arg: 1, scope: !283, type: !121)
// CHECK:STDOUT: !288 = !DILocalVariable(arg: 2, scope: !283, type: !7)
// CHECK:STDOUT: !289 = !DILocation(line: 104, column: 14, scope: !283)
// CHECK:STDOUT: !290 = !DILocation(line: 104, column: 13, scope: !283)
// CHECK:STDOUT: !291 = !DILocation(line: 104, column: 5, scope: !283)
// CHECK:STDOUT: !292 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.4de33f895b808f69", scope: null, file: !192, line: 110, type: !275, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !293)
// CHECK:STDOUT: !293 = !{!294, !295}
// CHECK:STDOUT: !294 = !DILocalVariable(arg: 1, scope: !292, type: !7)
// CHECK:STDOUT: !295 = !DILocalVariable(arg: 2, scope: !292, type: !121)
// CHECK:STDOUT: !296 = !DILocation(line: 112, column: 29, scope: !292)
// CHECK:STDOUT: !297 = !DILocation(line: 112, column: 13, scope: !292)
// CHECK:STDOUT: !298 = !DILocation(line: 112, column: 5, scope: !292)
// CHECK:STDOUT: !299 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6", scope: null, file: !192, line: 94, type: !300, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !302)
// CHECK:STDOUT: !300 = !DISubroutineType(types: !301)
// CHECK:STDOUT: !301 = !{!7, !7, !135}
// CHECK:STDOUT: !302 = !{!303, !304}
// CHECK:STDOUT: !303 = !DILocalVariable(arg: 1, scope: !299, type: !7)
// CHECK:STDOUT: !304 = !DILocalVariable(arg: 2, scope: !299, type: !135)
// CHECK:STDOUT: !305 = !DILocation(line: 96, column: 29, scope: !299)
// CHECK:STDOUT: !306 = !DILocation(line: 96, column: 13, scope: !299)
// CHECK:STDOUT: !307 = !DILocation(line: 96, column: 5, scope: !299)
// CHECK:STDOUT: !308 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.ae42bf297a1ee9b6", scope: null, file: !192, line: 102, type: !309, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !311)
// CHECK:STDOUT: !309 = !DISubroutineType(types: !310)
// CHECK:STDOUT: !310 = !{!7, !135, !7}
// CHECK:STDOUT: !311 = !{!312, !313}
// CHECK:STDOUT: !312 = !DILocalVariable(arg: 1, scope: !308, type: !135)
// CHECK:STDOUT: !313 = !DILocalVariable(arg: 2, scope: !308, type: !7)
// CHECK:STDOUT: !314 = !DILocation(line: 104, column: 14, scope: !308)
// CHECK:STDOUT: !315 = !DILocation(line: 104, column: 13, scope: !308)
// CHECK:STDOUT: !316 = !DILocation(line: 104, column: 5, scope: !308)
// CHECK:STDOUT: !317 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.ae42bf297a1ee9b6", scope: null, file: !192, line: 110, type: !300, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !318)
// CHECK:STDOUT: !318 = !{!319, !320}
// CHECK:STDOUT: !319 = !DILocalVariable(arg: 1, scope: !317, type: !7)
// CHECK:STDOUT: !320 = !DILocalVariable(arg: 2, scope: !317, type: !135)
// CHECK:STDOUT: !321 = !DILocation(line: 112, column: 29, scope: !317)
// CHECK:STDOUT: !322 = !DILocation(line: 112, column: 13, scope: !317)
// CHECK:STDOUT: !323 = !DILocation(line: 112, column: 5, scope: !317)
// CHECK:STDOUT: !324 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f", scope: null, file: !192, line: 94, type: !325, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !327)
// CHECK:STDOUT: !325 = !DISubroutineType(types: !326)
// CHECK:STDOUT: !326 = !{!7, !7, !7}
// CHECK:STDOUT: !327 = !{!328, !329}
// CHECK:STDOUT: !328 = !DILocalVariable(arg: 1, scope: !324, type: !7)
// CHECK:STDOUT: !329 = !DILocalVariable(arg: 2, scope: !324, type: !7)
// CHECK:STDOUT: !330 = !DILocation(line: 96, column: 13, scope: !324)
// CHECK:STDOUT: !331 = !DILocation(line: 96, column: 5, scope: !324)
// CHECK:STDOUT: !332 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.3ff23939bf07ad5f", scope: null, file: !192, line: 102, type: !325, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !333)
// CHECK:STDOUT: !333 = !{!334, !335}
// CHECK:STDOUT: !334 = !DILocalVariable(arg: 1, scope: !332, type: !7)
// CHECK:STDOUT: !335 = !DILocalVariable(arg: 2, scope: !332, type: !7)
// CHECK:STDOUT: !336 = !DILocation(line: 104, column: 13, scope: !332)
// CHECK:STDOUT: !337 = !DILocation(line: 104, column: 5, scope: !332)
// CHECK:STDOUT: !338 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.3ff23939bf07ad5f", scope: null, file: !192, line: 110, type: !325, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !339)
// CHECK:STDOUT: !339 = !{!340, !341}
// CHECK:STDOUT: !340 = !DILocalVariable(arg: 1, scope: !338, type: !7)
// CHECK:STDOUT: !341 = !DILocalVariable(arg: 2, scope: !338, type: !7)
// CHECK:STDOUT: !342 = !DILocation(line: 112, column: 13, scope: !338)
// CHECK:STDOUT: !343 = !DILocation(line: 112, column: 5, scope: !338)
// CHECK:STDOUT: !344 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.a374c478e4368fa7", scope: null, file: !192, line: 94, type: !345, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !347)
// CHECK:STDOUT: !345 = !DISubroutineType(types: !346)
// CHECK:STDOUT: !346 = !{!7, !7, !160}
// CHECK:STDOUT: !347 = !{!348, !349}
// CHECK:STDOUT: !348 = !DILocalVariable(arg: 1, scope: !344, type: !7)
// CHECK:STDOUT: !349 = !DILocalVariable(arg: 2, scope: !344, type: !160)
// CHECK:STDOUT: !350 = !DILocation(line: 96, column: 29, scope: !344)
// CHECK:STDOUT: !351 = !DILocation(line: 96, column: 13, scope: !344)
// CHECK:STDOUT: !352 = !DILocation(line: 96, column: 5, scope: !344)
// CHECK:STDOUT: !353 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.a374c478e4368fa7", scope: null, file: !192, line: 102, type: !354, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !356)
// CHECK:STDOUT: !354 = !DISubroutineType(types: !355)
// CHECK:STDOUT: !355 = !{!7, !160, !7}
// CHECK:STDOUT: !356 = !{!357, !358}
// CHECK:STDOUT: !357 = !DILocalVariable(arg: 1, scope: !353, type: !160)
// CHECK:STDOUT: !358 = !DILocalVariable(arg: 2, scope: !353, type: !7)
// CHECK:STDOUT: !359 = !DILocation(line: 104, column: 14, scope: !353)
// CHECK:STDOUT: !360 = !DILocation(line: 104, column: 13, scope: !353)
// CHECK:STDOUT: !361 = !DILocation(line: 104, column: 5, scope: !353)
// CHECK:STDOUT: !362 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.a374c478e4368fa7", scope: null, file: !192, line: 110, type: !345, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !363)
// CHECK:STDOUT: !363 = !{!364, !365}
// CHECK:STDOUT: !364 = !DILocalVariable(arg: 1, scope: !362, type: !7)
// CHECK:STDOUT: !365 = !DILocalVariable(arg: 2, scope: !362, type: !160)
// CHECK:STDOUT: !366 = !DILocation(line: 112, column: 29, scope: !362)
// CHECK:STDOUT: !367 = !DILocation(line: 112, column: 13, scope: !362)
// CHECK:STDOUT: !368 = !DILocation(line: 112, column: 5, scope: !362)
// CHECK:STDOUT: !369 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:AddWith.d9344a06b22b3b94.Core.bca5dfe49de51c76", scope: null, file: !192, line: 94, type: !370, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !372)
// CHECK:STDOUT: !370 = !DISubroutineType(types: !371)
// CHECK:STDOUT: !371 = !{!7, !7, !174}
// CHECK:STDOUT: !372 = !{!373, !374}
// CHECK:STDOUT: !373 = !DILocalVariable(arg: 1, scope: !369, type: !7)
// CHECK:STDOUT: !374 = !DILocalVariable(arg: 2, scope: !369, type: !174)
// CHECK:STDOUT: !375 = !DILocation(line: 96, column: 29, scope: !369)
// CHECK:STDOUT: !376 = !DILocation(line: 96, column: 13, scope: !369)
// CHECK:STDOUT: !377 = !DILocation(line: 96, column: 5, scope: !369)
// CHECK:STDOUT: !378 = distinct !DISubprogram(name: "Op", linkageName: "_COp.fdd9948a9ef8c6e5:AddWith.295cf9b677564c5c.Core.bca5dfe49de51c76", scope: null, file: !192, line: 102, type: !379, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !381)
// CHECK:STDOUT: !379 = !DISubroutineType(types: !380)
// CHECK:STDOUT: !380 = !{!7, !174, !7}
// CHECK:STDOUT: !381 = !{!382, !383}
// CHECK:STDOUT: !382 = !DILocalVariable(arg: 1, scope: !378, type: !174)
// CHECK:STDOUT: !383 = !DILocalVariable(arg: 2, scope: !378, type: !7)
// CHECK:STDOUT: !384 = !DILocation(line: 104, column: 14, scope: !378)
// CHECK:STDOUT: !385 = !DILocation(line: 104, column: 13, scope: !378)
// CHECK:STDOUT: !386 = !DILocation(line: 104, column: 5, scope: !378)
// CHECK:STDOUT: !387 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Char.Core:SubWith.d9344a06b22b3b94.Core.bca5dfe49de51c76", scope: null, file: !192, line: 110, type: !370, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !388)
// CHECK:STDOUT: !388 = !{!389, !390}
// CHECK:STDOUT: !389 = !DILocalVariable(arg: 1, scope: !387, type: !7)
// CHECK:STDOUT: !390 = !DILocalVariable(arg: 2, scope: !387, type: !174)
// CHECK:STDOUT: !391 = !DILocation(line: 112, column: 29, scope: !387)
// CHECK:STDOUT: !392 = !DILocation(line: 112, column: 13, scope: !387)
// CHECK:STDOUT: !393 = !DILocation(line: 112, column: 5, scope: !387)
// CHECK:STDOUT: !394 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b906c5940adce8f0", scope: null, file: !395, line: 88, type: !193, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !396)
// CHECK:STDOUT: !395 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !396 = !{!397}
// CHECK:STDOUT: !397 = !DILocalVariable(arg: 1, scope: !394, type: !7)
// CHECK:STDOUT: !398 = !DILocation(line: 89, column: 12, scope: !394)
// CHECK:STDOUT: !399 = !DILocation(line: 89, column: 5, scope: !394)
// CHECK:STDOUT: !400 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dbd6eca082a732b7", scope: null, file: !395, line: 88, type: !205, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !401)
// CHECK:STDOUT: !401 = !{!402}
// CHECK:STDOUT: !402 = !DILocalVariable(arg: 1, scope: !400, type: !7)
// CHECK:STDOUT: !403 = !DILocation(line: 89, column: 12, scope: !400)
// CHECK:STDOUT: !404 = !DILocation(line: 89, column: 5, scope: !400)
// CHECK:STDOUT: !405 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.dc14f658d346b3e8", scope: null, file: !395, line: 88, type: !217, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !406)
// CHECK:STDOUT: !406 = !{!407}
// CHECK:STDOUT: !407 = !DILocalVariable(arg: 1, scope: !405, type: !7)
// CHECK:STDOUT: !408 = !DILocation(line: 89, column: 12, scope: !405)
// CHECK:STDOUT: !409 = !DILocation(line: 89, column: 5, scope: !405)
// CHECK:STDOUT: !410 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.f48df0d8add56b17.Core.b26071071fee62b9", scope: null, file: !395, line: 88, type: !223, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !411)
// CHECK:STDOUT: !411 = !{!412}
// CHECK:STDOUT: !412 = !DILocalVariable(arg: 1, scope: !410, type: !7)
// CHECK:STDOUT: !413 = !DILocation(line: 89, column: 12, scope: !410)
// CHECK:STDOUT: !414 = !DILocation(line: 89, column: 5, scope: !410)
// CHECK:STDOUT: !415 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Char.Core:As.5d7343568c0b5715.Core.db6d5218df1134ee", scope: null, file: !192, line: 42, type: !223, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !416)
// CHECK:STDOUT: !416 = !{!417}
// CHECK:STDOUT: !417 = !DILocalVariable(arg: 1, scope: !415, type: !7)
// CHECK:STDOUT: !418 = !DILocation(line: 43, column: 5, scope: !415)
// CHECK:STDOUT: !419 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.c3acaf94968d88d2", scope: null, file: !395, line: 72, type: !193, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !420)
// CHECK:STDOUT: !420 = !{!421}
// CHECK:STDOUT: !421 = !DILocalVariable(arg: 1, scope: !419, type: !7)
// CHECK:STDOUT: !422 = !DILocation(line: 74, column: 17, scope: !419)
// CHECK:STDOUT: !423 = !DILocation(line: 74, column: 12, scope: !419)
// CHECK:STDOUT: !424 = !DILocation(line: 74, column: 5, scope: !419)
// CHECK:STDOUT: !425 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.a1a82301830d1e08.Core.d93106ca8c7912cd", scope: null, file: !395, line: 72, type: !205, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !426)
// CHECK:STDOUT: !426 = !{!427}
// CHECK:STDOUT: !427 = !DILocalVariable(arg: 1, scope: !425, type: !7)
// CHECK:STDOUT: !428 = !DILocation(line: 74, column: 17, scope: !425)
// CHECK:STDOUT: !429 = !DILocation(line: 74, column: 12, scope: !425)
// CHECK:STDOUT: !430 = !DILocation(line: 74, column: 5, scope: !425)
// CHECK:STDOUT: !431 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.8d84eea89906cfec", scope: null, file: !395, line: 80, type: !217, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !432)
// CHECK:STDOUT: !432 = !{!433}
// CHECK:STDOUT: !433 = !DILocalVariable(arg: 1, scope: !431, type: !7)
// CHECK:STDOUT: !434 = !DILocation(line: 82, column: 17, scope: !431)
// CHECK:STDOUT: !435 = !DILocation(line: 82, column: 12, scope: !431)
// CHECK:STDOUT: !436 = !DILocation(line: 82, column: 5, scope: !431)
// CHECK:STDOUT: !437 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.Int.9452f4c51951679b.Core:FromUInt.94db85caea4a44af.Core.1d53f86bcab954ee", scope: null, file: !395, line: 80, type: !223, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !438)
// CHECK:STDOUT: !438 = !{!439}
// CHECK:STDOUT: !439 = !DILocalVariable(arg: 1, scope: !437, type: !7)
// CHECK:STDOUT: !440 = !DILocation(line: 82, column: 17, scope: !437)
// CHECK:STDOUT: !441 = !DILocation(line: 82, column: 12, scope: !437)
// CHECK:STDOUT: !442 = !DILocation(line: 82, column: 5, scope: !437)
// CHECK:STDOUT: !443 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.ca1ac10d3fc7c1a8", scope: null, file: !395, line: 60, type: !444, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !446)
// CHECK:STDOUT: !444 = !DISubroutineType(types: !445)
// CHECK:STDOUT: !445 = !{!7, !7}
// CHECK:STDOUT: !446 = !{!447}
// CHECK:STDOUT: !447 = !DILocalVariable(arg: 1, scope: !443, type: !7)
// CHECK:STDOUT: !448 = !DILocation(line: 60, column: 29, scope: !443)
