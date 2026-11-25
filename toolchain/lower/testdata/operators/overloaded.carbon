// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/operators/overloaded.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/operators/overloaded.carbon

class Number {
  var is_positive: bool;
}

impl Number as Core.Negate where .Result = Number {
  fn Op(self: Number) -> Number {
    return {.is_positive = not self.is_positive};
  }
}
impl Number as Core.MulWith(Number) where .Result = Number {
  fn Op(self: Number, other: Number) -> Number {
    return {.is_positive = (self.is_positive and other.is_positive) or
                            (not self.is_positive and not other.is_positive)};
  }
}
impl Number as Core.Div {
  fn Op(self: Number, other: Number) -> Number {
    return self * other;
  }
}
impl Number as Core.MulAssign {
  fn Op(ref self: Number, other: Number) {
    self.is_positive = (self.is_positive and other.is_positive) or
                       (not self.is_positive and not other.is_positive);
  }
}

fn Calculate(a: Number, b: Number) -> Number {
  return -a * b;
}

fn CalculateDiv(a: Number, b: Number) -> Number {
  return a / b;
}

fn CalculateMulAssign(ref a: Number, b: Number) {
  a *= b;
}

fn GenericMul[T: Core.Mul](a: T, b: T) -> T {
  return a * b;
}

fn GenericDiv[T: Core.Div](a: T, b: T) -> T {
  return a / b;
}

fn GenericMulAssign[T: Core.MulAssign](ref a: T, b: T) {
  a *= b;
}

fn GenericArithmetic[
    T: Core.Copy & Core.Destroy & Core.Add & Core.Sub & Core.Mul & Core.Div &
       Core.Mod & Core.AddAssign & Core.SubAssign & Core.MulAssign &
       Core.DivAssign & Core.ModAssign](ref a: T, b: T) -> T {
  a += b;
  a -= b;
  a *= b;
  a /= b;
  a %= b;
  return (a + b) - (a * b) + (a / b) + (a % b);
}

fn GenericBitwise[
    T: Core.Copy & Core.Destroy & Core.BitAnd & Core.BitOr & Core.BitXor &
       Core.LeftShift & Core.RightShift & Core.BitAndAssign &
       Core.BitOrAssign & Core.BitXorAssign & Core.LeftShiftAssign &
       Core.RightShiftAssign](ref a: T, b: T) -> T {
  a &= b;
  a |= b;
  a ^= b;
  a <<= b;
  a >>= b;
  return ((a & b) | (a ^ b)) | ((a << b) ^ (a >> b));
}

fn TestGeneric(a: Number, b: Number, ref c: Number, ref x: u32, y: u32) {
  let _: Number = GenericMul(a, b);
  let _: Number = GenericDiv(a, b);
  GenericMulAssign(ref c, b);
  let _: u32 = GenericArithmetic(ref x, y);
  let _: u32 = GenericBitwise(ref x, y);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'overloaded.carbon'
// CHECK:STDOUT: source_filename = "overloaded.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.Number.Main:Negate.Core"(ptr sret({ i1 }) %return, ptr %self) #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc19_36.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !23
// CHECK:STDOUT:   %.loc19_36.2 = load i8, ptr %.loc19_36.1.is_positive, align 1, !dbg !23
// CHECK:STDOUT:   %.loc19_36.21 = trunc i8 %.loc19_36.2 to i1, !dbg !23
// CHECK:STDOUT:   %.loc19_28 = xor i1 %.loc19_36.21, true, !dbg !24
// CHECK:STDOUT:   %.loc19_48.2.is_positive = getelementptr inbounds nuw { i1 }, ptr %return, i32 0, i32 0, !dbg !25
// CHECK:STDOUT:   %.loc19_48.3 = zext i1 %.loc19_28 to i8, !dbg !25
// CHECK:STDOUT:   store i8 %.loc19_48.3, ptr %.loc19_48.2.is_positive, align 1, !dbg !25
// CHECK:STDOUT:   ret void, !dbg !26
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.Number.Main:MulWith.124acaad20680602.Core"(ptr sret({ i1 }) %return, ptr %self, ptr %other) #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc24_33.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !35
// CHECK:STDOUT:   %.loc24_33.2 = load i8, ptr %.loc24_33.1.is_positive, align 1, !dbg !35
// CHECK:STDOUT:   %.loc24_33.21 = trunc i8 %.loc24_33.2 to i1, !dbg !35
// CHECK:STDOUT:   br i1 %.loc24_33.21, label %and.rhs.loc24, label %and.result.loc24, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc24:                                    ; preds = %entry
// CHECK:STDOUT:   %.loc24_55.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %other, i32 0, i32 0, !dbg !36
// CHECK:STDOUT:   %.loc24_55.2 = load i8, ptr %.loc24_55.1.is_positive, align 1, !dbg !36
// CHECK:STDOUT:   %.loc24_55.22 = trunc i8 %.loc24_55.2 to i1, !dbg !36
// CHECK:STDOUT:   br label %and.result.loc24, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc24:                                 ; preds = %and.rhs.loc24, %entry
// CHECK:STDOUT:   %0 = phi i1 [ false, %entry ], [ %.loc24_55.22, %and.rhs.loc24 ]
// CHECK:STDOUT:   %.loc24_69.1 = xor i1 %0, true, !dbg !37
// CHECK:STDOUT:   br i1 %.loc24_69.1, label %or.rhs, label %or.result, !dbg !37
// CHECK:STDOUT:
// CHECK:STDOUT: or.rhs:                                           ; preds = %and.result.loc24
// CHECK:STDOUT:   %.loc25_38.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !38
// CHECK:STDOUT:   %.loc25_38.2 = load i8, ptr %.loc25_38.1.is_positive, align 1, !dbg !38
// CHECK:STDOUT:   %.loc25_38.23 = trunc i8 %.loc25_38.2 to i1, !dbg !38
// CHECK:STDOUT:   %.loc25_30 = xor i1 %.loc25_38.23, true, !dbg !39
// CHECK:STDOUT:   br i1 %.loc25_30, label %and.rhs.loc25, label %and.result.loc25, !dbg !39
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc25:                                    ; preds = %or.rhs
// CHECK:STDOUT:   %.loc25_64.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %other, i32 0, i32 0, !dbg !40
// CHECK:STDOUT:   %.loc25_64.2 = load i8, ptr %.loc25_64.1.is_positive, align 1, !dbg !40
// CHECK:STDOUT:   %.loc25_64.24 = trunc i8 %.loc25_64.2 to i1, !dbg !40
// CHECK:STDOUT:   %.loc25_55 = xor i1 %.loc25_64.24, true, !dbg !41
// CHECK:STDOUT:   br label %and.result.loc25, !dbg !39
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc25:                                 ; preds = %and.rhs.loc25, %or.rhs
// CHECK:STDOUT:   %1 = phi i1 [ false, %or.rhs ], [ %.loc25_55, %and.rhs.loc25 ]
// CHECK:STDOUT:   br label %or.result, !dbg !37
// CHECK:STDOUT:
// CHECK:STDOUT: or.result:                                        ; preds = %and.result.loc25, %and.result.loc24
// CHECK:STDOUT:   %2 = phi i1 [ true, %and.result.loc24 ], [ %1, %and.result.loc25 ]
// CHECK:STDOUT:   %.loc25_77.2.is_positive = getelementptr inbounds nuw { i1 }, ptr %return, i32 0, i32 0, !dbg !42
// CHECK:STDOUT:   %.loc25_77.3 = zext i1 %2 to i8, !dbg !42
// CHECK:STDOUT:   store i8 %.loc25_77.3, ptr %.loc25_77.2.is_positive, align 1, !dbg !42
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.Number.Main:DivWith.124acaad20680602.Core"(ptr sret({ i1 }) %return, ptr %self, ptr %other) #0 !dbg !45 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Number.Main:MulWith.124acaad20680602.Core"(ptr %return, ptr %self, ptr %other), !dbg !52
// CHECK:STDOUT:   ret void, !dbg !54
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_COp.Number.Main:MulAssignWith.124acaad20680602.Core"(ptr %self, ptr %other) #0 !dbg !56 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_9.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !61
// CHECK:STDOUT:   %.loc35_29.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !62
// CHECK:STDOUT:   %.loc35_29.2 = load i8, ptr %.loc35_29.1.is_positive, align 1, !dbg !62
// CHECK:STDOUT:   %.loc35_29.21 = trunc i8 %.loc35_29.2 to i1, !dbg !62
// CHECK:STDOUT:   br i1 %.loc35_29.21, label %and.rhs.loc35, label %and.result.loc35, !dbg !62
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc35:                                    ; preds = %entry
// CHECK:STDOUT:   %.loc35_51.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %other, i32 0, i32 0, !dbg !63
// CHECK:STDOUT:   %.loc35_51.2 = load i8, ptr %.loc35_51.1.is_positive, align 1, !dbg !63
// CHECK:STDOUT:   %.loc35_51.22 = trunc i8 %.loc35_51.2 to i1, !dbg !63
// CHECK:STDOUT:   br label %and.result.loc35, !dbg !62
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc35:                                 ; preds = %and.rhs.loc35, %entry
// CHECK:STDOUT:   %0 = phi i1 [ false, %entry ], [ %.loc35_51.22, %and.rhs.loc35 ]
// CHECK:STDOUT:   %.loc35_65.1 = xor i1 %0, true, !dbg !64
// CHECK:STDOUT:   br i1 %.loc35_65.1, label %or.rhs, label %or.result, !dbg !64
// CHECK:STDOUT:
// CHECK:STDOUT: or.rhs:                                           ; preds = %and.result.loc35
// CHECK:STDOUT:   %.loc36_33.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !65
// CHECK:STDOUT:   %.loc36_33.2 = load i8, ptr %.loc36_33.1.is_positive, align 1, !dbg !65
// CHECK:STDOUT:   %.loc36_33.23 = trunc i8 %.loc36_33.2 to i1, !dbg !65
// CHECK:STDOUT:   %.loc36_25 = xor i1 %.loc36_33.23, true, !dbg !66
// CHECK:STDOUT:   br i1 %.loc36_25, label %and.rhs.loc36, label %and.result.loc36, !dbg !66
// CHECK:STDOUT:
// CHECK:STDOUT: and.rhs.loc36:                                    ; preds = %or.rhs
// CHECK:STDOUT:   %.loc36_59.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %other, i32 0, i32 0, !dbg !67
// CHECK:STDOUT:   %.loc36_59.2 = load i8, ptr %.loc36_59.1.is_positive, align 1, !dbg !67
// CHECK:STDOUT:   %.loc36_59.24 = trunc i8 %.loc36_59.2 to i1, !dbg !67
// CHECK:STDOUT:   %.loc36_50 = xor i1 %.loc36_59.24, true, !dbg !68
// CHECK:STDOUT:   br label %and.result.loc36, !dbg !66
// CHECK:STDOUT:
// CHECK:STDOUT: and.result.loc36:                                 ; preds = %and.rhs.loc36, %or.rhs
// CHECK:STDOUT:   %1 = phi i1 [ false, %or.rhs ], [ %.loc36_50, %and.rhs.loc36 ]
// CHECK:STDOUT:   br label %or.result, !dbg !64
// CHECK:STDOUT:
// CHECK:STDOUT: or.result:                                        ; preds = %and.result.loc36, %and.result.loc35
// CHECK:STDOUT:   %2 = phi i1 [ true, %and.result.loc35 ], [ %1, %and.result.loc36 ]
// CHECK:STDOUT:   %3 = zext i1 %2 to i8, !dbg !61
// CHECK:STDOUT:   store i8 %3, ptr %.loc35_9.is_positive, align 1, !dbg !61
// CHECK:STDOUT:   ret void, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCalculate.Main(ptr sret({ i1 }) %return, ptr %a, ptr %b) #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc41_10.1.temp = alloca { i1 }, align 1, !dbg !79
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc41_10.1.temp), !dbg !79
// CHECK:STDOUT:   call void @"_COp.Number.Main:Negate.Core"(ptr %.loc41_10.1.temp, ptr %a), !dbg !79
// CHECK:STDOUT:   call void @"_COp.Number.Main:MulWith.124acaad20680602.Core"(ptr %return, ptr %.loc41_10.1.temp, ptr %b), !dbg !79
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8479fbb1a0ad0f(ptr %.loc41_10.1.temp), !dbg !79
// CHECK:STDOUT:   ret void, !dbg !81
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.4fdb30b3bf76b447:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc41_10.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !85
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %.loc41_10.1.is_positive), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.3e6115156c640ef2:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !87 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc41_10.1.is_positive = getelementptr inbounds nuw { i1 }, ptr %self, i32 0, i32 0, !dbg !89
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %.loc41_10.1.is_positive), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !89
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCalculateDiv.Main(ptr sret({ i1 }) %return, ptr %a, ptr %b) #0 !dbg !91 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Number.Main:DivWith.124acaad20680602.Core"(ptr %return, ptr %a, ptr %b), !dbg !98
// CHECK:STDOUT:   ret void, !dbg !100
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCalculateMulAssign.Main(ptr %a, ptr %b) #0 !dbg !102 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Number.Main:MulAssignWith.124acaad20680602.Core"(ptr %a, ptr %b), !dbg !107
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTestGeneric.Main(ptr %a, ptr %b, ptr %c, ptr %x, i32 %y) #0 !dbg !111 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc90_34.5.temp = alloca { i1 }, align 1, !dbg !124
// CHECK:STDOUT:   %.loc91_34.5.temp = alloca { i1 }, align 1, !dbg !129
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc90_34.5.temp), !dbg !124
// CHECK:STDOUT:   call void @_CGenericMul.Main.d08cf41c1315b1dc(ptr %.loc90_34.5.temp, ptr %a, ptr %b), !dbg !124
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc91_34.5.temp), !dbg !129
// CHECK:STDOUT:   call void @_CGenericDiv.Main.14ed27a6ee9f03fc(ptr %.loc91_34.5.temp, ptr %a, ptr %b), !dbg !129
// CHECK:STDOUT:   call void @_CGenericMulAssign.Main.14ed27a6ee9f03fc(ptr %c, ptr %b), !dbg !135
// CHECK:STDOUT:   %GenericArithmetic.call = call i32 @_CGenericArithmetic.Main.5ca27a73e80d6b45(ptr %x, i32 %y), !dbg !139
// CHECK:STDOUT:   %GenericBitwise.call = call i32 @_CGenericBitwise.Main.f92613992234a1d6(ptr %x, i32 %y), !dbg !147
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8479fbb1a0ad0f(ptr %.loc91_34.5.temp), !dbg !129
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7f8479fbb1a0ad0f(ptr %.loc90_34.5.temp), !dbg !124
// CHECK:STDOUT:   ret void, !dbg !150
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7f8479fbb1a0ad0f(ptr %t) #0 !dbg !153 {
// CHECK:STDOUT:   call void @"_COp.3e6115156c640ef2:core.SubobjectDestroy.Core"(ptr %t), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !158
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %t) #0 !dbg !160 {
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGenericMul.Main.d08cf41c1315b1dc(ptr sret({ i1 }) %return, ptr %a, ptr %b) #0 !dbg !167 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Number.Main:MulWith.124acaad20680602.Core"(ptr %return, ptr %a, ptr %b), !dbg !174
// CHECK:STDOUT:   ret void, !dbg !176
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGenericDiv.Main.14ed27a6ee9f03fc(ptr sret({ i1 }) %return, ptr %a, ptr %b) #0 !dbg !178 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Number.Main:DivWith.124acaad20680602.Core"(ptr %return, ptr %a, ptr %b), !dbg !185
// CHECK:STDOUT:   ret void, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CGenericMulAssign.Main.14ed27a6ee9f03fc(ptr %a, ptr %b) #0 !dbg !189 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.Number.Main:MulAssignWith.124acaad20680602.Core"(ptr %a, ptr %b), !dbg !194
// CHECK:STDOUT:   ret void, !dbg !196
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGenericArithmetic.Main.5ca27a73e80d6b45(ptr %a, i32 %b) #0 !dbg !198 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc73_13.2.temp = alloca i32, align 4, !dbg !217
// CHECK:STDOUT:   %.loc73_23.2.temp = alloca i32, align 4, !dbg !219
// CHECK:STDOUT:   %.loc73_18.2.temp = alloca i32, align 4, !dbg !221
// CHECK:STDOUT:   %.loc73_33.2.temp = alloca i32, align 4, !dbg !222
// CHECK:STDOUT:   %.loc73_28.2.temp = alloca i32, align 4, !dbg !221
// CHECK:STDOUT:   %.loc73_43.2.temp = alloca i32, align 4, !dbg !224
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.03689a112aff8ff9.Core:UInt.9452f4c51951679b.Core:AddAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %a, i32 %b), !dbg !205
// CHECK:STDOUT:   call void @"_COp:thunk:SubAssignWith.746c0460e15898f0.Core:UInt.9452f4c51951679b.Core:SubAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %a, i32 %b), !dbg !209
// CHECK:STDOUT:   call void @"_COp:thunk:MulAssignWith.d888a541870a2f88.Core:UInt.9452f4c51951679b.Core:MulAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %a, i32 %b), !dbg !211
// CHECK:STDOUT:   call void @"_COp:thunk:DivAssignWith.4e62faba5c1dc840.Core:UInt.9452f4c51951679b.Core:DivAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %a, i32 %b), !dbg !213
// CHECK:STDOUT:   call void @"_COp:thunk:ModAssignWith.a30eea9c8c788e29.Core:UInt.9452f4c51951679b.Core:ModAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %a, i32 %b), !dbg !215
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc73_13.2.temp), !dbg !217
// CHECK:STDOUT:   %.loc73_11 = load i32, ptr %a, align 4, !dbg !217
// CHECK:STDOUT:   %AddWith.WithSelf.Op.call.loc73_13 = add i32 %.loc73_11, %b, !dbg !217
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc73_23.2.temp), !dbg !219
// CHECK:STDOUT:   %.loc73_21 = load i32, ptr %a, align 4, !dbg !219
// CHECK:STDOUT:   %MulWith.WithSelf.Op.call = mul i32 %.loc73_21, %b, !dbg !219
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc73_18.2.temp), !dbg !221
// CHECK:STDOUT:   store i32 %AddWith.WithSelf.Op.call.loc73_13, ptr %.loc73_13.2.temp, align 4, !dbg !217
// CHECK:STDOUT:   %.loc73_13.4 = load i32, ptr %.loc73_13.2.temp, align 4, !dbg !217
// CHECK:STDOUT:   store i32 %MulWith.WithSelf.Op.call, ptr %.loc73_23.2.temp, align 4, !dbg !219
// CHECK:STDOUT:   %.loc73_23.4 = load i32, ptr %.loc73_23.2.temp, align 4, !dbg !219
// CHECK:STDOUT:   %SubWith.WithSelf.Op.call = sub i32 %.loc73_13.4, %.loc73_23.4, !dbg !221
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc73_33.2.temp), !dbg !222
// CHECK:STDOUT:   %.loc73_31 = load i32, ptr %a, align 4, !dbg !222
// CHECK:STDOUT:   %DivWith.WithSelf.Op.call = udiv i32 %.loc73_31, %b, !dbg !222
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc73_28.2.temp), !dbg !221
// CHECK:STDOUT:   store i32 %SubWith.WithSelf.Op.call, ptr %.loc73_18.2.temp, align 4, !dbg !221
// CHECK:STDOUT:   %.loc73_18.4 = load i32, ptr %.loc73_18.2.temp, align 4, !dbg !221
// CHECK:STDOUT:   store i32 %DivWith.WithSelf.Op.call, ptr %.loc73_33.2.temp, align 4, !dbg !222
// CHECK:STDOUT:   %.loc73_33.4 = load i32, ptr %.loc73_33.2.temp, align 4, !dbg !222
// CHECK:STDOUT:   %AddWith.WithSelf.Op.call.loc73_28 = add i32 %.loc73_18.4, %.loc73_33.4, !dbg !221
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc73_43.2.temp), !dbg !224
// CHECK:STDOUT:   %.loc73_41 = load i32, ptr %a, align 4, !dbg !224
// CHECK:STDOUT:   %ModWith.WithSelf.Op.call = urem i32 %.loc73_41, %b, !dbg !224
// CHECK:STDOUT:   store i32 %AddWith.WithSelf.Op.call.loc73_28, ptr %.loc73_28.2.temp, align 4, !dbg !221
// CHECK:STDOUT:   %.loc73_28.4 = load i32, ptr %.loc73_28.2.temp, align 4, !dbg !221
// CHECK:STDOUT:   store i32 %ModWith.WithSelf.Op.call, ptr %.loc73_43.2.temp, align 4, !dbg !224
// CHECK:STDOUT:   %.loc73_43.4 = load i32, ptr %.loc73_43.2.temp, align 4, !dbg !224
// CHECK:STDOUT:   %AddWith.WithSelf.Op.call.loc73_38 = add i32 %.loc73_28.4, %.loc73_43.4, !dbg !221
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc73_43.2.temp), !dbg !224
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc73_28.2.temp), !dbg !221
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc73_33.2.temp), !dbg !222
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc73_18.2.temp), !dbg !221
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc73_23.2.temp), !dbg !219
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc73_13.2.temp), !dbg !217
// CHECK:STDOUT:   ret i32 %AddWith.WithSelf.Op.call.loc73_38, !dbg !228
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CGenericBitwise.Main.f92613992234a1d6(ptr %a, i32 %b) #0 !dbg !230 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc86_14.2.temp = alloca i32, align 4, !dbg !247
// CHECK:STDOUT:   %.loc86_24.2.temp = alloca i32, align 4, !dbg !249
// CHECK:STDOUT:   %.loc86_19.2.temp = alloca i32, align 4, !dbg !251
// CHECK:STDOUT:   %.loc86_36.2.temp = alloca i32, align 4, !dbg !252
// CHECK:STDOUT:   %.loc86_47.2.temp = alloca i32, align 4, !dbg !254
// CHECK:STDOUT:   %.loc86_42.2.temp = alloca i32, align 4, !dbg !256
// CHECK:STDOUT:   call void @"_COp:thunk:BitAndAssignWith.8b3ad9f7b42a617b.Core:UInt.9452f4c51951679b.Core:BitAndAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %a, i32 %b), !dbg !237
// CHECK:STDOUT:   call void @"_COp:thunk:BitOrAssignWith.2640c3c555bf1274.Core:UInt.9452f4c51951679b.Core:BitOrAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %a, i32 %b), !dbg !239
// CHECK:STDOUT:   call void @"_COp:thunk:BitXorAssignWith.0bc4e9852e2182b6.Core:UInt.9452f4c51951679b.Core:BitXorAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %a, i32 %b), !dbg !241
// CHECK:STDOUT:   call void @"_COp:thunk:LeftShiftAssignWith.07eab5aac2f19589.Core:UInt.9452f4c51951679b.Core:LeftShiftAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %a, i32 %b), !dbg !243
// CHECK:STDOUT:   call void @"_COp:thunk:RightShiftAssignWith.6142f39d2fc5fb60.Core:UInt.9452f4c51951679b.Core:RightShiftAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %a, i32 %b), !dbg !245
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc86_14.2.temp), !dbg !247
// CHECK:STDOUT:   %.loc86_12 = load i32, ptr %a, align 4, !dbg !247
// CHECK:STDOUT:   %BitAndWith.WithSelf.Op.call = and i32 %.loc86_12, %b, !dbg !247
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc86_24.2.temp), !dbg !249
// CHECK:STDOUT:   %.loc86_22 = load i32, ptr %a, align 4, !dbg !249
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.call.loc86_24 = xor i32 %.loc86_22, %b, !dbg !249
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc86_19.2.temp), !dbg !251
// CHECK:STDOUT:   store i32 %BitAndWith.WithSelf.Op.call, ptr %.loc86_14.2.temp, align 4, !dbg !247
// CHECK:STDOUT:   %.loc86_14.4 = load i32, ptr %.loc86_14.2.temp, align 4, !dbg !247
// CHECK:STDOUT:   store i32 %BitXorWith.WithSelf.Op.call.loc86_24, ptr %.loc86_24.2.temp, align 4, !dbg !249
// CHECK:STDOUT:   %.loc86_24.4 = load i32, ptr %.loc86_24.2.temp, align 4, !dbg !249
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.call.loc86_19 = or i32 %.loc86_14.4, %.loc86_24.4, !dbg !251
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc86_36.2.temp), !dbg !252
// CHECK:STDOUT:   %.loc86_34 = load i32, ptr %a, align 4, !dbg !252
// CHECK:STDOUT:   %LeftShiftWith.WithSelf.Op.call = shl i32 %.loc86_34, %b, !dbg !252
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc86_47.2.temp), !dbg !254
// CHECK:STDOUT:   %.loc86_45 = load i32, ptr %a, align 4, !dbg !254
// CHECK:STDOUT:   %RightShiftWith.WithSelf.Op.call = lshr i32 %.loc86_45, %b, !dbg !254
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc86_42.2.temp), !dbg !256
// CHECK:STDOUT:   store i32 %LeftShiftWith.WithSelf.Op.call, ptr %.loc86_36.2.temp, align 4, !dbg !252
// CHECK:STDOUT:   %.loc86_36.4 = load i32, ptr %.loc86_36.2.temp, align 4, !dbg !252
// CHECK:STDOUT:   store i32 %RightShiftWith.WithSelf.Op.call, ptr %.loc86_47.2.temp, align 4, !dbg !254
// CHECK:STDOUT:   %.loc86_47.4 = load i32, ptr %.loc86_47.2.temp, align 4, !dbg !254
// CHECK:STDOUT:   %BitXorWith.WithSelf.Op.call.loc86_42 = xor i32 %.loc86_36.4, %.loc86_47.4, !dbg !256
// CHECK:STDOUT:   store i32 %BitOrWith.WithSelf.Op.call.loc86_19, ptr %.loc86_19.2.temp, align 4, !dbg !251
// CHECK:STDOUT:   %.loc86_19.4 = load i32, ptr %.loc86_19.2.temp, align 4, !dbg !251
// CHECK:STDOUT:   store i32 %BitXorWith.WithSelf.Op.call.loc86_42, ptr %.loc86_42.2.temp, align 4, !dbg !256
// CHECK:STDOUT:   %.loc86_42.4 = load i32, ptr %.loc86_42.2.temp, align 4, !dbg !256
// CHECK:STDOUT:   %BitOrWith.WithSelf.Op.call.loc86_30 = or i32 %.loc86_19.4, %.loc86_42.4, !dbg !257
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc86_42.2.temp), !dbg !256
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc86_19.2.temp), !dbg !251
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc86_47.2.temp), !dbg !254
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc86_36.2.temp), !dbg !252
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc86_24.2.temp), !dbg !249
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ff339b9821763c59(ptr %.loc86_14.2.temp), !dbg !247
// CHECK:STDOUT:   ret i32 %BitOrWith.WithSelf.Op.call.loc86_30, !dbg !258
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.03689a112aff8ff9.Core:UInt.9452f4c51951679b.Core:AddAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %self, i32 %other) #2 !dbg !261 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %other), !dbg !265
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !265
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !265
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !265
// CHECK:STDOUT:   ret void, !dbg !265
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:SubAssignWith.746c0460e15898f0.Core:UInt.9452f4c51951679b.Core:SubAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %self, i32 %other) #2 !dbg !269 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %other), !dbg !273
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !273
// CHECK:STDOUT:   %3 = sub i32 %2, %1, !dbg !273
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !273
// CHECK:STDOUT:   ret void, !dbg !273
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:MulAssignWith.d888a541870a2f88.Core:UInt.9452f4c51951679b.Core:MulAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %self, i32 %other) #2 !dbg !275 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %other), !dbg !279
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !279
// CHECK:STDOUT:   %3 = mul i32 %2, %1, !dbg !279
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !279
// CHECK:STDOUT:   ret void, !dbg !279
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:DivAssignWith.4e62faba5c1dc840.Core:UInt.9452f4c51951679b.Core:DivAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %self, i32 %other) #2 !dbg !281 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %other), !dbg !285
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !285
// CHECK:STDOUT:   %3 = udiv i32 %2, %1, !dbg !285
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !285
// CHECK:STDOUT:   ret void, !dbg !285
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:ModAssignWith.a30eea9c8c788e29.Core:UInt.9452f4c51951679b.Core:ModAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %self, i32 %other) #2 !dbg !287 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %other), !dbg !291
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !291
// CHECK:STDOUT:   %3 = urem i32 %2, %1, !dbg !291
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !291
// CHECK:STDOUT:   ret void, !dbg !291
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ff339b9821763c59(ptr %t) #0 !dbg !293 {
// CHECK:STDOUT:   ret void, !dbg !298
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:BitAndAssignWith.8b3ad9f7b42a617b.Core:UInt.9452f4c51951679b.Core:BitAndAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %self, i32 %other) #2 !dbg !300 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %other), !dbg !304
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !304
// CHECK:STDOUT:   %3 = and i32 %2, %1, !dbg !304
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !304
// CHECK:STDOUT:   ret void, !dbg !304
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:BitOrAssignWith.2640c3c555bf1274.Core:UInt.9452f4c51951679b.Core:BitOrAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %self, i32 %other) #2 !dbg !306 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %other), !dbg !310
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !310
// CHECK:STDOUT:   %3 = or i32 %2, %1, !dbg !310
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !310
// CHECK:STDOUT:   ret void, !dbg !310
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:BitXorAssignWith.0bc4e9852e2182b6.Core:UInt.9452f4c51951679b.Core:BitXorAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %self, i32 %other) #2 !dbg !312 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %other), !dbg !316
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !316
// CHECK:STDOUT:   %3 = xor i32 %2, %1, !dbg !316
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !316
// CHECK:STDOUT:   ret void, !dbg !316
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:LeftShiftAssignWith.07eab5aac2f19589.Core:UInt.9452f4c51951679b.Core:LeftShiftAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %self, i32 %other) #2 !dbg !318 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %other), !dbg !322
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !322
// CHECK:STDOUT:   %3 = shl i32 %2, %1, !dbg !322
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !322
// CHECK:STDOUT:   ret void, !dbg !322
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:RightShiftAssignWith.6142f39d2fc5fb60.Core:UInt.9452f4c51951679b.Core:RightShiftAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712"(ptr %self, i32 %other) #2 !dbg !324 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %other), !dbg !328
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !328
// CHECK:STDOUT:   %3 = lshr i32 %2, %1, !dbg !328
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !328
// CHECK:STDOUT:   ret void, !dbg !328
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa"(i32 %self) #0 !dbg !330 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.180681b028ad46dc.Core.29b9f977f12c3f24"(i32 %self), !dbg !336
// CHECK:STDOUT:   ret i32 %1, !dbg !337
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.UInt.9452f4c51951679b.Core:FromUInt.180681b028ad46dc.Core.29b9f977f12c3f24"(i32 %from) #0 !dbg !339 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.be1e879c1ad406d8"(i32 %from), !dbg !344
// CHECK:STDOUT:   ret i32 %1, !dbg !346
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !348 {
// CHECK:STDOUT:   ret i32 %self, !dbg !354
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 13, 12 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.7f8479fbb1a0ad0f, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.29e6aa423646b61a, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.ff339b9821763c59, { 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa", { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "overloaded.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_unsigned)
// CHECK:STDOUT: !7 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !8 = !{!7, !7}
// CHECK:STDOUT: !9 = !DISubroutineType(types: !8)
// CHECK:STDOUT: !10 = !{!7, !7, !7}
// CHECK:STDOUT: !11 = !DISubroutineType(types: !10)
// CHECK:STDOUT: !12 = !{null, !7, !7}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !7}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !{null, !7, !7, !7, !6, !6}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Number.Main:Negate.Core", scope: null, file: !4, line: 18, type: !9, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !27)
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !18, type: !7)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 32, scope: !18)
// CHECK:STDOUT: !24 = !DILocation(line: 19, column: 28, scope: !18)
// CHECK:STDOUT: !25 = !DILocation(line: 19, column: 12, scope: !18)
// CHECK:STDOUT: !26 = !DILocation(line: 19, column: 5, scope: !18)
// CHECK:STDOUT: !27 = !{!19}
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Number.Main:MulWith.124acaad20680602.Core", scope: null, file: !4, line: 23, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !44)
// CHECK:STDOUT: !29 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !30 = !DILocalVariable(arg: 2, scope: !28, type: !7)
// CHECK:STDOUT: !35 = !DILocation(line: 24, column: 29, scope: !28)
// CHECK:STDOUT: !36 = !DILocation(line: 24, column: 50, scope: !28)
// CHECK:STDOUT: !37 = !DILocation(line: 24, column: 28, scope: !28)
// CHECK:STDOUT: !38 = !DILocation(line: 25, column: 34, scope: !28)
// CHECK:STDOUT: !39 = !DILocation(line: 25, column: 30, scope: !28)
// CHECK:STDOUT: !40 = !DILocation(line: 25, column: 59, scope: !28)
// CHECK:STDOUT: !41 = !DILocation(line: 25, column: 55, scope: !28)
// CHECK:STDOUT: !42 = !DILocation(line: 24, column: 12, scope: !28)
// CHECK:STDOUT: !43 = !DILocation(line: 24, column: 5, scope: !28)
// CHECK:STDOUT: !44 = !{!29, !30}
// CHECK:STDOUT: !45 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Number.Main:DivWith.124acaad20680602.Core", scope: null, file: !4, line: 29, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !55)
// CHECK:STDOUT: !46 = !DILocalVariable(arg: 1, scope: !45, type: !7)
// CHECK:STDOUT: !47 = !DILocalVariable(arg: 2, scope: !45, type: !7)
// CHECK:STDOUT: !52 = !DILocation(line: 30, column: 12, scope: !45)
// CHECK:STDOUT: !54 = !DILocation(line: 30, column: 5, scope: !45)
// CHECK:STDOUT: !55 = !{!46, !47}
// CHECK:STDOUT: !56 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Number.Main:MulAssignWith.124acaad20680602.Core", scope: null, file: !4, line: 34, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !70)
// CHECK:STDOUT: !57 = !DILocalVariable(arg: 1, scope: !56, type: !7)
// CHECK:STDOUT: !58 = !DILocalVariable(arg: 2, scope: !56, type: !7)
// CHECK:STDOUT: !61 = !DILocation(line: 35, column: 5, scope: !56)
// CHECK:STDOUT: !62 = !DILocation(line: 35, column: 25, scope: !56)
// CHECK:STDOUT: !63 = !DILocation(line: 35, column: 46, scope: !56)
// CHECK:STDOUT: !64 = !DILocation(line: 35, column: 24, scope: !56)
// CHECK:STDOUT: !65 = !DILocation(line: 36, column: 29, scope: !56)
// CHECK:STDOUT: !66 = !DILocation(line: 36, column: 25, scope: !56)
// CHECK:STDOUT: !67 = !DILocation(line: 36, column: 54, scope: !56)
// CHECK:STDOUT: !68 = !DILocation(line: 36, column: 50, scope: !56)
// CHECK:STDOUT: !69 = !DILocation(line: 34, column: 3, scope: !56)
// CHECK:STDOUT: !70 = !{!57, !58}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "Calculate", linkageName: "_CCalculate.Main", scope: null, file: !4, line: 40, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !82)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !7)
// CHECK:STDOUT: !73 = !DILocalVariable(arg: 2, scope: !71, type: !7)
// CHECK:STDOUT: !79 = !DILocation(line: 41, column: 10, scope: !71)
// CHECK:STDOUT: !81 = !DILocation(line: 41, column: 3, scope: !71)
// CHECK:STDOUT: !82 = !{!72, !73}
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "Op", linkageName: "_COp.4fdb30b3bf76b447:core.SubobjectDestroy.Core", scope: null, file: !4, line: 41, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !86)
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !83, type: !7)
// CHECK:STDOUT: !85 = !DILocation(line: 41, column: 10, scope: !83)
// CHECK:STDOUT: !86 = !{!84}
// CHECK:STDOUT: !87 = distinct !DISubprogram(name: "Op", linkageName: "_COp.3e6115156c640ef2:core.SubobjectDestroy.Core", scope: null, file: !4, line: 41, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !90)
// CHECK:STDOUT: !88 = !DILocalVariable(arg: 1, scope: !87, type: !7)
// CHECK:STDOUT: !89 = !DILocation(line: 41, column: 10, scope: !87)
// CHECK:STDOUT: !90 = !{!88}
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "CalculateDiv", linkageName: "_CCalculateDiv.Main", scope: null, file: !4, line: 44, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !101)
// CHECK:STDOUT: !92 = !DILocalVariable(arg: 1, scope: !91, type: !7)
// CHECK:STDOUT: !93 = !DILocalVariable(arg: 2, scope: !91, type: !7)
// CHECK:STDOUT: !98 = !DILocation(line: 45, column: 10, scope: !91)
// CHECK:STDOUT: !100 = !DILocation(line: 45, column: 3, scope: !91)
// CHECK:STDOUT: !101 = !{!92, !93}
// CHECK:STDOUT: !102 = distinct !DISubprogram(name: "CalculateMulAssign", linkageName: "_CCalculateMulAssign.Main", scope: null, file: !4, line: 48, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !110)
// CHECK:STDOUT: !103 = !DILocalVariable(arg: 1, scope: !102, type: !7)
// CHECK:STDOUT: !104 = !DILocalVariable(arg: 2, scope: !102, type: !7)
// CHECK:STDOUT: !107 = !DILocation(line: 49, column: 3, scope: !102)
// CHECK:STDOUT: !109 = !DILocation(line: 48, column: 1, scope: !102)
// CHECK:STDOUT: !110 = !{!103, !104}
// CHECK:STDOUT: !111 = distinct !DISubprogram(name: "TestGeneric", linkageName: "_CTestGeneric.Main", scope: null, file: !4, line: 89, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !151)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 1, scope: !111, type: !7)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 2, scope: !111, type: !7)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 3, scope: !111, type: !7)
// CHECK:STDOUT: !115 = !DILocalVariable(arg: 4, scope: !111, type: !6)
// CHECK:STDOUT: !116 = !DILocalVariable(arg: 5, scope: !111, type: !6)
// CHECK:STDOUT: !124 = !DILocation(line: 90, column: 19, scope: !111)
// CHECK:STDOUT: !129 = !DILocation(line: 91, column: 19, scope: !111)
// CHECK:STDOUT: !135 = !DILocation(line: 92, column: 3, scope: !111)
// CHECK:STDOUT: !139 = !DILocation(line: 93, column: 16, scope: !111)
// CHECK:STDOUT: !140 = !{!6, !6, !6}
// CHECK:STDOUT: !141 = !DISubroutineType(types: !140)
// CHECK:STDOUT: !147 = !DILocation(line: 94, column: 16, scope: !111)
// CHECK:STDOUT: !150 = !DILocation(line: 89, column: 1, scope: !111)
// CHECK:STDOUT: !151 = !{!112, !113, !114, !115, !116}
// CHECK:STDOUT: !152 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !153 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7f8479fbb1a0ad0f", scope: null, file: !152, line: 36, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !159)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 1, scope: !153, type: !7)
// CHECK:STDOUT: !157 = !DILocation(line: 38, column: 5, scope: !153)
// CHECK:STDOUT: !158 = !DILocation(line: 36, column: 1, scope: !153)
// CHECK:STDOUT: !159 = !{!154}
// CHECK:STDOUT: !160 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.29e6aa423646b61a", scope: null, file: !152, line: 36, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !166)
// CHECK:STDOUT: !161 = !DILocalVariable(arg: 1, scope: !160, type: !7)
// CHECK:STDOUT: !165 = !DILocation(line: 36, column: 1, scope: !160)
// CHECK:STDOUT: !166 = !{!161}
// CHECK:STDOUT: !167 = distinct !DISubprogram(name: "GenericMul", linkageName: "_CGenericMul.Main.d08cf41c1315b1dc", scope: null, file: !4, line: 52, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !177)
// CHECK:STDOUT: !168 = !DILocalVariable(arg: 1, scope: !167, type: !7)
// CHECK:STDOUT: !169 = !DILocalVariable(arg: 2, scope: !167, type: !7)
// CHECK:STDOUT: !174 = !DILocation(line: 53, column: 10, scope: !167)
// CHECK:STDOUT: !176 = !DILocation(line: 53, column: 3, scope: !167)
// CHECK:STDOUT: !177 = !{!168, !169}
// CHECK:STDOUT: !178 = distinct !DISubprogram(name: "GenericDiv", linkageName: "_CGenericDiv.Main.14ed27a6ee9f03fc", scope: null, file: !4, line: 56, type: !11, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !188)
// CHECK:STDOUT: !179 = !DILocalVariable(arg: 1, scope: !178, type: !7)
// CHECK:STDOUT: !180 = !DILocalVariable(arg: 2, scope: !178, type: !7)
// CHECK:STDOUT: !185 = !DILocation(line: 57, column: 10, scope: !178)
// CHECK:STDOUT: !187 = !DILocation(line: 57, column: 3, scope: !178)
// CHECK:STDOUT: !188 = !{!179, !180}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "GenericMulAssign", linkageName: "_CGenericMulAssign.Main.14ed27a6ee9f03fc", scope: null, file: !4, line: 60, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !197)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !7)
// CHECK:STDOUT: !191 = !DILocalVariable(arg: 2, scope: !189, type: !7)
// CHECK:STDOUT: !194 = !DILocation(line: 61, column: 3, scope: !189)
// CHECK:STDOUT: !196 = !DILocation(line: 60, column: 1, scope: !189)
// CHECK:STDOUT: !197 = !{!190, !191}
// CHECK:STDOUT: !198 = distinct !DISubprogram(name: "GenericArithmetic", linkageName: "_CGenericArithmetic.Main.5ca27a73e80d6b45", scope: null, file: !4, line: 64, type: !141, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !229)
// CHECK:STDOUT: !199 = !DILocalVariable(arg: 1, scope: !198, type: !6)
// CHECK:STDOUT: !200 = !DILocalVariable(arg: 2, scope: !198, type: !6)
// CHECK:STDOUT: !205 = !DILocation(line: 68, column: 3, scope: !198)
// CHECK:STDOUT: !207 = !{null, !6, !6}
// CHECK:STDOUT: !208 = !DISubroutineType(types: !207)
// CHECK:STDOUT: !209 = !DILocation(line: 69, column: 3, scope: !198)
// CHECK:STDOUT: !211 = !DILocation(line: 70, column: 3, scope: !198)
// CHECK:STDOUT: !213 = !DILocation(line: 71, column: 3, scope: !198)
// CHECK:STDOUT: !215 = !DILocation(line: 72, column: 3, scope: !198)
// CHECK:STDOUT: !217 = !DILocation(line: 73, column: 11, scope: !198)
// CHECK:STDOUT: !219 = !DILocation(line: 73, column: 21, scope: !198)
// CHECK:STDOUT: !221 = !DILocation(line: 73, column: 10, scope: !198)
// CHECK:STDOUT: !222 = !DILocation(line: 73, column: 31, scope: !198)
// CHECK:STDOUT: !224 = !DILocation(line: 73, column: 41, scope: !198)
// CHECK:STDOUT: !226 = !{null, !6}
// CHECK:STDOUT: !227 = !DISubroutineType(types: !226)
// CHECK:STDOUT: !228 = !DILocation(line: 73, column: 3, scope: !198)
// CHECK:STDOUT: !229 = !{!199, !200}
// CHECK:STDOUT: !230 = distinct !DISubprogram(name: "GenericBitwise", linkageName: "_CGenericBitwise.Main.f92613992234a1d6", scope: null, file: !4, line: 76, type: !141, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !259)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 1, scope: !230, type: !6)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 2, scope: !230, type: !6)
// CHECK:STDOUT: !237 = !DILocation(line: 81, column: 3, scope: !230)
// CHECK:STDOUT: !239 = !DILocation(line: 82, column: 3, scope: !230)
// CHECK:STDOUT: !241 = !DILocation(line: 83, column: 3, scope: !230)
// CHECK:STDOUT: !243 = !DILocation(line: 84, column: 3, scope: !230)
// CHECK:STDOUT: !245 = !DILocation(line: 85, column: 3, scope: !230)
// CHECK:STDOUT: !247 = !DILocation(line: 86, column: 12, scope: !230)
// CHECK:STDOUT: !249 = !DILocation(line: 86, column: 22, scope: !230)
// CHECK:STDOUT: !251 = !DILocation(line: 86, column: 11, scope: !230)
// CHECK:STDOUT: !252 = !DILocation(line: 86, column: 34, scope: !230)
// CHECK:STDOUT: !254 = !DILocation(line: 86, column: 45, scope: !230)
// CHECK:STDOUT: !256 = !DILocation(line: 86, column: 33, scope: !230)
// CHECK:STDOUT: !257 = !DILocation(line: 86, column: 10, scope: !230)
// CHECK:STDOUT: !258 = !DILocation(line: 86, column: 3, scope: !230)
// CHECK:STDOUT: !259 = !{!231, !232}
// CHECK:STDOUT: !260 = !DIFile(filename: "{{.*}}/prelude/types/uint.carbon", directory: "")
// CHECK:STDOUT: !261 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.03689a112aff8ff9.Core:UInt.9452f4c51951679b.Core:AddAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712", scope: null, file: !260, line: 393, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !268)
// CHECK:STDOUT: !262 = !DILocalVariable(arg: 1, scope: !261, type: !6)
// CHECK:STDOUT: !263 = !DILocalVariable(arg: 2, scope: !261, type: !6)
// CHECK:STDOUT: !265 = !DILocation(line: 393, column: 3, scope: !261)
// CHECK:STDOUT: !266 = !{!6, !6}
// CHECK:STDOUT: !267 = !DISubroutineType(types: !266)
// CHECK:STDOUT: !268 = !{!262, !263}
// CHECK:STDOUT: !269 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:SubAssignWith.746c0460e15898f0.Core:UInt.9452f4c51951679b.Core:SubAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712", scope: null, file: !260, line: 438, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !274)
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 1, scope: !269, type: !6)
// CHECK:STDOUT: !271 = !DILocalVariable(arg: 2, scope: !269, type: !6)
// CHECK:STDOUT: !273 = !DILocation(line: 438, column: 3, scope: !269)
// CHECK:STDOUT: !274 = !{!270, !271}
// CHECK:STDOUT: !275 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:MulAssignWith.d888a541870a2f88.Core:UInt.9452f4c51951679b.Core:MulAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712", scope: null, file: !260, line: 428, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !280)
// CHECK:STDOUT: !276 = !DILocalVariable(arg: 1, scope: !275, type: !6)
// CHECK:STDOUT: !277 = !DILocalVariable(arg: 2, scope: !275, type: !6)
// CHECK:STDOUT: !279 = !DILocation(line: 428, column: 3, scope: !275)
// CHECK:STDOUT: !280 = !{!276, !277}
// CHECK:STDOUT: !281 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:DivAssignWith.4e62faba5c1dc840.Core:UInt.9452f4c51951679b.Core:DivAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712", scope: null, file: !260, line: 413, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !286)
// CHECK:STDOUT: !282 = !DILocalVariable(arg: 1, scope: !281, type: !6)
// CHECK:STDOUT: !283 = !DILocalVariable(arg: 2, scope: !281, type: !6)
// CHECK:STDOUT: !285 = !DILocation(line: 413, column: 3, scope: !281)
// CHECK:STDOUT: !286 = !{!282, !283}
// CHECK:STDOUT: !287 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:ModAssignWith.a30eea9c8c788e29.Core:UInt.9452f4c51951679b.Core:ModAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712", scope: null, file: !260, line: 423, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !292)
// CHECK:STDOUT: !288 = !DILocalVariable(arg: 1, scope: !287, type: !6)
// CHECK:STDOUT: !289 = !DILocalVariable(arg: 2, scope: !287, type: !6)
// CHECK:STDOUT: !291 = !DILocation(line: 423, column: 3, scope: !287)
// CHECK:STDOUT: !292 = !{!288, !289}
// CHECK:STDOUT: !293 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ff339b9821763c59", scope: null, file: !152, line: 36, type: !227, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !299)
// CHECK:STDOUT: !294 = !DILocalVariable(arg: 1, scope: !293, type: !6)
// CHECK:STDOUT: !298 = !DILocation(line: 36, column: 1, scope: !293)
// CHECK:STDOUT: !299 = !{!294}
// CHECK:STDOUT: !300 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:BitAndAssignWith.8b3ad9f7b42a617b.Core:UInt.9452f4c51951679b.Core:BitAndAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712", scope: null, file: !260, line: 398, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !305)
// CHECK:STDOUT: !301 = !DILocalVariable(arg: 1, scope: !300, type: !6)
// CHECK:STDOUT: !302 = !DILocalVariable(arg: 2, scope: !300, type: !6)
// CHECK:STDOUT: !304 = !DILocation(line: 398, column: 3, scope: !300)
// CHECK:STDOUT: !305 = !{!301, !302}
// CHECK:STDOUT: !306 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:BitOrAssignWith.2640c3c555bf1274.Core:UInt.9452f4c51951679b.Core:BitOrAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712", scope: null, file: !260, line: 403, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !311)
// CHECK:STDOUT: !307 = !DILocalVariable(arg: 1, scope: !306, type: !6)
// CHECK:STDOUT: !308 = !DILocalVariable(arg: 2, scope: !306, type: !6)
// CHECK:STDOUT: !310 = !DILocation(line: 403, column: 3, scope: !306)
// CHECK:STDOUT: !311 = !{!307, !308}
// CHECK:STDOUT: !312 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:BitXorAssignWith.0bc4e9852e2182b6.Core:UInt.9452f4c51951679b.Core:BitXorAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712", scope: null, file: !260, line: 408, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !317)
// CHECK:STDOUT: !313 = !DILocalVariable(arg: 1, scope: !312, type: !6)
// CHECK:STDOUT: !314 = !DILocalVariable(arg: 2, scope: !312, type: !6)
// CHECK:STDOUT: !316 = !DILocation(line: 408, column: 3, scope: !312)
// CHECK:STDOUT: !317 = !{!313, !314}
// CHECK:STDOUT: !318 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:LeftShiftAssignWith.07eab5aac2f19589.Core:UInt.9452f4c51951679b.Core:LeftShiftAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712", scope: null, file: !260, line: 418, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !323)
// CHECK:STDOUT: !319 = !DILocalVariable(arg: 1, scope: !318, type: !6)
// CHECK:STDOUT: !320 = !DILocalVariable(arg: 2, scope: !318, type: !6)
// CHECK:STDOUT: !322 = !DILocation(line: 418, column: 3, scope: !318)
// CHECK:STDOUT: !323 = !{!319, !320}
// CHECK:STDOUT: !324 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:RightShiftAssignWith.6142f39d2fc5fb60.Core:UInt.9452f4c51951679b.Core:RightShiftAssignWith.1d47ea6b06d1da31.Core.17a45d1efeed2712", scope: null, file: !260, line: 433, type: !208, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !329)
// CHECK:STDOUT: !325 = !DILocalVariable(arg: 1, scope: !324, type: !6)
// CHECK:STDOUT: !326 = !DILocalVariable(arg: 2, scope: !324, type: !6)
// CHECK:STDOUT: !328 = !DILocation(line: 433, column: 3, scope: !324)
// CHECK:STDOUT: !329 = !{!325, !326}
// CHECK:STDOUT: !330 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:ImplicitAs.3260578ee2895055.Core.06558c4d3a81f7aa", scope: null, file: !260, line: 88, type: !267, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !338)
// CHECK:STDOUT: !331 = !DILocalVariable(arg: 1, scope: !330, type: !6)
// CHECK:STDOUT: !336 = !DILocation(line: 89, column: 12, scope: !330)
// CHECK:STDOUT: !337 = !DILocation(line: 89, column: 5, scope: !330)
// CHECK:STDOUT: !338 = !{!331}
// CHECK:STDOUT: !339 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.UInt.9452f4c51951679b.Core:FromUInt.180681b028ad46dc.Core.29b9f977f12c3f24", scope: null, file: !260, line: 72, type: !267, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !347)
// CHECK:STDOUT: !340 = !DILocalVariable(arg: 1, scope: !339, type: !6)
// CHECK:STDOUT: !344 = !DILocation(line: 74, column: 17, scope: !339)
// CHECK:STDOUT: !346 = !DILocation(line: 74, column: 5, scope: !339)
// CHECK:STDOUT: !347 = !{!340}
// CHECK:STDOUT: !348 = distinct !DISubprogram(name: "AsUInt", linkageName: "_CAsUInt.UInt.9452f4c51951679b.Core:AnyUInt.Core.be1e879c1ad406d8", scope: null, file: !260, line: 60, type: !267, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !355)
// CHECK:STDOUT: !349 = !DILocalVariable(arg: 1, scope: !348, type: !6)
// CHECK:STDOUT: !354 = !DILocation(line: 60, column: 29, scope: !348)
// CHECK:STDOUT: !355 = !{!349}
// CHECK:STDOUT:
