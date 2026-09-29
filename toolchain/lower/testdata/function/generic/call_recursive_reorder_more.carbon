// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_reorder_more.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_reorder_more.carbon

// Builds on `call_recursive_reorder.carbon`.

fn F[T: type, U: type, V: type](x: T, y: U, z: V, count: i32) -> i32 {
  if (count > 2) {
    return count;
  }
  return F(x, z, y, count + 1);
}

fn M() {
  var val_i16 : i16;
  var val_i32 : i32;
  var val_i64 : i64;
  var val_f64 : f64;
  var val_bool : bool;

  var ptr_i16 : i16* = &val_i16;
  var ptr_i32 : i32* = &val_i32;
  var ptr_i64 : i64* = &val_i64;
  var ptr_bool : bool* = &val_bool;

  // Insert noise: non equivalent with the second set of calls.
  // (4*2 + 1 functions generated; first 3 coalesce with each other - self
  // recursion - , 4th remains as 2 specifics, 5th is already self recursion.
  // Start: 9 functions. End: 6 functions.
  F(val_i16, ptr_i32, ptr_bool, 0);
  F(val_i32, ptr_bool, ptr_i64, 0);
  F(val_bool, ptr_bool, ptr_i16, 0);
  F(val_i64, val_bool, ptr_bool, 0);
  F(val_f64, ptr_bool, ptr_bool, 0);

  // All equivalent (6*2 + 1 generated initially, coalesced to one)
  // Start: 13 functions. End: 1 function.
  F(ptr_bool, ptr_i16, ptr_i32, 0);
  F(ptr_bool, ptr_i16, ptr_i64, 0);
  F(ptr_i32, ptr_bool, ptr_i16, 0);
  F(ptr_i32, ptr_bool, ptr_i64, 0);
  F(ptr_i16, ptr_i32, ptr_i64, 0);
  F(ptr_i64, ptr_bool, ptr_i16, 0);
  F(ptr_bool, ptr_bool, ptr_bool, 0);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_recursive_reorder_more.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_reorder_more.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !20 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %val_i16.var = alloca i16, align 2, !dbg !21
// CHECK:STDOUT:   %val_i32.var = alloca i32, align 4, !dbg !23
// CHECK:STDOUT:   %val_i64.var = alloca i64, align 8, !dbg !25
// CHECK:STDOUT:   %val_f64.var = alloca double, align 8, !dbg !27
// CHECK:STDOUT:   %val_bool.var = alloca i1, align 1, !dbg !29
// CHECK:STDOUT:   %ptr_i16.var = alloca ptr, align 8, !dbg !31
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !35
// CHECK:STDOUT:   %ptr_i64.var = alloca ptr, align 8, !dbg !39
// CHECK:STDOUT:   %ptr_bool.var = alloca ptr, align 8, !dbg !43
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_i16.var), !dbg !21
// CHECK:STDOUT:   store i16 poison, ptr %val_i16.var, align 2, !dbg !21
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_i32.var), !dbg !23
// CHECK:STDOUT:   store i32 poison, ptr %val_i32.var, align 4, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_i64.var), !dbg !25
// CHECK:STDOUT:   store i64 poison, ptr %val_i64.var, align 8, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_f64.var), !dbg !27
// CHECK:STDOUT:   store double poison, ptr %val_f64.var, align 8, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_bool.var), !dbg !29
// CHECK:STDOUT:   store i8 poison, ptr %val_bool.var, align 1, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i16.var), !dbg !31
// CHECK:STDOUT:   store ptr %val_i16.var, ptr %ptr_i16.var, align 8, !dbg !31
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !35
// CHECK:STDOUT:   store ptr %val_i32.var, ptr %ptr_i32.var, align 8, !dbg !35
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i64.var), !dbg !39
// CHECK:STDOUT:   store ptr %val_i64.var, ptr %ptr_i64.var, align 8, !dbg !39
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_bool.var), !dbg !43
// CHECK:STDOUT:   store ptr %val_bool.var, ptr %ptr_bool.var, align 8, !dbg !43
// CHECK:STDOUT:   %.loc38_5 = load i16, ptr %val_i16.var, align 2, !dbg !47
// CHECK:STDOUT:   %.loc38_14 = load ptr, ptr %ptr_i32.var, align 8, !dbg !48
// CHECK:STDOUT:   %.loc38_23 = load ptr, ptr %ptr_bool.var, align 8, !dbg !49
// CHECK:STDOUT:   %F.call.loc38 = call i32 @_CF.Main.bb004feba9dd474d(i16 %.loc38_5, ptr %.loc38_14, ptr %.loc38_23, i32 0), !dbg !50
// CHECK:STDOUT:   %.loc39_5 = load i32, ptr %val_i32.var, align 4, !dbg !53
// CHECK:STDOUT:   %.loc39_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !54
// CHECK:STDOUT:   %.loc39_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !55
// CHECK:STDOUT:   %F.call.loc39 = call i32 @_CF.Main.f24eb6b05368f488(i32 %.loc39_5, ptr %.loc39_14, ptr %.loc39_24, i32 0), !dbg !56
// CHECK:STDOUT:   %.loc40_5 = load i8, ptr %val_bool.var, align 1, !dbg !59
// CHECK:STDOUT:   %.loc40_51 = trunc i8 %.loc40_5 to i1, !dbg !59
// CHECK:STDOUT:   %.loc40_15 = load ptr, ptr %ptr_bool.var, align 8, !dbg !60
// CHECK:STDOUT:   %.loc40_25 = load ptr, ptr %ptr_i16.var, align 8, !dbg !61
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.427cba2b3278d1dc(i1 %.loc40_51, ptr %.loc40_15, ptr %.loc40_25, i32 0), !dbg !62
// CHECK:STDOUT:   %.loc41_5 = load i64, ptr %val_i64.var, align 8, !dbg !65
// CHECK:STDOUT:   %.loc41_14 = load i8, ptr %val_bool.var, align 1, !dbg !66
// CHECK:STDOUT:   %.loc41_142 = trunc i8 %.loc41_14 to i1, !dbg !66
// CHECK:STDOUT:   %.loc41_24 = load ptr, ptr %ptr_bool.var, align 8, !dbg !67
// CHECK:STDOUT:   %F.call.loc41 = call i32 @_CF.Main.ea32bb7f769e7967(i64 %.loc41_5, i1 %.loc41_142, ptr %.loc41_24, i32 0), !dbg !68
// CHECK:STDOUT:   %.loc42_5 = load double, ptr %val_f64.var, align 8, !dbg !71
// CHECK:STDOUT:   %.loc42_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !72
// CHECK:STDOUT:   %.loc42_24 = load ptr, ptr %ptr_bool.var, align 8, !dbg !73
// CHECK:STDOUT:   %F.call.loc42 = call i32 @_CF.Main.6eab5c6411742e70(double %.loc42_5, ptr %.loc42_14, ptr %.loc42_24, i32 0), !dbg !74
// CHECK:STDOUT:   %.loc46_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !75
// CHECK:STDOUT:   %.loc46_15 = load ptr, ptr %ptr_i16.var, align 8, !dbg !76
// CHECK:STDOUT:   %.loc46_24 = load ptr, ptr %ptr_i32.var, align 8, !dbg !77
// CHECK:STDOUT:   %F.call.loc46 = call i32 @_CF.Main.732da705e71cd359(ptr %.loc46_5, ptr %.loc46_15, ptr %.loc46_24, i32 0), !dbg !78
// CHECK:STDOUT:   %.loc47_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !79
// CHECK:STDOUT:   %.loc47_15 = load ptr, ptr %ptr_i16.var, align 8, !dbg !80
// CHECK:STDOUT:   %.loc47_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !81
// CHECK:STDOUT:   %F.call.loc47 = call i32 @_CF.Main.732da705e71cd359(ptr %.loc47_5, ptr %.loc47_15, ptr %.loc47_24, i32 0), !dbg !82
// CHECK:STDOUT:   %.loc48_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !83
// CHECK:STDOUT:   %.loc48_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !84
// CHECK:STDOUT:   %.loc48_24 = load ptr, ptr %ptr_i16.var, align 8, !dbg !85
// CHECK:STDOUT:   %F.call.loc48 = call i32 @_CF.Main.732da705e71cd359(ptr %.loc48_5, ptr %.loc48_14, ptr %.loc48_24, i32 0), !dbg !86
// CHECK:STDOUT:   %.loc49_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !87
// CHECK:STDOUT:   %.loc49_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !88
// CHECK:STDOUT:   %.loc49_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !89
// CHECK:STDOUT:   %F.call.loc49 = call i32 @_CF.Main.732da705e71cd359(ptr %.loc49_5, ptr %.loc49_14, ptr %.loc49_24, i32 0), !dbg !90
// CHECK:STDOUT:   %.loc50_5 = load ptr, ptr %ptr_i16.var, align 8, !dbg !91
// CHECK:STDOUT:   %.loc50_14 = load ptr, ptr %ptr_i32.var, align 8, !dbg !92
// CHECK:STDOUT:   %.loc50_23 = load ptr, ptr %ptr_i64.var, align 8, !dbg !93
// CHECK:STDOUT:   %F.call.loc50 = call i32 @_CF.Main.732da705e71cd359(ptr %.loc50_5, ptr %.loc50_14, ptr %.loc50_23, i32 0), !dbg !94
// CHECK:STDOUT:   %.loc51_5 = load ptr, ptr %ptr_i64.var, align 8, !dbg !95
// CHECK:STDOUT:   %.loc51_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !96
// CHECK:STDOUT:   %.loc51_24 = load ptr, ptr %ptr_i16.var, align 8, !dbg !97
// CHECK:STDOUT:   %F.call.loc51 = call i32 @_CF.Main.732da705e71cd359(ptr %.loc51_5, ptr %.loc51_14, ptr %.loc51_24, i32 0), !dbg !98
// CHECK:STDOUT:   %.loc52_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !99
// CHECK:STDOUT:   %.loc52_15 = load ptr, ptr %ptr_bool.var, align 8, !dbg !100
// CHECK:STDOUT:   %.loc52_25 = load ptr, ptr %ptr_bool.var, align 8, !dbg !101
// CHECK:STDOUT:   %F.call.loc52 = call i32 @_CF.Main.732da705e71cd359(ptr %.loc52_5, ptr %.loc52_15, ptr %.loc52_25, i32 0), !dbg !102
// CHECK:STDOUT:   call void @"_CSelfDestruct.bb2383e5e4785e19:core.Destroy.Core"(ptr %ptr_bool.var), !dbg !43
// CHECK:STDOUT:   call void @"_CSelfDestruct.6438500ec3acb664:core.Destroy.Core"(ptr %ptr_i64.var), !dbg !39
// CHECK:STDOUT:   call void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %ptr_i32.var), !dbg !35
// CHECK:STDOUT:   call void @"_CSelfDestruct.686cf03349556866:core.Destroy.Core"(ptr %ptr_i16.var), !dbg !31
// CHECK:STDOUT:   call void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %val_bool.var), !dbg !29
// CHECK:STDOUT:   call void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %val_f64.var), !dbg !27
// CHECK:STDOUT:   call void @"_CSelfDestruct.5aacfd7dcc330b17:core.Destroy.Core"(ptr %val_i64.var), !dbg !25
// CHECK:STDOUT:   call void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %val_i32.var), !dbg !23
// CHECK:STDOUT:   call void @"_CSelfDestruct.d57cb32fcf12ec07:core.Destroy.Core"(ptr %val_i16.var), !dbg !21
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.bb2383e5e4785e19:core.Destroy.Core"(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.6438500ec3acb664:core.Destroy.Core"(ptr %self) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d088c7eb746488be:core.Destroy.Core"(ptr %self) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.686cf03349556866:core.Destroy.Core"(ptr %self) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core"(ptr %self) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core"(ptr %self) #0 !dbg !124 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !126
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.5aacfd7dcc330b17:core.Destroy.Core"(ptr %self) #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !130
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.f58524831c37807e:core.Destroy.Core"(ptr %self) #0 !dbg !132 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_CSelfDestruct.d57cb32fcf12ec07:core.Destroy.Core"(ptr %self) #0 !dbg !136 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !138
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.bb004feba9dd474d(i16 %x, ptr %y, ptr %z, i32 %count) #0 !dbg !140 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !151
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !152
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !154
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !158
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.bb004feba9dd474d(i16 %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !159
// CHECK:STDOUT:   ret i32 %F.call, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.f24eb6b05368f488(i32 %x, ptr %y, ptr %z, i32 %count) #0 !dbg !162 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !173
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !174
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !176
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !180
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.f24eb6b05368f488(i32 %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !181
// CHECK:STDOUT:   ret i32 %F.call, !dbg !182
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.427cba2b3278d1dc(i1 %x, ptr %y, ptr %z, i32 %count) #0 !dbg !184 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !195
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !196
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !198
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !202
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.427cba2b3278d1dc(i1 %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !203
// CHECK:STDOUT:   ret i32 %F.call, !dbg !204
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.ea32bb7f769e7967(i64 %x, i1 %y, ptr %z, i32 %count) #0 !dbg !206 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !217
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !218
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !220
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !224
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.b1f6aa8c7fcac0ca(i64 %x, ptr %z, i1 %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !225
// CHECK:STDOUT:   ret i32 %F.call, !dbg !226
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.6eab5c6411742e70(double %x, ptr %y, ptr %z, i32 %count) #0 !dbg !228 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !239
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !240
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !242
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !246
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.6eab5c6411742e70(double %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !247
// CHECK:STDOUT:   ret i32 %F.call, !dbg !248
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.732da705e71cd359(ptr %x, ptr %y, ptr %z, i32 %count) #0 !dbg !250 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !261
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !262
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !264
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !268
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.732da705e71cd359(ptr %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !269
// CHECK:STDOUT:   ret i32 %F.call, !dbg !270
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.b1f6aa8c7fcac0ca(i64 %x, ptr %y, i1 %z, i32 %count) #0 !dbg !470 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !481
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !482
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !484
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !488
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.ea32bb7f769e7967(i64 %x, i1 %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !489
// CHECK:STDOUT:   ret i32 %F.call, !dbg !490
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.732da705e71cd359, { 5, 7, 6, 4, 0, 1, 2, 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "call_recursive_reorder_more.carbon", directory: "")
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !4, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !12 = !{null, !11}
// CHECK:STDOUT: !13 = !DISubroutineType(types: !12)
// CHECK:STDOUT: !14 = !{null, !8}
// CHECK:STDOUT: !15 = !DISubroutineType(types: !14)
// CHECK:STDOUT: !16 = !{null, !6}
// CHECK:STDOUT: !17 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !18 = !{null, !7}
// CHECK:STDOUT: !19 = !DISubroutineType(types: !18)
// CHECK:STDOUT: !20 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 22, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !21 = !DILocation(line: 23, column: 3, scope: !20)
// CHECK:STDOUT: !23 = !DILocation(line: 24, column: 3, scope: !20)
// CHECK:STDOUT: !25 = !DILocation(line: 25, column: 3, scope: !20)
// CHECK:STDOUT: !27 = !DILocation(line: 26, column: 3, scope: !20)
// CHECK:STDOUT: !29 = !DILocation(line: 27, column: 3, scope: !20)
// CHECK:STDOUT: !31 = !DILocation(line: 29, column: 3, scope: !20)
// CHECK:STDOUT: !35 = !DILocation(line: 30, column: 3, scope: !20)
// CHECK:STDOUT: !39 = !DILocation(line: 31, column: 3, scope: !20)
// CHECK:STDOUT: !43 = !DILocation(line: 32, column: 3, scope: !20)
// CHECK:STDOUT: !47 = !DILocation(line: 38, column: 5, scope: !20)
// CHECK:STDOUT: !48 = !DILocation(line: 38, column: 14, scope: !20)
// CHECK:STDOUT: !49 = !DILocation(line: 38, column: 23, scope: !20)
// CHECK:STDOUT: !50 = !DILocation(line: 38, column: 3, scope: !20)
// CHECK:STDOUT: !51 = !{!6, !7, !11, !11, !6}
// CHECK:STDOUT: !52 = !DISubroutineType(types: !51)
// CHECK:STDOUT: !53 = !DILocation(line: 39, column: 5, scope: !20)
// CHECK:STDOUT: !54 = !DILocation(line: 39, column: 14, scope: !20)
// CHECK:STDOUT: !55 = !DILocation(line: 39, column: 24, scope: !20)
// CHECK:STDOUT: !56 = !DILocation(line: 39, column: 3, scope: !20)
// CHECK:STDOUT: !57 = !{!6, !6, !11, !11, !6}
// CHECK:STDOUT: !58 = !DISubroutineType(types: !57)
// CHECK:STDOUT: !59 = !DILocation(line: 40, column: 5, scope: !20)
// CHECK:STDOUT: !60 = !DILocation(line: 40, column: 15, scope: !20)
// CHECK:STDOUT: !61 = !DILocation(line: 40, column: 25, scope: !20)
// CHECK:STDOUT: !62 = !DILocation(line: 40, column: 3, scope: !20)
// CHECK:STDOUT: !63 = !{!6, !11, !11, !11, !6}
// CHECK:STDOUT: !64 = !DISubroutineType(types: !63)
// CHECK:STDOUT: !65 = !DILocation(line: 41, column: 5, scope: !20)
// CHECK:STDOUT: !66 = !DILocation(line: 41, column: 14, scope: !20)
// CHECK:STDOUT: !67 = !DILocation(line: 41, column: 24, scope: !20)
// CHECK:STDOUT: !68 = !DILocation(line: 41, column: 3, scope: !20)
// CHECK:STDOUT: !69 = !{!6, !8, !11, !11, !6}
// CHECK:STDOUT: !70 = !DISubroutineType(types: !69)
// CHECK:STDOUT: !71 = !DILocation(line: 42, column: 5, scope: !20)
// CHECK:STDOUT: !72 = !DILocation(line: 42, column: 14, scope: !20)
// CHECK:STDOUT: !73 = !DILocation(line: 42, column: 24, scope: !20)
// CHECK:STDOUT: !74 = !DILocation(line: 42, column: 3, scope: !20)
// CHECK:STDOUT: !75 = !DILocation(line: 46, column: 5, scope: !20)
// CHECK:STDOUT: !76 = !DILocation(line: 46, column: 15, scope: !20)
// CHECK:STDOUT: !77 = !DILocation(line: 46, column: 24, scope: !20)
// CHECK:STDOUT: !78 = !DILocation(line: 46, column: 3, scope: !20)
// CHECK:STDOUT: !79 = !DILocation(line: 47, column: 5, scope: !20)
// CHECK:STDOUT: !80 = !DILocation(line: 47, column: 15, scope: !20)
// CHECK:STDOUT: !81 = !DILocation(line: 47, column: 24, scope: !20)
// CHECK:STDOUT: !82 = !DILocation(line: 47, column: 3, scope: !20)
// CHECK:STDOUT: !83 = !DILocation(line: 48, column: 5, scope: !20)
// CHECK:STDOUT: !84 = !DILocation(line: 48, column: 14, scope: !20)
// CHECK:STDOUT: !85 = !DILocation(line: 48, column: 24, scope: !20)
// CHECK:STDOUT: !86 = !DILocation(line: 48, column: 3, scope: !20)
// CHECK:STDOUT: !87 = !DILocation(line: 49, column: 5, scope: !20)
// CHECK:STDOUT: !88 = !DILocation(line: 49, column: 14, scope: !20)
// CHECK:STDOUT: !89 = !DILocation(line: 49, column: 24, scope: !20)
// CHECK:STDOUT: !90 = !DILocation(line: 49, column: 3, scope: !20)
// CHECK:STDOUT: !91 = !DILocation(line: 50, column: 5, scope: !20)
// CHECK:STDOUT: !92 = !DILocation(line: 50, column: 14, scope: !20)
// CHECK:STDOUT: !93 = !DILocation(line: 50, column: 23, scope: !20)
// CHECK:STDOUT: !94 = !DILocation(line: 50, column: 3, scope: !20)
// CHECK:STDOUT: !95 = !DILocation(line: 51, column: 5, scope: !20)
// CHECK:STDOUT: !96 = !DILocation(line: 51, column: 14, scope: !20)
// CHECK:STDOUT: !97 = !DILocation(line: 51, column: 24, scope: !20)
// CHECK:STDOUT: !98 = !DILocation(line: 51, column: 3, scope: !20)
// CHECK:STDOUT: !99 = !DILocation(line: 52, column: 5, scope: !20)
// CHECK:STDOUT: !100 = !DILocation(line: 52, column: 15, scope: !20)
// CHECK:STDOUT: !101 = !DILocation(line: 52, column: 25, scope: !20)
// CHECK:STDOUT: !102 = !DILocation(line: 52, column: 3, scope: !20)
// CHECK:STDOUT: !103 = !DILocation(line: 22, column: 1, scope: !20)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.bb2383e5e4785e19:core.Destroy.Core", scope: null, file: !4, line: 32, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !107)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !11)
// CHECK:STDOUT: !106 = !DILocation(line: 32, column: 3, scope: !104)
// CHECK:STDOUT: !107 = !{!105}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.6438500ec3acb664:core.Destroy.Core", scope: null, file: !4, line: 31, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !111)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !11)
// CHECK:STDOUT: !110 = !DILocation(line: 31, column: 3, scope: !108)
// CHECK:STDOUT: !111 = !{!109}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d088c7eb746488be:core.Destroy.Core", scope: null, file: !4, line: 30, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !115)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !11)
// CHECK:STDOUT: !114 = !DILocation(line: 30, column: 3, scope: !112)
// CHECK:STDOUT: !115 = !{!113}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.686cf03349556866:core.Destroy.Core", scope: null, file: !4, line: 29, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !119)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !11)
// CHECK:STDOUT: !118 = !DILocation(line: 29, column: 3, scope: !116)
// CHECK:STDOUT: !119 = !{!117}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.48e0adf639fe7af4:core.Destroy.Core", scope: null, file: !4, line: 27, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !123)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !11)
// CHECK:STDOUT: !122 = !DILocation(line: 27, column: 3, scope: !120)
// CHECK:STDOUT: !123 = !{!121}
// CHECK:STDOUT: !124 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.dc14d4dd21801791:core.Destroy.Core", scope: null, file: !4, line: 26, type: !13, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !127)
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !124, type: !11)
// CHECK:STDOUT: !126 = !DILocation(line: 26, column: 3, scope: !124)
// CHECK:STDOUT: !127 = !{!125}
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.5aacfd7dcc330b17:core.Destroy.Core", scope: null, file: !4, line: 25, type: !15, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !131)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 1, scope: !128, type: !8)
// CHECK:STDOUT: !130 = !DILocation(line: 25, column: 3, scope: !128)
// CHECK:STDOUT: !131 = !{!129}
// CHECK:STDOUT: !132 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.f58524831c37807e:core.Destroy.Core", scope: null, file: !4, line: 24, type: !17, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !135)
// CHECK:STDOUT: !133 = !DILocalVariable(arg: 1, scope: !132, type: !6)
// CHECK:STDOUT: !134 = !DILocation(line: 24, column: 3, scope: !132)
// CHECK:STDOUT: !135 = !{!133}
// CHECK:STDOUT: !136 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.d57cb32fcf12ec07:core.Destroy.Core", scope: null, file: !4, line: 23, type: !19, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !139)
// CHECK:STDOUT: !137 = !DILocalVariable(arg: 1, scope: !136, type: !7)
// CHECK:STDOUT: !138 = !DILocation(line: 23, column: 3, scope: !136)
// CHECK:STDOUT: !139 = !{!137}
// CHECK:STDOUT: !140 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.bb004feba9dd474d", scope: null, file: !4, line: 15, type: !52, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !161)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 1, scope: !140, type: !7)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 2, scope: !140, type: !11)
// CHECK:STDOUT: !143 = !DILocalVariable(arg: 3, scope: !140, type: !11)
// CHECK:STDOUT: !144 = !DILocalVariable(arg: 4, scope: !140, type: !6)
// CHECK:STDOUT: !151 = !DILocation(line: 16, column: 7, scope: !140)
// CHECK:STDOUT: !152 = !DILocation(line: 16, column: 6, scope: !140)
// CHECK:STDOUT: !154 = !DILocation(line: 17, column: 5, scope: !140)
// CHECK:STDOUT: !158 = !DILocation(line: 19, column: 21, scope: !140)
// CHECK:STDOUT: !159 = !DILocation(line: 19, column: 10, scope: !140)
// CHECK:STDOUT: !160 = !DILocation(line: 19, column: 3, scope: !140)
// CHECK:STDOUT: !161 = !{!141, !142, !143, !144}
// CHECK:STDOUT: !162 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f24eb6b05368f488", scope: null, file: !4, line: 15, type: !58, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !183)
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !162, type: !6)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 2, scope: !162, type: !11)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 3, scope: !162, type: !11)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 4, scope: !162, type: !6)
// CHECK:STDOUT: !173 = !DILocation(line: 16, column: 7, scope: !162)
// CHECK:STDOUT: !174 = !DILocation(line: 16, column: 6, scope: !162)
// CHECK:STDOUT: !176 = !DILocation(line: 17, column: 5, scope: !162)
// CHECK:STDOUT: !180 = !DILocation(line: 19, column: 21, scope: !162)
// CHECK:STDOUT: !181 = !DILocation(line: 19, column: 10, scope: !162)
// CHECK:STDOUT: !182 = !DILocation(line: 19, column: 3, scope: !162)
// CHECK:STDOUT: !183 = !{!163, !164, !165, !166}
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.427cba2b3278d1dc", scope: null, file: !4, line: 15, type: !64, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !205)
// CHECK:STDOUT: !185 = !DILocalVariable(arg: 1, scope: !184, type: !11)
// CHECK:STDOUT: !186 = !DILocalVariable(arg: 2, scope: !184, type: !11)
// CHECK:STDOUT: !187 = !DILocalVariable(arg: 3, scope: !184, type: !11)
// CHECK:STDOUT: !188 = !DILocalVariable(arg: 4, scope: !184, type: !6)
// CHECK:STDOUT: !195 = !DILocation(line: 16, column: 7, scope: !184)
// CHECK:STDOUT: !196 = !DILocation(line: 16, column: 6, scope: !184)
// CHECK:STDOUT: !198 = !DILocation(line: 17, column: 5, scope: !184)
// CHECK:STDOUT: !202 = !DILocation(line: 19, column: 21, scope: !184)
// CHECK:STDOUT: !203 = !DILocation(line: 19, column: 10, scope: !184)
// CHECK:STDOUT: !204 = !DILocation(line: 19, column: 3, scope: !184)
// CHECK:STDOUT: !205 = !{!185, !186, !187, !188}
// CHECK:STDOUT: !206 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.ea32bb7f769e7967", scope: null, file: !4, line: 15, type: !70, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !227)
// CHECK:STDOUT: !207 = !DILocalVariable(arg: 1, scope: !206, type: !8)
// CHECK:STDOUT: !208 = !DILocalVariable(arg: 2, scope: !206, type: !11)
// CHECK:STDOUT: !209 = !DILocalVariable(arg: 3, scope: !206, type: !11)
// CHECK:STDOUT: !210 = !DILocalVariable(arg: 4, scope: !206, type: !6)
// CHECK:STDOUT: !217 = !DILocation(line: 16, column: 7, scope: !206)
// CHECK:STDOUT: !218 = !DILocation(line: 16, column: 6, scope: !206)
// CHECK:STDOUT: !220 = !DILocation(line: 17, column: 5, scope: !206)
// CHECK:STDOUT: !224 = !DILocation(line: 19, column: 21, scope: !206)
// CHECK:STDOUT: !225 = !DILocation(line: 19, column: 10, scope: !206)
// CHECK:STDOUT: !226 = !DILocation(line: 19, column: 3, scope: !206)
// CHECK:STDOUT: !227 = !{!207, !208, !209, !210}
// CHECK:STDOUT: !228 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.6eab5c6411742e70", scope: null, file: !4, line: 15, type: !64, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !249)
// CHECK:STDOUT: !229 = !DILocalVariable(arg: 1, scope: !228, type: !11)
// CHECK:STDOUT: !230 = !DILocalVariable(arg: 2, scope: !228, type: !11)
// CHECK:STDOUT: !231 = !DILocalVariable(arg: 3, scope: !228, type: !11)
// CHECK:STDOUT: !232 = !DILocalVariable(arg: 4, scope: !228, type: !6)
// CHECK:STDOUT: !239 = !DILocation(line: 16, column: 7, scope: !228)
// CHECK:STDOUT: !240 = !DILocation(line: 16, column: 6, scope: !228)
// CHECK:STDOUT: !242 = !DILocation(line: 17, column: 5, scope: !228)
// CHECK:STDOUT: !246 = !DILocation(line: 19, column: 21, scope: !228)
// CHECK:STDOUT: !247 = !DILocation(line: 19, column: 10, scope: !228)
// CHECK:STDOUT: !248 = !DILocation(line: 19, column: 3, scope: !228)
// CHECK:STDOUT: !249 = !{!229, !230, !231, !232}
// CHECK:STDOUT: !250 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.732da705e71cd359", scope: null, file: !4, line: 15, type: !64, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !271)
// CHECK:STDOUT: !251 = !DILocalVariable(arg: 1, scope: !250, type: !11)
// CHECK:STDOUT: !252 = !DILocalVariable(arg: 2, scope: !250, type: !11)
// CHECK:STDOUT: !253 = !DILocalVariable(arg: 3, scope: !250, type: !11)
// CHECK:STDOUT: !254 = !DILocalVariable(arg: 4, scope: !250, type: !6)
// CHECK:STDOUT: !261 = !DILocation(line: 16, column: 7, scope: !250)
// CHECK:STDOUT: !262 = !DILocation(line: 16, column: 6, scope: !250)
// CHECK:STDOUT: !264 = !DILocation(line: 17, column: 5, scope: !250)
// CHECK:STDOUT: !268 = !DILocation(line: 19, column: 21, scope: !250)
// CHECK:STDOUT: !269 = !DILocation(line: 19, column: 10, scope: !250)
// CHECK:STDOUT: !270 = !DILocation(line: 19, column: 3, scope: !250)
// CHECK:STDOUT: !271 = !{!251, !252, !253, !254}
// CHECK:STDOUT: !470 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b1f6aa8c7fcac0ca", scope: null, file: !4, line: 15, type: !70, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !491)
// CHECK:STDOUT: !471 = !DILocalVariable(arg: 1, scope: !470, type: !8)
// CHECK:STDOUT: !472 = !DILocalVariable(arg: 2, scope: !470, type: !11)
// CHECK:STDOUT: !473 = !DILocalVariable(arg: 3, scope: !470, type: !11)
// CHECK:STDOUT: !474 = !DILocalVariable(arg: 4, scope: !470, type: !6)
// CHECK:STDOUT: !481 = !DILocation(line: 16, column: 7, scope: !470)
// CHECK:STDOUT: !482 = !DILocation(line: 16, column: 6, scope: !470)
// CHECK:STDOUT: !484 = !DILocation(line: 17, column: 5, scope: !470)
// CHECK:STDOUT: !488 = !DILocation(line: 19, column: 21, scope: !470)
// CHECK:STDOUT: !489 = !DILocation(line: 19, column: 10, scope: !470)
// CHECK:STDOUT: !490 = !DILocation(line: 19, column: 3, scope: !470)
// CHECK:STDOUT: !491 = !{!471, !472, !473, !474}
// CHECK:STDOUT:
