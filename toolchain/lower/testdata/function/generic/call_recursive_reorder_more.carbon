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
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %val_i16.var = alloca i16, align 2, !dbg !12
// CHECK:STDOUT:   %val_i32.var = alloca i32, align 4, !dbg !14
// CHECK:STDOUT:   %val_i64.var = alloca i64, align 8, !dbg !16
// CHECK:STDOUT:   %val_f64.var = alloca double, align 8, !dbg !18
// CHECK:STDOUT:   %val_bool.var = alloca i1, align 1, !dbg !20
// CHECK:STDOUT:   %ptr_i16.var = alloca ptr, align 8, !dbg !22
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !26
// CHECK:STDOUT:   %ptr_i64.var = alloca ptr, align 8, !dbg !30
// CHECK:STDOUT:   %ptr_bool.var = alloca ptr, align 8, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_i16.var), !dbg !12
// CHECK:STDOUT:   store i16 poison, ptr %val_i16.var, align 2, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_i32.var), !dbg !14
// CHECK:STDOUT:   store i32 poison, ptr %val_i32.var, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_i64.var), !dbg !16
// CHECK:STDOUT:   store i64 poison, ptr %val_i64.var, align 8, !dbg !16
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_f64.var), !dbg !18
// CHECK:STDOUT:   store double poison, ptr %val_f64.var, align 8, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_bool.var), !dbg !20
// CHECK:STDOUT:   store i8 poison, ptr %val_bool.var, align 1, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i16.var), !dbg !22
// CHECK:STDOUT:   store ptr %val_i16.var, ptr %ptr_i16.var, align 8, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !26
// CHECK:STDOUT:   store ptr %val_i32.var, ptr %ptr_i32.var, align 8, !dbg !26
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i64.var), !dbg !30
// CHECK:STDOUT:   store ptr %val_i64.var, ptr %ptr_i64.var, align 8, !dbg !30
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_bool.var), !dbg !34
// CHECK:STDOUT:   store ptr %val_bool.var, ptr %ptr_bool.var, align 8, !dbg !34
// CHECK:STDOUT:   %.loc38_5 = load i16, ptr %val_i16.var, align 2, !dbg !38
// CHECK:STDOUT:   %.loc38_14 = load ptr, ptr %ptr_i32.var, align 8, !dbg !39
// CHECK:STDOUT:   %.loc38_23 = load ptr, ptr %ptr_bool.var, align 8, !dbg !40
// CHECK:STDOUT:   %F.call.loc38 = call i32 @_CF.Main.5b94c215f01213ad(i16 %.loc38_5, ptr %.loc38_14, ptr %.loc38_23, i32 0), !dbg !41
// CHECK:STDOUT:   %.loc39_5 = load i32, ptr %val_i32.var, align 4, !dbg !45
// CHECK:STDOUT:   %.loc39_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !46
// CHECK:STDOUT:   %.loc39_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !47
// CHECK:STDOUT:   %F.call.loc39 = call i32 @_CF.Main.54596334ff7c3bae(i32 %.loc39_5, ptr %.loc39_14, ptr %.loc39_24, i32 0), !dbg !48
// CHECK:STDOUT:   %.loc40_5 = load i8, ptr %val_bool.var, align 1, !dbg !51
// CHECK:STDOUT:   %.loc40_51 = trunc i8 %.loc40_5 to i1, !dbg !51
// CHECK:STDOUT:   %.loc40_15 = load ptr, ptr %ptr_bool.var, align 8, !dbg !52
// CHECK:STDOUT:   %.loc40_25 = load ptr, ptr %ptr_i16.var, align 8, !dbg !53
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.17ab41db4fbdd09e(i1 %.loc40_51, ptr %.loc40_15, ptr %.loc40_25, i32 0), !dbg !54
// CHECK:STDOUT:   %.loc41_5 = load i64, ptr %val_i64.var, align 8, !dbg !57
// CHECK:STDOUT:   %.loc41_14 = load i8, ptr %val_bool.var, align 1, !dbg !58
// CHECK:STDOUT:   %.loc41_142 = trunc i8 %.loc41_14 to i1, !dbg !58
// CHECK:STDOUT:   %.loc41_24 = load ptr, ptr %ptr_bool.var, align 8, !dbg !59
// CHECK:STDOUT:   %F.call.loc41 = call i32 @_CF.Main.4870fabf59aefef9(i64 %.loc41_5, i1 %.loc41_142, ptr %.loc41_24, i32 0), !dbg !60
// CHECK:STDOUT:   %.loc42_5 = load double, ptr %val_f64.var, align 8, !dbg !63
// CHECK:STDOUT:   %.loc42_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !64
// CHECK:STDOUT:   %.loc42_24 = load ptr, ptr %ptr_bool.var, align 8, !dbg !65
// CHECK:STDOUT:   %F.call.loc42 = call i32 @_CF.Main.e20e13691da7edc2(double %.loc42_5, ptr %.loc42_14, ptr %.loc42_24, i32 0), !dbg !66
// CHECK:STDOUT:   %.loc46_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !67
// CHECK:STDOUT:   %.loc46_15 = load ptr, ptr %ptr_i16.var, align 8, !dbg !68
// CHECK:STDOUT:   %.loc46_24 = load ptr, ptr %ptr_i32.var, align 8, !dbg !69
// CHECK:STDOUT:   %F.call.loc46 = call i32 @_CF.Main.1a6874a49ac98b64(ptr %.loc46_5, ptr %.loc46_15, ptr %.loc46_24, i32 0), !dbg !70
// CHECK:STDOUT:   %.loc47_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !71
// CHECK:STDOUT:   %.loc47_15 = load ptr, ptr %ptr_i16.var, align 8, !dbg !72
// CHECK:STDOUT:   %.loc47_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !73
// CHECK:STDOUT:   %F.call.loc47 = call i32 @_CF.Main.1a6874a49ac98b64(ptr %.loc47_5, ptr %.loc47_15, ptr %.loc47_24, i32 0), !dbg !74
// CHECK:STDOUT:   %.loc48_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !75
// CHECK:STDOUT:   %.loc48_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !76
// CHECK:STDOUT:   %.loc48_24 = load ptr, ptr %ptr_i16.var, align 8, !dbg !77
// CHECK:STDOUT:   %F.call.loc48 = call i32 @_CF.Main.1a6874a49ac98b64(ptr %.loc48_5, ptr %.loc48_14, ptr %.loc48_24, i32 0), !dbg !78
// CHECK:STDOUT:   %.loc49_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !79
// CHECK:STDOUT:   %.loc49_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !80
// CHECK:STDOUT:   %.loc49_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !81
// CHECK:STDOUT:   %F.call.loc49 = call i32 @_CF.Main.1a6874a49ac98b64(ptr %.loc49_5, ptr %.loc49_14, ptr %.loc49_24, i32 0), !dbg !82
// CHECK:STDOUT:   %.loc50_5 = load ptr, ptr %ptr_i16.var, align 8, !dbg !83
// CHECK:STDOUT:   %.loc50_14 = load ptr, ptr %ptr_i32.var, align 8, !dbg !84
// CHECK:STDOUT:   %.loc50_23 = load ptr, ptr %ptr_i64.var, align 8, !dbg !85
// CHECK:STDOUT:   %F.call.loc50 = call i32 @_CF.Main.1a6874a49ac98b64(ptr %.loc50_5, ptr %.loc50_14, ptr %.loc50_23, i32 0), !dbg !86
// CHECK:STDOUT:   %.loc51_5 = load ptr, ptr %ptr_i64.var, align 8, !dbg !87
// CHECK:STDOUT:   %.loc51_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !88
// CHECK:STDOUT:   %.loc51_24 = load ptr, ptr %ptr_i16.var, align 8, !dbg !89
// CHECK:STDOUT:   %F.call.loc51 = call i32 @_CF.Main.1a6874a49ac98b64(ptr %.loc51_5, ptr %.loc51_14, ptr %.loc51_24, i32 0), !dbg !90
// CHECK:STDOUT:   %.loc52_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !91
// CHECK:STDOUT:   %.loc52_15 = load ptr, ptr %ptr_bool.var, align 8, !dbg !92
// CHECK:STDOUT:   %.loc52_25 = load ptr, ptr %ptr_bool.var, align 8, !dbg !93
// CHECK:STDOUT:   %F.call.loc52 = call i32 @_CF.Main.1a6874a49ac98b64(ptr %.loc52_5, ptr %.loc52_15, ptr %.loc52_25, i32 0), !dbg !94
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.393ac80c0c8a44df(ptr %ptr_bool.var), !dbg !34
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.ca462fb0865f0b4d(ptr %ptr_i64.var), !dbg !30
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %ptr_i32.var), !dbg !26
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.a2001b647b825e99(ptr %ptr_i16.var), !dbg !22
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %val_bool.var), !dbg !20
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %val_f64.var), !dbg !18
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.7e9df21fc3951ad1(ptr %val_i64.var), !dbg !16
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %val_i32.var), !dbg !14
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.e43967308dd20a9f(ptr %val_i16.var), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.5b94c215f01213ad(i16 %x, ptr %y, ptr %z, i32 %count) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !115
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !116
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !118
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !122
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.5b94c215f01213ad(i16 %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !123
// CHECK:STDOUT:   ret i32 %F.call, !dbg !124
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.54596334ff7c3bae(i32 %x, ptr %y, ptr %z, i32 %count) #0 !dbg !126 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !137
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !138
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !140
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !144
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.54596334ff7c3bae(i32 %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !145
// CHECK:STDOUT:   ret i32 %F.call, !dbg !146
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.17ab41db4fbdd09e(i1 %x, ptr %y, ptr %z, i32 %count) #0 !dbg !148 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !159
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !160
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !162
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !166
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.17ab41db4fbdd09e(i1 %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !167
// CHECK:STDOUT:   ret i32 %F.call, !dbg !168
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4870fabf59aefef9(i64 %x, i1 %y, ptr %z, i32 %count) #0 !dbg !170 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !181
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !182
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !184
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !188
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.9687df6c12a8eb01(i64 %x, ptr %z, i1 %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !189
// CHECK:STDOUT:   ret i32 %F.call, !dbg !190
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.e20e13691da7edc2(double %x, ptr %y, ptr %z, i32 %count) #0 !dbg !192 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !203
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !204
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !206
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !210
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.e20e13691da7edc2(double %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !211
// CHECK:STDOUT:   ret i32 %F.call, !dbg !212
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.1a6874a49ac98b64(ptr %x, ptr %y, ptr %z, i32 %count) #0 !dbg !214 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !225
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !226
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !228
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !232
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.1a6874a49ac98b64(ptr %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !233
// CHECK:STDOUT:   ret i32 %F.call, !dbg !234
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.393ac80c0c8a44df(ptr %t) #0 !dbg !369 {
// CHECK:STDOUT:   ret void, !dbg !374
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.ca462fb0865f0b4d(ptr %t) #0 !dbg !376 {
// CHECK:STDOUT:   ret void, !dbg !381
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #0 !dbg !383 {
// CHECK:STDOUT:   ret void, !dbg !388
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.a2001b647b825e99(ptr %t) #0 !dbg !390 {
// CHECK:STDOUT:   ret void, !dbg !395
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.29e6aa423646b61a(ptr %t) #0 !dbg !397 {
// CHECK:STDOUT:   ret void, !dbg !402
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.2b38c2ca9f7f1a94(ptr %t) #0 !dbg !404 {
// CHECK:STDOUT:   ret void, !dbg !409
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.7e9df21fc3951ad1(ptr %t) #0 !dbg !411 {
// CHECK:STDOUT:   ret void, !dbg !416
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !418 {
// CHECK:STDOUT:   ret void, !dbg !423
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.e43967308dd20a9f(ptr %t) #0 !dbg !425 {
// CHECK:STDOUT:   ret void, !dbg !430
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.9687df6c12a8eb01(i64 %x, ptr %y, i1 %z, i32 %count) #0 !dbg !498 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !509
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !510
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !512
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !516
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4870fabf59aefef9(i64 %x, i1 %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !517
// CHECK:STDOUT:   ret i32 %F.call, !dbg !518
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.1a6874a49ac98b64, { 5, 7, 6, 4, 0, 1, 2, 3 }
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
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !4, line: 22, type: !10, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !12 = !DILocation(line: 23, column: 3, scope: !11)
// CHECK:STDOUT: !14 = !DILocation(line: 24, column: 3, scope: !11)
// CHECK:STDOUT: !16 = !DILocation(line: 25, column: 3, scope: !11)
// CHECK:STDOUT: !18 = !DILocation(line: 26, column: 3, scope: !11)
// CHECK:STDOUT: !20 = !DILocation(line: 27, column: 3, scope: !11)
// CHECK:STDOUT: !22 = !DILocation(line: 29, column: 3, scope: !11)
// CHECK:STDOUT: !26 = !DILocation(line: 30, column: 3, scope: !11)
// CHECK:STDOUT: !30 = !DILocation(line: 31, column: 3, scope: !11)
// CHECK:STDOUT: !34 = !DILocation(line: 32, column: 3, scope: !11)
// CHECK:STDOUT: !38 = !DILocation(line: 38, column: 5, scope: !11)
// CHECK:STDOUT: !39 = !DILocation(line: 38, column: 14, scope: !11)
// CHECK:STDOUT: !40 = !DILocation(line: 38, column: 23, scope: !11)
// CHECK:STDOUT: !41 = !DILocation(line: 38, column: 3, scope: !11)
// CHECK:STDOUT: !42 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !43 = !{!6, !7, !42, !42, !6}
// CHECK:STDOUT: !44 = !DISubroutineType(types: !43)
// CHECK:STDOUT: !45 = !DILocation(line: 39, column: 5, scope: !11)
// CHECK:STDOUT: !46 = !DILocation(line: 39, column: 14, scope: !11)
// CHECK:STDOUT: !47 = !DILocation(line: 39, column: 24, scope: !11)
// CHECK:STDOUT: !48 = !DILocation(line: 39, column: 3, scope: !11)
// CHECK:STDOUT: !49 = !{!6, !6, !42, !42, !6}
// CHECK:STDOUT: !50 = !DISubroutineType(types: !49)
// CHECK:STDOUT: !51 = !DILocation(line: 40, column: 5, scope: !11)
// CHECK:STDOUT: !52 = !DILocation(line: 40, column: 15, scope: !11)
// CHECK:STDOUT: !53 = !DILocation(line: 40, column: 25, scope: !11)
// CHECK:STDOUT: !54 = !DILocation(line: 40, column: 3, scope: !11)
// CHECK:STDOUT: !55 = !{!6, !42, !42, !42, !6}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !57 = !DILocation(line: 41, column: 5, scope: !11)
// CHECK:STDOUT: !58 = !DILocation(line: 41, column: 14, scope: !11)
// CHECK:STDOUT: !59 = !DILocation(line: 41, column: 24, scope: !11)
// CHECK:STDOUT: !60 = !DILocation(line: 41, column: 3, scope: !11)
// CHECK:STDOUT: !61 = !{!6, !8, !42, !42, !6}
// CHECK:STDOUT: !62 = !DISubroutineType(types: !61)
// CHECK:STDOUT: !63 = !DILocation(line: 42, column: 5, scope: !11)
// CHECK:STDOUT: !64 = !DILocation(line: 42, column: 14, scope: !11)
// CHECK:STDOUT: !65 = !DILocation(line: 42, column: 24, scope: !11)
// CHECK:STDOUT: !66 = !DILocation(line: 42, column: 3, scope: !11)
// CHECK:STDOUT: !67 = !DILocation(line: 46, column: 5, scope: !11)
// CHECK:STDOUT: !68 = !DILocation(line: 46, column: 15, scope: !11)
// CHECK:STDOUT: !69 = !DILocation(line: 46, column: 24, scope: !11)
// CHECK:STDOUT: !70 = !DILocation(line: 46, column: 3, scope: !11)
// CHECK:STDOUT: !71 = !DILocation(line: 47, column: 5, scope: !11)
// CHECK:STDOUT: !72 = !DILocation(line: 47, column: 15, scope: !11)
// CHECK:STDOUT: !73 = !DILocation(line: 47, column: 24, scope: !11)
// CHECK:STDOUT: !74 = !DILocation(line: 47, column: 3, scope: !11)
// CHECK:STDOUT: !75 = !DILocation(line: 48, column: 5, scope: !11)
// CHECK:STDOUT: !76 = !DILocation(line: 48, column: 14, scope: !11)
// CHECK:STDOUT: !77 = !DILocation(line: 48, column: 24, scope: !11)
// CHECK:STDOUT: !78 = !DILocation(line: 48, column: 3, scope: !11)
// CHECK:STDOUT: !79 = !DILocation(line: 49, column: 5, scope: !11)
// CHECK:STDOUT: !80 = !DILocation(line: 49, column: 14, scope: !11)
// CHECK:STDOUT: !81 = !DILocation(line: 49, column: 24, scope: !11)
// CHECK:STDOUT: !82 = !DILocation(line: 49, column: 3, scope: !11)
// CHECK:STDOUT: !83 = !DILocation(line: 50, column: 5, scope: !11)
// CHECK:STDOUT: !84 = !DILocation(line: 50, column: 14, scope: !11)
// CHECK:STDOUT: !85 = !DILocation(line: 50, column: 23, scope: !11)
// CHECK:STDOUT: !86 = !DILocation(line: 50, column: 3, scope: !11)
// CHECK:STDOUT: !87 = !DILocation(line: 51, column: 5, scope: !11)
// CHECK:STDOUT: !88 = !DILocation(line: 51, column: 14, scope: !11)
// CHECK:STDOUT: !89 = !DILocation(line: 51, column: 24, scope: !11)
// CHECK:STDOUT: !90 = !DILocation(line: 51, column: 3, scope: !11)
// CHECK:STDOUT: !91 = !DILocation(line: 52, column: 5, scope: !11)
// CHECK:STDOUT: !92 = !DILocation(line: 52, column: 15, scope: !11)
// CHECK:STDOUT: !93 = !DILocation(line: 52, column: 25, scope: !11)
// CHECK:STDOUT: !94 = !DILocation(line: 52, column: 3, scope: !11)
// CHECK:STDOUT: !95 = !{null, !42}
// CHECK:STDOUT: !96 = !DISubroutineType(types: !95)
// CHECK:STDOUT: !97 = !{null, !8}
// CHECK:STDOUT: !98 = !DISubroutineType(types: !97)
// CHECK:STDOUT: !99 = !{null, !6}
// CHECK:STDOUT: !100 = !DISubroutineType(types: !99)
// CHECK:STDOUT: !101 = !{null, !7}
// CHECK:STDOUT: !102 = !DISubroutineType(types: !101)
// CHECK:STDOUT: !103 = !DILocation(line: 22, column: 1, scope: !11)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.5b94c215f01213ad", scope: null, file: !4, line: 15, type: !44, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !125)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !7)
// CHECK:STDOUT: !106 = !DILocalVariable(arg: 2, scope: !104, type: !42)
// CHECK:STDOUT: !107 = !DILocalVariable(arg: 3, scope: !104, type: !42)
// CHECK:STDOUT: !108 = !DILocalVariable(arg: 4, scope: !104, type: !6)
// CHECK:STDOUT: !115 = !DILocation(line: 16, column: 7, scope: !104)
// CHECK:STDOUT: !116 = !DILocation(line: 16, column: 6, scope: !104)
// CHECK:STDOUT: !118 = !DILocation(line: 17, column: 5, scope: !104)
// CHECK:STDOUT: !122 = !DILocation(line: 19, column: 21, scope: !104)
// CHECK:STDOUT: !123 = !DILocation(line: 19, column: 10, scope: !104)
// CHECK:STDOUT: !124 = !DILocation(line: 19, column: 3, scope: !104)
// CHECK:STDOUT: !125 = !{!105, !106, !107, !108}
// CHECK:STDOUT: !126 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.54596334ff7c3bae", scope: null, file: !4, line: 15, type: !50, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !147)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 1, scope: !126, type: !6)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 2, scope: !126, type: !42)
// CHECK:STDOUT: !129 = !DILocalVariable(arg: 3, scope: !126, type: !42)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 4, scope: !126, type: !6)
// CHECK:STDOUT: !137 = !DILocation(line: 16, column: 7, scope: !126)
// CHECK:STDOUT: !138 = !DILocation(line: 16, column: 6, scope: !126)
// CHECK:STDOUT: !140 = !DILocation(line: 17, column: 5, scope: !126)
// CHECK:STDOUT: !144 = !DILocation(line: 19, column: 21, scope: !126)
// CHECK:STDOUT: !145 = !DILocation(line: 19, column: 10, scope: !126)
// CHECK:STDOUT: !146 = !DILocation(line: 19, column: 3, scope: !126)
// CHECK:STDOUT: !147 = !{!127, !128, !129, !130}
// CHECK:STDOUT: !148 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.17ab41db4fbdd09e", scope: null, file: !4, line: 15, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !169)
// CHECK:STDOUT: !149 = !DILocalVariable(arg: 1, scope: !148, type: !42)
// CHECK:STDOUT: !150 = !DILocalVariable(arg: 2, scope: !148, type: !42)
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 3, scope: !148, type: !42)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 4, scope: !148, type: !6)
// CHECK:STDOUT: !159 = !DILocation(line: 16, column: 7, scope: !148)
// CHECK:STDOUT: !160 = !DILocation(line: 16, column: 6, scope: !148)
// CHECK:STDOUT: !162 = !DILocation(line: 17, column: 5, scope: !148)
// CHECK:STDOUT: !166 = !DILocation(line: 19, column: 21, scope: !148)
// CHECK:STDOUT: !167 = !DILocation(line: 19, column: 10, scope: !148)
// CHECK:STDOUT: !168 = !DILocation(line: 19, column: 3, scope: !148)
// CHECK:STDOUT: !169 = !{!149, !150, !151, !152}
// CHECK:STDOUT: !170 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4870fabf59aefef9", scope: null, file: !4, line: 15, type: !62, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !191)
// CHECK:STDOUT: !171 = !DILocalVariable(arg: 1, scope: !170, type: !8)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 2, scope: !170, type: !42)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 3, scope: !170, type: !42)
// CHECK:STDOUT: !174 = !DILocalVariable(arg: 4, scope: !170, type: !6)
// CHECK:STDOUT: !181 = !DILocation(line: 16, column: 7, scope: !170)
// CHECK:STDOUT: !182 = !DILocation(line: 16, column: 6, scope: !170)
// CHECK:STDOUT: !184 = !DILocation(line: 17, column: 5, scope: !170)
// CHECK:STDOUT: !188 = !DILocation(line: 19, column: 21, scope: !170)
// CHECK:STDOUT: !189 = !DILocation(line: 19, column: 10, scope: !170)
// CHECK:STDOUT: !190 = !DILocation(line: 19, column: 3, scope: !170)
// CHECK:STDOUT: !191 = !{!171, !172, !173, !174}
// CHECK:STDOUT: !192 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.e20e13691da7edc2", scope: null, file: !4, line: 15, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !213)
// CHECK:STDOUT: !193 = !DILocalVariable(arg: 1, scope: !192, type: !42)
// CHECK:STDOUT: !194 = !DILocalVariable(arg: 2, scope: !192, type: !42)
// CHECK:STDOUT: !195 = !DILocalVariable(arg: 3, scope: !192, type: !42)
// CHECK:STDOUT: !196 = !DILocalVariable(arg: 4, scope: !192, type: !6)
// CHECK:STDOUT: !203 = !DILocation(line: 16, column: 7, scope: !192)
// CHECK:STDOUT: !204 = !DILocation(line: 16, column: 6, scope: !192)
// CHECK:STDOUT: !206 = !DILocation(line: 17, column: 5, scope: !192)
// CHECK:STDOUT: !210 = !DILocation(line: 19, column: 21, scope: !192)
// CHECK:STDOUT: !211 = !DILocation(line: 19, column: 10, scope: !192)
// CHECK:STDOUT: !212 = !DILocation(line: 19, column: 3, scope: !192)
// CHECK:STDOUT: !213 = !{!193, !194, !195, !196}
// CHECK:STDOUT: !214 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.1a6874a49ac98b64", scope: null, file: !4, line: 15, type: !56, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !235)
// CHECK:STDOUT: !215 = !DILocalVariable(arg: 1, scope: !214, type: !42)
// CHECK:STDOUT: !216 = !DILocalVariable(arg: 2, scope: !214, type: !42)
// CHECK:STDOUT: !217 = !DILocalVariable(arg: 3, scope: !214, type: !42)
// CHECK:STDOUT: !218 = !DILocalVariable(arg: 4, scope: !214, type: !6)
// CHECK:STDOUT: !225 = !DILocation(line: 16, column: 7, scope: !214)
// CHECK:STDOUT: !226 = !DILocation(line: 16, column: 6, scope: !214)
// CHECK:STDOUT: !228 = !DILocation(line: 17, column: 5, scope: !214)
// CHECK:STDOUT: !232 = !DILocation(line: 19, column: 21, scope: !214)
// CHECK:STDOUT: !233 = !DILocation(line: 19, column: 10, scope: !214)
// CHECK:STDOUT: !234 = !DILocation(line: 19, column: 3, scope: !214)
// CHECK:STDOUT: !235 = !{!215, !216, !217, !218}
// CHECK:STDOUT: !368 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !369 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.393ac80c0c8a44df", scope: null, file: !368, line: 35, type: !96, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !375)
// CHECK:STDOUT: !370 = !DILocalVariable(arg: 1, scope: !369, type: !42)
// CHECK:STDOUT: !374 = !DILocation(line: 35, column: 1, scope: !369)
// CHECK:STDOUT: !375 = !{!370}
// CHECK:STDOUT: !376 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.ca462fb0865f0b4d", scope: null, file: !368, line: 35, type: !96, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !382)
// CHECK:STDOUT: !377 = !DILocalVariable(arg: 1, scope: !376, type: !42)
// CHECK:STDOUT: !381 = !DILocation(line: 35, column: 1, scope: !376)
// CHECK:STDOUT: !382 = !{!377}
// CHECK:STDOUT: !383 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !368, line: 35, type: !96, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !389)
// CHECK:STDOUT: !384 = !DILocalVariable(arg: 1, scope: !383, type: !42)
// CHECK:STDOUT: !388 = !DILocation(line: 35, column: 1, scope: !383)
// CHECK:STDOUT: !389 = !{!384}
// CHECK:STDOUT: !390 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.a2001b647b825e99", scope: null, file: !368, line: 35, type: !96, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !396)
// CHECK:STDOUT: !391 = !DILocalVariable(arg: 1, scope: !390, type: !42)
// CHECK:STDOUT: !395 = !DILocation(line: 35, column: 1, scope: !390)
// CHECK:STDOUT: !396 = !{!391}
// CHECK:STDOUT: !397 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.29e6aa423646b61a", scope: null, file: !368, line: 35, type: !96, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !403)
// CHECK:STDOUT: !398 = !DILocalVariable(arg: 1, scope: !397, type: !42)
// CHECK:STDOUT: !402 = !DILocation(line: 35, column: 1, scope: !397)
// CHECK:STDOUT: !403 = !{!398}
// CHECK:STDOUT: !404 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.2b38c2ca9f7f1a94", scope: null, file: !368, line: 35, type: !96, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !410)
// CHECK:STDOUT: !405 = !DILocalVariable(arg: 1, scope: !404, type: !42)
// CHECK:STDOUT: !409 = !DILocation(line: 35, column: 1, scope: !404)
// CHECK:STDOUT: !410 = !{!405}
// CHECK:STDOUT: !411 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.7e9df21fc3951ad1", scope: null, file: !368, line: 35, type: !98, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !417)
// CHECK:STDOUT: !412 = !DILocalVariable(arg: 1, scope: !411, type: !8)
// CHECK:STDOUT: !416 = !DILocation(line: 35, column: 1, scope: !411)
// CHECK:STDOUT: !417 = !{!412}
// CHECK:STDOUT: !418 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !368, line: 35, type: !100, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !424)
// CHECK:STDOUT: !419 = !DILocalVariable(arg: 1, scope: !418, type: !6)
// CHECK:STDOUT: !423 = !DILocation(line: 35, column: 1, scope: !418)
// CHECK:STDOUT: !424 = !{!419}
// CHECK:STDOUT: !425 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.e43967308dd20a9f", scope: null, file: !368, line: 35, type: !102, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !431)
// CHECK:STDOUT: !426 = !DILocalVariable(arg: 1, scope: !425, type: !7)
// CHECK:STDOUT: !430 = !DILocation(line: 35, column: 1, scope: !425)
// CHECK:STDOUT: !431 = !{!426}
// CHECK:STDOUT: !498 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9687df6c12a8eb01", scope: null, file: !4, line: 15, type: !62, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !519)
// CHECK:STDOUT: !499 = !DILocalVariable(arg: 1, scope: !498, type: !8)
// CHECK:STDOUT: !500 = !DILocalVariable(arg: 2, scope: !498, type: !42)
// CHECK:STDOUT: !501 = !DILocalVariable(arg: 3, scope: !498, type: !42)
// CHECK:STDOUT: !502 = !DILocalVariable(arg: 4, scope: !498, type: !6)
// CHECK:STDOUT: !509 = !DILocation(line: 16, column: 7, scope: !498)
// CHECK:STDOUT: !510 = !DILocation(line: 16, column: 6, scope: !498)
// CHECK:STDOUT: !512 = !DILocation(line: 17, column: 5, scope: !498)
// CHECK:STDOUT: !516 = !DILocation(line: 19, column: 21, scope: !498)
// CHECK:STDOUT: !517 = !DILocation(line: 19, column: 10, scope: !498)
// CHECK:STDOUT: !518 = !DILocation(line: 19, column: 3, scope: !498)
// CHECK:STDOUT: !519 = !{!499, !500, !501, !502}
// CHECK:STDOUT:
