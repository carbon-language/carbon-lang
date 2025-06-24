// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_reorder_more.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_reorder_more.carbon

// Builds on `call_recursive_reorder.carbon`.

fn F[T:! type, U:! type, V:! type](x: T, y: U, z: V, count: i32) -> i32 {
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

  var ptr_i16 : i16*;
  var ptr_i32 : i32*;
  var ptr_i64 : i64*;
  var ptr_bool : bool*;

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

// CHECK:STDOUT: ; ModuleID = 'call_recursive_reorder_more.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_reorder_more.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CM.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %val_i16.var = alloca i16, align 2, !dbg !7
// CHECK:STDOUT:   %val_i32.var = alloca i32, align 4, !dbg !8
// CHECK:STDOUT:   %val_i64.var = alloca i64, align 8, !dbg !9
// CHECK:STDOUT:   %val_f64.var = alloca double, align 8, !dbg !10
// CHECK:STDOUT:   %val_bool.var = alloca i1, align 1, !dbg !11
// CHECK:STDOUT:   %ptr_i16.var = alloca ptr, align 8, !dbg !12
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !13
// CHECK:STDOUT:   %ptr_i64.var = alloca ptr, align 8, !dbg !14
// CHECK:STDOUT:   %ptr_bool.var = alloca ptr, align 8, !dbg !15
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 2, ptr %val_i16.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %val_i32.var), !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %val_i64.var), !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %val_f64.var), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 1, ptr %val_bool.var), !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i16.var), !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i32.var), !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i64.var), !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_bool.var), !dbg !15
// CHECK:STDOUT:   %.loc36_5 = load i16, ptr %val_i16.var, align 2, !dbg !16
// CHECK:STDOUT:   %.loc36_14 = load ptr, ptr %ptr_i32.var, align 8, !dbg !17
// CHECK:STDOUT:   %.loc36_23 = load ptr, ptr %ptr_bool.var, align 8, !dbg !18
// CHECK:STDOUT:   %F.call.loc36 = call i32 @_CF.Main.077d2bb2ab5eee86(i16 %.loc36_5, ptr %.loc36_14, ptr %.loc36_23, i32 0), !dbg !19
// CHECK:STDOUT:   %.loc37_5 = load i32, ptr %val_i32.var, align 4, !dbg !20
// CHECK:STDOUT:   %.loc37_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !21
// CHECK:STDOUT:   %.loc37_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !22
// CHECK:STDOUT:   %F.call.loc37 = call i32 @_CF.Main.9a89ac9e10448feb(i32 %.loc37_5, ptr %.loc37_14, ptr %.loc37_24, i32 0), !dbg !23
// CHECK:STDOUT:   %.loc38_5 = load i1, ptr %val_bool.var, align 1, !dbg !24
// CHECK:STDOUT:   %.loc38_15 = load ptr, ptr %ptr_bool.var, align 8, !dbg !25
// CHECK:STDOUT:   %.loc38_25 = load ptr, ptr %ptr_i16.var, align 8, !dbg !26
// CHECK:STDOUT:   %F.call.loc38 = call i32 @_CF.Main.76a945443862cc1f(i1 %.loc38_5, ptr %.loc38_15, ptr %.loc38_25, i32 0), !dbg !27
// CHECK:STDOUT:   %.loc39_5 = load i64, ptr %val_i64.var, align 4, !dbg !28
// CHECK:STDOUT:   %.loc39_14 = load i1, ptr %val_bool.var, align 1, !dbg !29
// CHECK:STDOUT:   %.loc39_24 = load ptr, ptr %ptr_bool.var, align 8, !dbg !30
// CHECK:STDOUT:   %F.call.loc39 = call i32 @_CF.Main.02da1a7614ea56b6(i64 %.loc39_5, i1 %.loc39_14, ptr %.loc39_24, i32 0), !dbg !31
// CHECK:STDOUT:   %.loc40_5 = load double, ptr %val_f64.var, align 8, !dbg !32
// CHECK:STDOUT:   %.loc40_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !33
// CHECK:STDOUT:   %.loc40_24 = load ptr, ptr %ptr_bool.var, align 8, !dbg !34
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.f51a89281f4fb62e(double %.loc40_5, ptr %.loc40_14, ptr %.loc40_24, i32 0), !dbg !35
// CHECK:STDOUT:   %.loc44_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !36
// CHECK:STDOUT:   %.loc44_15 = load ptr, ptr %ptr_i16.var, align 8, !dbg !37
// CHECK:STDOUT:   %.loc44_24 = load ptr, ptr %ptr_i32.var, align 8, !dbg !38
// CHECK:STDOUT:   %F.call.loc44 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc44_5, ptr %.loc44_15, ptr %.loc44_24, i32 0), !dbg !39
// CHECK:STDOUT:   %.loc45_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !40
// CHECK:STDOUT:   %.loc45_15 = load ptr, ptr %ptr_i16.var, align 8, !dbg !41
// CHECK:STDOUT:   %.loc45_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !42
// CHECK:STDOUT:   %F.call.loc45 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc45_5, ptr %.loc45_15, ptr %.loc45_24, i32 0), !dbg !43
// CHECK:STDOUT:   %.loc46_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !44
// CHECK:STDOUT:   %.loc46_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !45
// CHECK:STDOUT:   %.loc46_24 = load ptr, ptr %ptr_i16.var, align 8, !dbg !46
// CHECK:STDOUT:   %F.call.loc46 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc46_5, ptr %.loc46_14, ptr %.loc46_24, i32 0), !dbg !47
// CHECK:STDOUT:   %.loc47_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !48
// CHECK:STDOUT:   %.loc47_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !49
// CHECK:STDOUT:   %.loc47_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !50
// CHECK:STDOUT:   %F.call.loc47 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc47_5, ptr %.loc47_14, ptr %.loc47_24, i32 0), !dbg !51
// CHECK:STDOUT:   %.loc48_5 = load ptr, ptr %ptr_i16.var, align 8, !dbg !52
// CHECK:STDOUT:   %.loc48_14 = load ptr, ptr %ptr_i32.var, align 8, !dbg !53
// CHECK:STDOUT:   %.loc48_23 = load ptr, ptr %ptr_i64.var, align 8, !dbg !54
// CHECK:STDOUT:   %F.call.loc48 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc48_5, ptr %.loc48_14, ptr %.loc48_23, i32 0), !dbg !55
// CHECK:STDOUT:   %.loc49_5 = load ptr, ptr %ptr_i64.var, align 8, !dbg !56
// CHECK:STDOUT:   %.loc49_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !57
// CHECK:STDOUT:   %.loc49_24 = load ptr, ptr %ptr_i16.var, align 8, !dbg !58
// CHECK:STDOUT:   %F.call.loc49 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc49_5, ptr %.loc49_14, ptr %.loc49_24, i32 0), !dbg !59
// CHECK:STDOUT:   %.loc50_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !60
// CHECK:STDOUT:   %.loc50_15 = load ptr, ptr %ptr_bool.var, align 8, !dbg !61
// CHECK:STDOUT:   %.loc50_25 = load ptr, ptr %ptr_bool.var, align 8, !dbg !62
// CHECK:STDOUT:   %F.call.loc50 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc50_5, ptr %.loc50_15, ptr %.loc50_25, i32 0), !dbg !63
// CHECK:STDOUT:   %.loc30_3 = load ptr, ptr %ptr_bool.var, align 8, !dbg !15
// CHECK:STDOUT:   %.loc29_3 = load ptr, ptr %ptr_i64.var, align 8, !dbg !14
// CHECK:STDOUT:   %.loc28_3 = load ptr, ptr %ptr_i32.var, align 8, !dbg !13
// CHECK:STDOUT:   %.loc27_3 = load ptr, ptr %ptr_i16.var, align 8, !dbg !12
// CHECK:STDOUT:   %.loc25_3 = load i1, ptr %val_bool.var, align 1, !dbg !11
// CHECK:STDOUT:   %.loc24_3 = load double, ptr %val_f64.var, align 8, !dbg !10
// CHECK:STDOUT:   %.loc23_3 = load i64, ptr %val_i64.var, align 4, !dbg !9
// CHECK:STDOUT:   %.loc22_3 = load i32, ptr %val_i32.var, align 4, !dbg !8
// CHECK:STDOUT:   %.loc21_3 = load i16, ptr %val_i16.var, align 2, !dbg !7
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.077d2bb2ab5eee86(i16 %x, ptr %y, ptr %z, i32 %count) !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 2, !dbg !66
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !67
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !68
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !69
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.077d2bb2ab5eee86(i16 %x, ptr %z, ptr %y, i32 %int.sadd), !dbg !70
// CHECK:STDOUT:   ret i32 %F.call, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.9a89ac9e10448feb(i32 %x, ptr %y, ptr %z, i32 %count) !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 2, !dbg !73
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !74
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !75
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !76
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.9a89ac9e10448feb(i32 %x, ptr %z, ptr %y, i32 %int.sadd), !dbg !77
// CHECK:STDOUT:   ret i32 %F.call, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.76a945443862cc1f(i1 %x, ptr %y, ptr %z, i32 %count) !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 2, !dbg !80
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !81
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !82
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !83
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.76a945443862cc1f(i1 %x, ptr %z, ptr %y, i32 %int.sadd), !dbg !84
// CHECK:STDOUT:   ret i32 %F.call, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.02da1a7614ea56b6(i64 %x, i1 %y, ptr %z, i32 %count) !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 2, !dbg !87
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !88
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !89
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !90
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.bea8bae6e14e4034(i64 %x, ptr %z, i1 %y, i32 %int.sadd), !dbg !91
// CHECK:STDOUT:   ret i32 %F.call, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.f51a89281f4fb62e(double %x, ptr %y, ptr %z, i32 %count) !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 2, !dbg !94
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !95
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !96
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !97
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.f51a89281f4fb62e(double %x, ptr %z, ptr %y, i32 %int.sadd), !dbg !98
// CHECK:STDOUT:   ret i32 %F.call, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.7776e910959584d9(ptr %x, ptr %y, ptr %z, i32 %count) !dbg !100 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 2, !dbg !101
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !102
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !103
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !104
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.7776e910959584d9(ptr %x, ptr %z, ptr %y, i32 %int.sadd), !dbg !105
// CHECK:STDOUT:   ret i32 %F.call, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.bea8bae6e14e4034(i64 %x, ptr %y, i1 %z, i32 %count) !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 2, !dbg !108
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !109
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !110
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !111
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.02da1a7614ea56b6(i64 %x, i1 %z, ptr %y, i32 %int.sadd), !dbg !112
// CHECK:STDOUT:   ret i32 %F.call, !dbg !113
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.7776e910959584d9, { 5, 7, 6, 4, 0, 1, 2, 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_reorder_more.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 20, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 21, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 22, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 23, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 25, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 27, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 28, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 29, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 30, column: 3, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 36, column: 5, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 36, column: 14, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 36, column: 23, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 36, column: 3, scope: !4)
// CHECK:STDOUT: !20 = !DILocation(line: 37, column: 5, scope: !4)
// CHECK:STDOUT: !21 = !DILocation(line: 37, column: 14, scope: !4)
// CHECK:STDOUT: !22 = !DILocation(line: 37, column: 24, scope: !4)
// CHECK:STDOUT: !23 = !DILocation(line: 37, column: 3, scope: !4)
// CHECK:STDOUT: !24 = !DILocation(line: 38, column: 5, scope: !4)
// CHECK:STDOUT: !25 = !DILocation(line: 38, column: 15, scope: !4)
// CHECK:STDOUT: !26 = !DILocation(line: 38, column: 25, scope: !4)
// CHECK:STDOUT: !27 = !DILocation(line: 38, column: 3, scope: !4)
// CHECK:STDOUT: !28 = !DILocation(line: 39, column: 5, scope: !4)
// CHECK:STDOUT: !29 = !DILocation(line: 39, column: 14, scope: !4)
// CHECK:STDOUT: !30 = !DILocation(line: 39, column: 24, scope: !4)
// CHECK:STDOUT: !31 = !DILocation(line: 39, column: 3, scope: !4)
// CHECK:STDOUT: !32 = !DILocation(line: 40, column: 5, scope: !4)
// CHECK:STDOUT: !33 = !DILocation(line: 40, column: 14, scope: !4)
// CHECK:STDOUT: !34 = !DILocation(line: 40, column: 24, scope: !4)
// CHECK:STDOUT: !35 = !DILocation(line: 40, column: 3, scope: !4)
// CHECK:STDOUT: !36 = !DILocation(line: 44, column: 5, scope: !4)
// CHECK:STDOUT: !37 = !DILocation(line: 44, column: 15, scope: !4)
// CHECK:STDOUT: !38 = !DILocation(line: 44, column: 24, scope: !4)
// CHECK:STDOUT: !39 = !DILocation(line: 44, column: 3, scope: !4)
// CHECK:STDOUT: !40 = !DILocation(line: 45, column: 5, scope: !4)
// CHECK:STDOUT: !41 = !DILocation(line: 45, column: 15, scope: !4)
// CHECK:STDOUT: !42 = !DILocation(line: 45, column: 24, scope: !4)
// CHECK:STDOUT: !43 = !DILocation(line: 45, column: 3, scope: !4)
// CHECK:STDOUT: !44 = !DILocation(line: 46, column: 5, scope: !4)
// CHECK:STDOUT: !45 = !DILocation(line: 46, column: 14, scope: !4)
// CHECK:STDOUT: !46 = !DILocation(line: 46, column: 24, scope: !4)
// CHECK:STDOUT: !47 = !DILocation(line: 46, column: 3, scope: !4)
// CHECK:STDOUT: !48 = !DILocation(line: 47, column: 5, scope: !4)
// CHECK:STDOUT: !49 = !DILocation(line: 47, column: 14, scope: !4)
// CHECK:STDOUT: !50 = !DILocation(line: 47, column: 24, scope: !4)
// CHECK:STDOUT: !51 = !DILocation(line: 47, column: 3, scope: !4)
// CHECK:STDOUT: !52 = !DILocation(line: 48, column: 5, scope: !4)
// CHECK:STDOUT: !53 = !DILocation(line: 48, column: 14, scope: !4)
// CHECK:STDOUT: !54 = !DILocation(line: 48, column: 23, scope: !4)
// CHECK:STDOUT: !55 = !DILocation(line: 48, column: 3, scope: !4)
// CHECK:STDOUT: !56 = !DILocation(line: 49, column: 5, scope: !4)
// CHECK:STDOUT: !57 = !DILocation(line: 49, column: 14, scope: !4)
// CHECK:STDOUT: !58 = !DILocation(line: 49, column: 24, scope: !4)
// CHECK:STDOUT: !59 = !DILocation(line: 49, column: 3, scope: !4)
// CHECK:STDOUT: !60 = !DILocation(line: 50, column: 5, scope: !4)
// CHECK:STDOUT: !61 = !DILocation(line: 50, column: 15, scope: !4)
// CHECK:STDOUT: !62 = !DILocation(line: 50, column: 25, scope: !4)
// CHECK:STDOUT: !63 = !DILocation(line: 50, column: 3, scope: !4)
// CHECK:STDOUT: !64 = !DILocation(line: 20, column: 1, scope: !4)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.077d2bb2ab5eee86", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !66 = !DILocation(line: 14, column: 7, scope: !65)
// CHECK:STDOUT: !67 = !DILocation(line: 14, column: 6, scope: !65)
// CHECK:STDOUT: !68 = !DILocation(line: 15, column: 5, scope: !65)
// CHECK:STDOUT: !69 = !DILocation(line: 17, column: 21, scope: !65)
// CHECK:STDOUT: !70 = !DILocation(line: 17, column: 10, scope: !65)
// CHECK:STDOUT: !71 = !DILocation(line: 17, column: 3, scope: !65)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9a89ac9e10448feb", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !73 = !DILocation(line: 14, column: 7, scope: !72)
// CHECK:STDOUT: !74 = !DILocation(line: 14, column: 6, scope: !72)
// CHECK:STDOUT: !75 = !DILocation(line: 15, column: 5, scope: !72)
// CHECK:STDOUT: !76 = !DILocation(line: 17, column: 21, scope: !72)
// CHECK:STDOUT: !77 = !DILocation(line: 17, column: 10, scope: !72)
// CHECK:STDOUT: !78 = !DILocation(line: 17, column: 3, scope: !72)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.76a945443862cc1f", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !80 = !DILocation(line: 14, column: 7, scope: !79)
// CHECK:STDOUT: !81 = !DILocation(line: 14, column: 6, scope: !79)
// CHECK:STDOUT: !82 = !DILocation(line: 15, column: 5, scope: !79)
// CHECK:STDOUT: !83 = !DILocation(line: 17, column: 21, scope: !79)
// CHECK:STDOUT: !84 = !DILocation(line: 17, column: 10, scope: !79)
// CHECK:STDOUT: !85 = !DILocation(line: 17, column: 3, scope: !79)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.02da1a7614ea56b6", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !87 = !DILocation(line: 14, column: 7, scope: !86)
// CHECK:STDOUT: !88 = !DILocation(line: 14, column: 6, scope: !86)
// CHECK:STDOUT: !89 = !DILocation(line: 15, column: 5, scope: !86)
// CHECK:STDOUT: !90 = !DILocation(line: 17, column: 21, scope: !86)
// CHECK:STDOUT: !91 = !DILocation(line: 17, column: 10, scope: !86)
// CHECK:STDOUT: !92 = !DILocation(line: 17, column: 3, scope: !86)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.f51a89281f4fb62e", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !94 = !DILocation(line: 14, column: 7, scope: !93)
// CHECK:STDOUT: !95 = !DILocation(line: 14, column: 6, scope: !93)
// CHECK:STDOUT: !96 = !DILocation(line: 15, column: 5, scope: !93)
// CHECK:STDOUT: !97 = !DILocation(line: 17, column: 21, scope: !93)
// CHECK:STDOUT: !98 = !DILocation(line: 17, column: 10, scope: !93)
// CHECK:STDOUT: !99 = !DILocation(line: 17, column: 3, scope: !93)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.7776e910959584d9", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !101 = !DILocation(line: 14, column: 7, scope: !100)
// CHECK:STDOUT: !102 = !DILocation(line: 14, column: 6, scope: !100)
// CHECK:STDOUT: !103 = !DILocation(line: 15, column: 5, scope: !100)
// CHECK:STDOUT: !104 = !DILocation(line: 17, column: 21, scope: !100)
// CHECK:STDOUT: !105 = !DILocation(line: 17, column: 10, scope: !100)
// CHECK:STDOUT: !106 = !DILocation(line: 17, column: 3, scope: !100)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.bea8bae6e14e4034", scope: null, file: !3, line: 13, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !108 = !DILocation(line: 14, column: 7, scope: !107)
// CHECK:STDOUT: !109 = !DILocation(line: 14, column: 6, scope: !107)
// CHECK:STDOUT: !110 = !DILocation(line: 15, column: 5, scope: !107)
// CHECK:STDOUT: !111 = !DILocation(line: 17, column: 21, scope: !107)
// CHECK:STDOUT: !112 = !DILocation(line: 17, column: 10, scope: !107)
// CHECK:STDOUT: !113 = !DILocation(line: 17, column: 3, scope: !107)
