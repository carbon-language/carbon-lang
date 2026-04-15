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

// CHECK:STDOUT: ; ModuleID = 'call_recursive_reorder_more.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_reorder_more.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CM.Main() #0 !dbg !4 {
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
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_i16.var), !dbg !7
// CHECK:STDOUT:   store i16 poison, ptr %val_i16.var, align 2, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_i32.var), !dbg !8
// CHECK:STDOUT:   store i32 poison, ptr %val_i32.var, align 4, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_i64.var), !dbg !9
// CHECK:STDOUT:   store i64 poison, ptr %val_i64.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_f64.var), !dbg !10
// CHECK:STDOUT:   store double poison, ptr %val_f64.var, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %val_bool.var), !dbg !11
// CHECK:STDOUT:   store i8 poison, ptr %val_bool.var, align 1, !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i16.var), !dbg !12
// CHECK:STDOUT:   store ptr %val_i16.var, ptr %ptr_i16.var, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !13
// CHECK:STDOUT:   store ptr %val_i32.var, ptr %ptr_i32.var, align 8, !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i64.var), !dbg !14
// CHECK:STDOUT:   store ptr %val_i64.var, ptr %ptr_i64.var, align 8, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_bool.var), !dbg !15
// CHECK:STDOUT:   store ptr %val_bool.var, ptr %ptr_bool.var, align 8, !dbg !15
// CHECK:STDOUT:   %.loc38_5 = load i16, ptr %val_i16.var, align 2, !dbg !16
// CHECK:STDOUT:   %.loc38_14 = load ptr, ptr %ptr_i32.var, align 8, !dbg !17
// CHECK:STDOUT:   %.loc38_23 = load ptr, ptr %ptr_bool.var, align 8, !dbg !18
// CHECK:STDOUT:   %F.call.loc38 = call i32 @_CF.Main.077d2bb2ab5eee86(i16 %.loc38_5, ptr %.loc38_14, ptr %.loc38_23, i32 0), !dbg !19
// CHECK:STDOUT:   %.loc39_5 = load i32, ptr %val_i32.var, align 4, !dbg !20
// CHECK:STDOUT:   %.loc39_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !21
// CHECK:STDOUT:   %.loc39_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !22
// CHECK:STDOUT:   %F.call.loc39 = call i32 @_CF.Main.9a89ac9e10448feb(i32 %.loc39_5, ptr %.loc39_14, ptr %.loc39_24, i32 0), !dbg !23
// CHECK:STDOUT:   %.loc40_5 = load i8, ptr %val_bool.var, align 1, !dbg !24
// CHECK:STDOUT:   %.loc40_51 = trunc i8 %.loc40_5 to i1, !dbg !24
// CHECK:STDOUT:   %.loc40_15 = load ptr, ptr %ptr_bool.var, align 8, !dbg !25
// CHECK:STDOUT:   %.loc40_25 = load ptr, ptr %ptr_i16.var, align 8, !dbg !26
// CHECK:STDOUT:   %F.call.loc40 = call i32 @_CF.Main.76a945443862cc1f(i1 %.loc40_51, ptr %.loc40_15, ptr %.loc40_25, i32 0), !dbg !27
// CHECK:STDOUT:   %.loc41_5 = load i64, ptr %val_i64.var, align 4, !dbg !28
// CHECK:STDOUT:   %.loc41_14 = load i8, ptr %val_bool.var, align 1, !dbg !29
// CHECK:STDOUT:   %.loc41_142 = trunc i8 %.loc41_14 to i1, !dbg !29
// CHECK:STDOUT:   %.loc41_24 = load ptr, ptr %ptr_bool.var, align 8, !dbg !30
// CHECK:STDOUT:   %F.call.loc41 = call i32 @_CF.Main.02da1a7614ea56b6(i64 %.loc41_5, i1 %.loc41_142, ptr %.loc41_24, i32 0), !dbg !31
// CHECK:STDOUT:   %.loc42_5 = load double, ptr %val_f64.var, align 8, !dbg !32
// CHECK:STDOUT:   %.loc42_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !33
// CHECK:STDOUT:   %.loc42_24 = load ptr, ptr %ptr_bool.var, align 8, !dbg !34
// CHECK:STDOUT:   %F.call.loc42 = call i32 @_CF.Main.4dcab66ec095dfec(double %.loc42_5, ptr %.loc42_14, ptr %.loc42_24, i32 0), !dbg !35
// CHECK:STDOUT:   %.loc46_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !36
// CHECK:STDOUT:   %.loc46_15 = load ptr, ptr %ptr_i16.var, align 8, !dbg !37
// CHECK:STDOUT:   %.loc46_24 = load ptr, ptr %ptr_i32.var, align 8, !dbg !38
// CHECK:STDOUT:   %F.call.loc46 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc46_5, ptr %.loc46_15, ptr %.loc46_24, i32 0), !dbg !39
// CHECK:STDOUT:   %.loc47_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !40
// CHECK:STDOUT:   %.loc47_15 = load ptr, ptr %ptr_i16.var, align 8, !dbg !41
// CHECK:STDOUT:   %.loc47_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !42
// CHECK:STDOUT:   %F.call.loc47 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc47_5, ptr %.loc47_15, ptr %.loc47_24, i32 0), !dbg !43
// CHECK:STDOUT:   %.loc48_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !44
// CHECK:STDOUT:   %.loc48_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !45
// CHECK:STDOUT:   %.loc48_24 = load ptr, ptr %ptr_i16.var, align 8, !dbg !46
// CHECK:STDOUT:   %F.call.loc48 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc48_5, ptr %.loc48_14, ptr %.loc48_24, i32 0), !dbg !47
// CHECK:STDOUT:   %.loc49_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !48
// CHECK:STDOUT:   %.loc49_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !49
// CHECK:STDOUT:   %.loc49_24 = load ptr, ptr %ptr_i64.var, align 8, !dbg !50
// CHECK:STDOUT:   %F.call.loc49 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc49_5, ptr %.loc49_14, ptr %.loc49_24, i32 0), !dbg !51
// CHECK:STDOUT:   %.loc50_5 = load ptr, ptr %ptr_i16.var, align 8, !dbg !52
// CHECK:STDOUT:   %.loc50_14 = load ptr, ptr %ptr_i32.var, align 8, !dbg !53
// CHECK:STDOUT:   %.loc50_23 = load ptr, ptr %ptr_i64.var, align 8, !dbg !54
// CHECK:STDOUT:   %F.call.loc50 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc50_5, ptr %.loc50_14, ptr %.loc50_23, i32 0), !dbg !55
// CHECK:STDOUT:   %.loc51_5 = load ptr, ptr %ptr_i64.var, align 8, !dbg !56
// CHECK:STDOUT:   %.loc51_14 = load ptr, ptr %ptr_bool.var, align 8, !dbg !57
// CHECK:STDOUT:   %.loc51_24 = load ptr, ptr %ptr_i16.var, align 8, !dbg !58
// CHECK:STDOUT:   %F.call.loc51 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc51_5, ptr %.loc51_14, ptr %.loc51_24, i32 0), !dbg !59
// CHECK:STDOUT:   %.loc52_5 = load ptr, ptr %ptr_bool.var, align 8, !dbg !60
// CHECK:STDOUT:   %.loc52_15 = load ptr, ptr %ptr_bool.var, align 8, !dbg !61
// CHECK:STDOUT:   %.loc52_25 = load ptr, ptr %ptr_bool.var, align 8, !dbg !62
// CHECK:STDOUT:   %F.call.loc52 = call i32 @_CF.Main.7776e910959584d9(ptr %.loc52_5, ptr %.loc52_15, ptr %.loc52_25, i32 0), !dbg !63
// CHECK:STDOUT:   call void @"_COp.98006a366c2ffd6c:core.Destroy.Core"(ptr %val_f64.var), !dbg !10
// CHECK:STDOUT:   call void @"_COp.0d6779e60704359e:core.Destroy.Core"(ptr %val_i64.var), !dbg !9
// CHECK:STDOUT:   call void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %val_i32.var), !dbg !8
// CHECK:STDOUT:   call void @"_COp.406738f1b62008ed:core.Destroy.Core"(ptr %val_i16.var), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !64
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.98006a366c2ffd6c:core.Destroy.Core"(ptr %self) #0 !dbg !65 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !71
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.0d6779e60704359e:core.Destroy.Core"(ptr %self) #0 !dbg !72 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !78
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.7e389eab4a7e5487:core.Destroy.Core"(ptr %self) #0 !dbg !79 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.406738f1b62008ed:core.Destroy.Core"(ptr %self) #0 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.077d2bb2ab5eee86(i16 %x, ptr %y, ptr %z, i32 %count) #0 !dbg !93 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !101
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !102
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !103
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !104
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.077d2bb2ab5eee86(i16 %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !105
// CHECK:STDOUT:   ret i32 %F.call, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.9a89ac9e10448feb(i32 %x, ptr %y, ptr %z, i32 %count) #0 !dbg !107 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !115
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !116
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !117
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !118
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.9a89ac9e10448feb(i32 %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !119
// CHECK:STDOUT:   ret i32 %F.call, !dbg !120
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.76a945443862cc1f(i1 %x, ptr %y, ptr %z, i32 %count) #0 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !129
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !130
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !131
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !132
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.76a945443862cc1f(i1 %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !133
// CHECK:STDOUT:   ret i32 %F.call, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.02da1a7614ea56b6(i64 %x, i1 %y, ptr %z, i32 %count) #0 !dbg !135 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !143
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !144
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !145
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !146
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.bea8bae6e14e4034(i64 %x, ptr %z, i1 %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !147
// CHECK:STDOUT:   ret i32 %F.call, !dbg !148
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.4dcab66ec095dfec(double %x, ptr %y, ptr %z, i32 %count) #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !155
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !156
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !157
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !158
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.4dcab66ec095dfec(double %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !159
// CHECK:STDOUT:   ret i32 %F.call, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.7776e910959584d9(ptr %x, ptr %y, ptr %z, i32 %count) #0 !dbg !161 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !167
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !168
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !169
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !170
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.7776e910959584d9(ptr %x, ptr %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !171
// CHECK:STDOUT:   ret i32 %F.call, !dbg !172
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.bea8bae6e14e4034(i64 %x, ptr %y, i1 %z, i32 %count) #0 !dbg !173 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 2, !dbg !179
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !180
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %count, !dbg !181
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !182
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.02da1a7614ea56b6(i64 %x, i1 %z, ptr %y, i32 %Int.as.AddWith.impl.Op.call), !dbg !183
// CHECK:STDOUT:   ret i32 %F.call, !dbg !184
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.7776e910959584d9, { 5, 7, 6, 4, 0, 1, 2, 3 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_reorder_more.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{null}
// CHECK:STDOUT: !7 = !DILocation(line: 23, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 25, column: 3, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 26, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 27, column: 3, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 29, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 30, column: 3, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 31, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 32, column: 3, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 38, column: 5, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 38, column: 14, scope: !4)
// CHECK:STDOUT: !18 = !DILocation(line: 38, column: 23, scope: !4)
// CHECK:STDOUT: !19 = !DILocation(line: 38, column: 3, scope: !4)
// CHECK:STDOUT: !20 = !DILocation(line: 39, column: 5, scope: !4)
// CHECK:STDOUT: !21 = !DILocation(line: 39, column: 14, scope: !4)
// CHECK:STDOUT: !22 = !DILocation(line: 39, column: 24, scope: !4)
// CHECK:STDOUT: !23 = !DILocation(line: 39, column: 3, scope: !4)
// CHECK:STDOUT: !24 = !DILocation(line: 40, column: 5, scope: !4)
// CHECK:STDOUT: !25 = !DILocation(line: 40, column: 15, scope: !4)
// CHECK:STDOUT: !26 = !DILocation(line: 40, column: 25, scope: !4)
// CHECK:STDOUT: !27 = !DILocation(line: 40, column: 3, scope: !4)
// CHECK:STDOUT: !28 = !DILocation(line: 41, column: 5, scope: !4)
// CHECK:STDOUT: !29 = !DILocation(line: 41, column: 14, scope: !4)
// CHECK:STDOUT: !30 = !DILocation(line: 41, column: 24, scope: !4)
// CHECK:STDOUT: !31 = !DILocation(line: 41, column: 3, scope: !4)
// CHECK:STDOUT: !32 = !DILocation(line: 42, column: 5, scope: !4)
// CHECK:STDOUT: !33 = !DILocation(line: 42, column: 14, scope: !4)
// CHECK:STDOUT: !34 = !DILocation(line: 42, column: 24, scope: !4)
// CHECK:STDOUT: !35 = !DILocation(line: 42, column: 3, scope: !4)
// CHECK:STDOUT: !36 = !DILocation(line: 46, column: 5, scope: !4)
// CHECK:STDOUT: !37 = !DILocation(line: 46, column: 15, scope: !4)
// CHECK:STDOUT: !38 = !DILocation(line: 46, column: 24, scope: !4)
// CHECK:STDOUT: !39 = !DILocation(line: 46, column: 3, scope: !4)
// CHECK:STDOUT: !40 = !DILocation(line: 47, column: 5, scope: !4)
// CHECK:STDOUT: !41 = !DILocation(line: 47, column: 15, scope: !4)
// CHECK:STDOUT: !42 = !DILocation(line: 47, column: 24, scope: !4)
// CHECK:STDOUT: !43 = !DILocation(line: 47, column: 3, scope: !4)
// CHECK:STDOUT: !44 = !DILocation(line: 48, column: 5, scope: !4)
// CHECK:STDOUT: !45 = !DILocation(line: 48, column: 14, scope: !4)
// CHECK:STDOUT: !46 = !DILocation(line: 48, column: 24, scope: !4)
// CHECK:STDOUT: !47 = !DILocation(line: 48, column: 3, scope: !4)
// CHECK:STDOUT: !48 = !DILocation(line: 49, column: 5, scope: !4)
// CHECK:STDOUT: !49 = !DILocation(line: 49, column: 14, scope: !4)
// CHECK:STDOUT: !50 = !DILocation(line: 49, column: 24, scope: !4)
// CHECK:STDOUT: !51 = !DILocation(line: 49, column: 3, scope: !4)
// CHECK:STDOUT: !52 = !DILocation(line: 50, column: 5, scope: !4)
// CHECK:STDOUT: !53 = !DILocation(line: 50, column: 14, scope: !4)
// CHECK:STDOUT: !54 = !DILocation(line: 50, column: 23, scope: !4)
// CHECK:STDOUT: !55 = !DILocation(line: 50, column: 3, scope: !4)
// CHECK:STDOUT: !56 = !DILocation(line: 51, column: 5, scope: !4)
// CHECK:STDOUT: !57 = !DILocation(line: 51, column: 14, scope: !4)
// CHECK:STDOUT: !58 = !DILocation(line: 51, column: 24, scope: !4)
// CHECK:STDOUT: !59 = !DILocation(line: 51, column: 3, scope: !4)
// CHECK:STDOUT: !60 = !DILocation(line: 52, column: 5, scope: !4)
// CHECK:STDOUT: !61 = !DILocation(line: 52, column: 15, scope: !4)
// CHECK:STDOUT: !62 = !DILocation(line: 52, column: 25, scope: !4)
// CHECK:STDOUT: !63 = !DILocation(line: 52, column: 3, scope: !4)
// CHECK:STDOUT: !64 = !DILocation(line: 22, column: 1, scope: !4)
// CHECK:STDOUT: !65 = distinct !DISubprogram(name: "Op", linkageName: "_COp.98006a366c2ffd6c:core.Destroy.Core", scope: null, file: !3, line: 26, type: !66, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !69)
// CHECK:STDOUT: !66 = !DISubroutineType(types: !67)
// CHECK:STDOUT: !67 = !{null, !68}
// CHECK:STDOUT: !68 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !69 = !{!70}
// CHECK:STDOUT: !70 = !DILocalVariable(arg: 1, scope: !65, type: !68)
// CHECK:STDOUT: !71 = !DILocation(line: 26, column: 3, scope: !65)
// CHECK:STDOUT: !72 = distinct !DISubprogram(name: "Op", linkageName: "_COp.0d6779e60704359e:core.Destroy.Core", scope: null, file: !3, line: 25, type: !73, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !76)
// CHECK:STDOUT: !73 = !DISubroutineType(types: !74)
// CHECK:STDOUT: !74 = !{null, !75}
// CHECK:STDOUT: !75 = !DIBasicType(name: "int", size: 64, encoding: DW_ATE_signed)
// CHECK:STDOUT: !76 = !{!77}
// CHECK:STDOUT: !77 = !DILocalVariable(arg: 1, scope: !72, type: !75)
// CHECK:STDOUT: !78 = !DILocation(line: 25, column: 3, scope: !72)
// CHECK:STDOUT: !79 = distinct !DISubprogram(name: "Op", linkageName: "_COp.7e389eab4a7e5487:core.Destroy.Core", scope: null, file: !3, line: 24, type: !80, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !83)
// CHECK:STDOUT: !80 = !DISubroutineType(types: !81)
// CHECK:STDOUT: !81 = !{null, !82}
// CHECK:STDOUT: !82 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !83 = !{!84}
// CHECK:STDOUT: !84 = !DILocalVariable(arg: 1, scope: !79, type: !82)
// CHECK:STDOUT: !85 = !DILocation(line: 24, column: 3, scope: !79)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Op", linkageName: "_COp.406738f1b62008ed:core.Destroy.Core", scope: null, file: !3, line: 23, type: !87, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !90)
// CHECK:STDOUT: !87 = !DISubroutineType(types: !88)
// CHECK:STDOUT: !88 = !{null, !89}
// CHECK:STDOUT: !89 = !DIBasicType(name: "int", size: 16, encoding: DW_ATE_signed)
// CHECK:STDOUT: !90 = !{!91}
// CHECK:STDOUT: !91 = !DILocalVariable(arg: 1, scope: !86, type: !89)
// CHECK:STDOUT: !92 = !DILocation(line: 23, column: 3, scope: !86)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.077d2bb2ab5eee86", scope: null, file: !3, line: 15, type: !94, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !96)
// CHECK:STDOUT: !94 = !DISubroutineType(types: !95)
// CHECK:STDOUT: !95 = !{!82, !89, !68, !68, !82}
// CHECK:STDOUT: !96 = !{!97, !98, !99, !100}
// CHECK:STDOUT: !97 = !DILocalVariable(arg: 1, scope: !93, type: !89)
// CHECK:STDOUT: !98 = !DILocalVariable(arg: 2, scope: !93, type: !68)
// CHECK:STDOUT: !99 = !DILocalVariable(arg: 3, scope: !93, type: !68)
// CHECK:STDOUT: !100 = !DILocalVariable(arg: 4, scope: !93, type: !82)
// CHECK:STDOUT: !101 = !DILocation(line: 16, column: 7, scope: !93)
// CHECK:STDOUT: !102 = !DILocation(line: 16, column: 6, scope: !93)
// CHECK:STDOUT: !103 = !DILocation(line: 17, column: 5, scope: !93)
// CHECK:STDOUT: !104 = !DILocation(line: 19, column: 21, scope: !93)
// CHECK:STDOUT: !105 = !DILocation(line: 19, column: 10, scope: !93)
// CHECK:STDOUT: !106 = !DILocation(line: 19, column: 3, scope: !93)
// CHECK:STDOUT: !107 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.9a89ac9e10448feb", scope: null, file: !3, line: 15, type: !108, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !110)
// CHECK:STDOUT: !108 = !DISubroutineType(types: !109)
// CHECK:STDOUT: !109 = !{!82, !82, !68, !68, !82}
// CHECK:STDOUT: !110 = !{!111, !112, !113, !114}
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !107, type: !82)
// CHECK:STDOUT: !112 = !DILocalVariable(arg: 2, scope: !107, type: !68)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 3, scope: !107, type: !68)
// CHECK:STDOUT: !114 = !DILocalVariable(arg: 4, scope: !107, type: !82)
// CHECK:STDOUT: !115 = !DILocation(line: 16, column: 7, scope: !107)
// CHECK:STDOUT: !116 = !DILocation(line: 16, column: 6, scope: !107)
// CHECK:STDOUT: !117 = !DILocation(line: 17, column: 5, scope: !107)
// CHECK:STDOUT: !118 = !DILocation(line: 19, column: 21, scope: !107)
// CHECK:STDOUT: !119 = !DILocation(line: 19, column: 10, scope: !107)
// CHECK:STDOUT: !120 = !DILocation(line: 19, column: 3, scope: !107)
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.76a945443862cc1f", scope: null, file: !3, line: 15, type: !122, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !124)
// CHECK:STDOUT: !122 = !DISubroutineType(types: !123)
// CHECK:STDOUT: !123 = !{!82, !68, !68, !68, !82}
// CHECK:STDOUT: !124 = !{!125, !126, !127, !128}
// CHECK:STDOUT: !125 = !DILocalVariable(arg: 1, scope: !121, type: !68)
// CHECK:STDOUT: !126 = !DILocalVariable(arg: 2, scope: !121, type: !68)
// CHECK:STDOUT: !127 = !DILocalVariable(arg: 3, scope: !121, type: !68)
// CHECK:STDOUT: !128 = !DILocalVariable(arg: 4, scope: !121, type: !82)
// CHECK:STDOUT: !129 = !DILocation(line: 16, column: 7, scope: !121)
// CHECK:STDOUT: !130 = !DILocation(line: 16, column: 6, scope: !121)
// CHECK:STDOUT: !131 = !DILocation(line: 17, column: 5, scope: !121)
// CHECK:STDOUT: !132 = !DILocation(line: 19, column: 21, scope: !121)
// CHECK:STDOUT: !133 = !DILocation(line: 19, column: 10, scope: !121)
// CHECK:STDOUT: !134 = !DILocation(line: 19, column: 3, scope: !121)
// CHECK:STDOUT: !135 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.02da1a7614ea56b6", scope: null, file: !3, line: 15, type: !136, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !138)
// CHECK:STDOUT: !136 = !DISubroutineType(types: !137)
// CHECK:STDOUT: !137 = !{!82, !75, !68, !68, !82}
// CHECK:STDOUT: !138 = !{!139, !140, !141, !142}
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !135, type: !75)
// CHECK:STDOUT: !140 = !DILocalVariable(arg: 2, scope: !135, type: !68)
// CHECK:STDOUT: !141 = !DILocalVariable(arg: 3, scope: !135, type: !68)
// CHECK:STDOUT: !142 = !DILocalVariable(arg: 4, scope: !135, type: !82)
// CHECK:STDOUT: !143 = !DILocation(line: 16, column: 7, scope: !135)
// CHECK:STDOUT: !144 = !DILocation(line: 16, column: 6, scope: !135)
// CHECK:STDOUT: !145 = !DILocation(line: 17, column: 5, scope: !135)
// CHECK:STDOUT: !146 = !DILocation(line: 19, column: 21, scope: !135)
// CHECK:STDOUT: !147 = !DILocation(line: 19, column: 10, scope: !135)
// CHECK:STDOUT: !148 = !DILocation(line: 19, column: 3, scope: !135)
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.4dcab66ec095dfec", scope: null, file: !3, line: 15, type: !122, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !150)
// CHECK:STDOUT: !150 = !{!151, !152, !153, !154}
// CHECK:STDOUT: !151 = !DILocalVariable(arg: 1, scope: !149, type: !68)
// CHECK:STDOUT: !152 = !DILocalVariable(arg: 2, scope: !149, type: !68)
// CHECK:STDOUT: !153 = !DILocalVariable(arg: 3, scope: !149, type: !68)
// CHECK:STDOUT: !154 = !DILocalVariable(arg: 4, scope: !149, type: !82)
// CHECK:STDOUT: !155 = !DILocation(line: 16, column: 7, scope: !149)
// CHECK:STDOUT: !156 = !DILocation(line: 16, column: 6, scope: !149)
// CHECK:STDOUT: !157 = !DILocation(line: 17, column: 5, scope: !149)
// CHECK:STDOUT: !158 = !DILocation(line: 19, column: 21, scope: !149)
// CHECK:STDOUT: !159 = !DILocation(line: 19, column: 10, scope: !149)
// CHECK:STDOUT: !160 = !DILocation(line: 19, column: 3, scope: !149)
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.7776e910959584d9", scope: null, file: !3, line: 15, type: !122, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !162)
// CHECK:STDOUT: !162 = !{!163, !164, !165, !166}
// CHECK:STDOUT: !163 = !DILocalVariable(arg: 1, scope: !161, type: !68)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 2, scope: !161, type: !68)
// CHECK:STDOUT: !165 = !DILocalVariable(arg: 3, scope: !161, type: !68)
// CHECK:STDOUT: !166 = !DILocalVariable(arg: 4, scope: !161, type: !82)
// CHECK:STDOUT: !167 = !DILocation(line: 16, column: 7, scope: !161)
// CHECK:STDOUT: !168 = !DILocation(line: 16, column: 6, scope: !161)
// CHECK:STDOUT: !169 = !DILocation(line: 17, column: 5, scope: !161)
// CHECK:STDOUT: !170 = !DILocation(line: 19, column: 21, scope: !161)
// CHECK:STDOUT: !171 = !DILocation(line: 19, column: 10, scope: !161)
// CHECK:STDOUT: !172 = !DILocation(line: 19, column: 3, scope: !161)
// CHECK:STDOUT: !173 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.bea8bae6e14e4034", scope: null, file: !3, line: 15, type: !136, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !174)
// CHECK:STDOUT: !174 = !{!175, !176, !177, !178}
// CHECK:STDOUT: !175 = !DILocalVariable(arg: 1, scope: !173, type: !75)
// CHECK:STDOUT: !176 = !DILocalVariable(arg: 2, scope: !173, type: !68)
// CHECK:STDOUT: !177 = !DILocalVariable(arg: 3, scope: !173, type: !68)
// CHECK:STDOUT: !178 = !DILocalVariable(arg: 4, scope: !173, type: !82)
// CHECK:STDOUT: !179 = !DILocation(line: 16, column: 7, scope: !173)
// CHECK:STDOUT: !180 = !DILocation(line: 16, column: 6, scope: !173)
// CHECK:STDOUT: !181 = !DILocation(line: 17, column: 5, scope: !173)
// CHECK:STDOUT: !182 = !DILocation(line: 19, column: 21, scope: !173)
// CHECK:STDOUT: !183 = !DILocation(line: 19, column: 10, scope: !173)
// CHECK:STDOUT: !184 = !DILocation(line: 19, column: 3, scope: !173)
