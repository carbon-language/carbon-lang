// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --optimize=none foo.carbon --target=x86_64-unknown-linux-gnu --phase=optimize --dump-llvm-ir --exclude-dump-file-prefix=%{core}
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/optimize/optimize_none.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/optimize/optimize_none.carbon

// --- foo.carbon
fn Rand() -> i32;

fn NoInlineWithOz() -> i32 {
  return Rand() + Rand();
}

fn CallNoInlineWithOptSize() -> i32 {
  // Should not be inlined with --optimize=size, because the body is larger than the call.
  return NoInlineWithOz();
}

fn VectorizedWithOptSpeed(a: array(i32, 65536)*) {
  // Should be vectorized with --optimize=speed, but not in other modes.
  var n: i32 = 0;
  while (n < 65536) {
    (*a)[n] *= 2;
    ++n;
  }
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'foo.carbon'
// CHECK:STDOUT: source_filename = "foo.carbon"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CRand.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline nounwind optnone
// CHECK:STDOUT: define i32 @_CNoInlineWithOz.Main() #0 !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Rand.call.loc4_15 = call i32 @_CRand.Main(), !dbg !11
// CHECK:STDOUT:   %Rand.call.loc4_24 = call i32 @_CRand.Main(), !dbg !12
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %Rand.call.loc4_15, %Rand.call.loc4_24, !dbg !11
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline nounwind optnone
// CHECK:STDOUT: define i32 @_CCallNoInlineWithOptSize.Main() #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %NoInlineWithOz.call = call i32 @_CNoInlineWithOz.Main(), !dbg !17
// CHECK:STDOUT:   ret i32 %NoInlineWithOz.call, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline nounwind optnone
// CHECK:STDOUT: define void @_CVectorizedWithOptSpeed.Main(ptr %a) #0 !dbg !19 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !22
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !22
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !22
// CHECK:STDOUT:   br label %while.cond, !dbg !24
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %while.body, %entry
// CHECK:STDOUT:   %.loc15_10 = load i32, ptr %n.var, align 4, !dbg !25
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.call = icmp slt i32 %.loc15_10, 65536, !dbg !25
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Less.call, label %while.body, label %while.done, !dbg !24
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   %.loc16_10 = load i32, ptr %n.var, align 4, !dbg !28
// CHECK:STDOUT:   %.loc16_11.array.index = getelementptr inbounds [65536 x i32], ptr %a, i32 0, i32 %.loc16_10, !dbg !29
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call = load i32, ptr %.loc16_11.array.index, align 4, !dbg !29
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call1 = mul i32 %Int.as.MulAssignWith.impl.Op.call, 2, !dbg !29
// CHECK:STDOUT:   store i32 %Int.as.MulAssignWith.impl.Op.call1, ptr %.loc16_11.array.index, align 4, !dbg !29
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %n.var), !dbg !31
// CHECK:STDOUT:   br label %while.cond, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %n.var), !dbg !22
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !38 {
// CHECK:STDOUT:   call void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 1), !dbg !41
// CHECK:STDOUT:   ret void, !dbg !44
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !47 {
// CHECK:STDOUT:   ret void, !dbg !52
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26"(ptr %self, i32 %other) #0 !dbg !54 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %other), !dbg !58
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !58
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !58
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !58
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701"(i32 %self) #0 !dbg !62 {
// CHECK:STDOUT:   %1 = call i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self), !dbg !67
// CHECK:STDOUT:   ret i32 %1, !dbg !69
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr i32 @"_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8"(i32 %self) #0 !dbg !71 {
// CHECK:STDOUT:   ret i32 %self, !dbg !77
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { noinline nounwind optnone }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!1}
// CHECK:STDOUT: !llvm.module.flags = !{!79, !80}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !DIFile(filename: "foo.carbon", directory: "")
// CHECK:STDOUT: !1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !0, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !2 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !3 = !{!2}
// CHECK:STDOUT: !4 = !DISubroutineType(types: !3)
// CHECK:STDOUT: !5 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !6 = !{null, !5}
// CHECK:STDOUT: !7 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "NoInlineWithOz", linkageName: "_CNoInlineWithOz.Main", scope: null, file: !0, line: 3, type: !4, spFlags: DISPFlagDefinition, unit: !1)
// CHECK:STDOUT: !11 = !DILocation(line: 4, column: 10, scope: !8)
// CHECK:STDOUT: !12 = !DILocation(line: 4, column: 19, scope: !8)
// CHECK:STDOUT: !13 = !DILocation(line: 4, column: 3, scope: !8)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "CallNoInlineWithOptSize", linkageName: "_CCallNoInlineWithOptSize.Main", scope: null, file: !0, line: 7, type: !4, spFlags: DISPFlagDefinition, unit: !1)
// CHECK:STDOUT: !17 = !DILocation(line: 9, column: 10, scope: !14)
// CHECK:STDOUT: !18 = !DILocation(line: 9, column: 3, scope: !14)
// CHECK:STDOUT: !19 = distinct !DISubprogram(name: "VectorizedWithOptSpeed", linkageName: "_CVectorizedWithOptSpeed.Main", scope: null, file: !0, line: 12, type: !7, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !36)
// CHECK:STDOUT: !20 = !DILocalVariable(arg: 1, scope: !19, type: !5)
// CHECK:STDOUT: !22 = !DILocation(line: 14, column: 3, scope: !19)
// CHECK:STDOUT: !24 = !DILocation(line: 15, column: 9, scope: !19)
// CHECK:STDOUT: !25 = !DILocation(line: 15, column: 10, scope: !19)
// CHECK:STDOUT: !28 = !DILocation(line: 16, column: 10, scope: !19)
// CHECK:STDOUT: !29 = !DILocation(line: 16, column: 5, scope: !19)
// CHECK:STDOUT: !31 = !DILocation(line: 17, column: 5, scope: !19)
// CHECK:STDOUT: !32 = !{null, !2}
// CHECK:STDOUT: !33 = !DISubroutineType(types: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 15, column: 3, scope: !19)
// CHECK:STDOUT: !35 = !DILocation(line: 12, column: 1, scope: !19)
// CHECK:STDOUT: !36 = !{!20}
// CHECK:STDOUT: !37 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !38 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !37, line: 396, type: !33, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !45)
// CHECK:STDOUT: !39 = !DILocalVariable(arg: 1, scope: !38, type: !2)
// CHECK:STDOUT: !41 = !DILocation(line: 398, column: 5, scope: !38)
// CHECK:STDOUT: !42 = !{null, !2, !2}
// CHECK:STDOUT: !43 = !DISubroutineType(types: !42)
// CHECK:STDOUT: !44 = !DILocation(line: 396, column: 3, scope: !38)
// CHECK:STDOUT: !45 = !{!39}
// CHECK:STDOUT: !46 = !DIFile(filename: "{{.*}}/prelude/destroy.carbon", directory: "")
// CHECK:STDOUT: !47 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !46, line: 35, type: !33, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !53)
// CHECK:STDOUT: !48 = !DILocalVariable(arg: 1, scope: !47, type: !2)
// CHECK:STDOUT: !52 = !DILocation(line: 35, column: 1, scope: !47)
// CHECK:STDOUT: !53 = !{!48}
// CHECK:STDOUT: !54 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:AddAssignWith.b59e374b0649c9ad.Core:Int.9452f4c51951679b.Core:AddAssignWith.ab03190a9e911cb4.Core.306167be98c24f26", scope: null, file: !37, line: 331, type: !43, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !61)
// CHECK:STDOUT: !55 = !DILocalVariable(arg: 1, scope: !54, type: !2)
// CHECK:STDOUT: !56 = !DILocalVariable(arg: 2, scope: !54, type: !2)
// CHECK:STDOUT: !58 = !DILocation(line: 331, column: 3, scope: !54)
// CHECK:STDOUT: !59 = !{!2, !2}
// CHECK:STDOUT: !60 = !DISubroutineType(types: !59)
// CHECK:STDOUT: !61 = !{!55, !56}
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.df8c28cb3df17b7d:ImplicitAs.c95da57eee1d2d0f.Core.25ef994321ea0701", scope: null, file: !37, line: 64, type: !60, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !70)
// CHECK:STDOUT: !63 = !DILocalVariable(arg: 1, scope: !62, type: !2)
// CHECK:STDOUT: !67 = !DILocation(line: 66, column: 17, scope: !62)
// CHECK:STDOUT: !69 = !DILocation(line: 66, column: 5, scope: !62)
// CHECK:STDOUT: !70 = !{!63}
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "AsInt", linkageName: "_CAsInt.Int.9452f4c51951679b.Core:AnyInt.Core.be1e879c1ad406d8", scope: null, file: !37, line: 59, type: !60, spFlags: DISPFlagDefinition, unit: !1, retainedNodes: !78)
// CHECK:STDOUT: !72 = !DILocalVariable(arg: 1, scope: !71, type: !2)
// CHECK:STDOUT: !77 = !DILocation(line: 59, column: 28, scope: !71)
// CHECK:STDOUT: !78 = !{!72}
// CHECK:STDOUT: !79 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !80 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT:
