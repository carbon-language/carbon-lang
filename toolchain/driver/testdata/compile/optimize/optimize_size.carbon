// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: compile --optimize=size foo.carbon --target=x86_64-unknown-linux-gnu --phase=lower --dump-llvm-ir --exclude-dump-file-prefix=%{core}
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/optimize/optimize_size.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/optimize/optimize_size.carbon

// --- foo.carbon

import Core library "range";

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

// CHECK:STDOUT: ; ModuleID = 'foo.carbon'
// CHECK:STDOUT: source_filename = "foo.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: declare i32 @_CRand.Main()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: minsize nounwind optsize
// CHECK:STDOUT: define i32 @_CNoInlineWithOz.Main() #0 !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Rand.call.loc7_15 = call i32 @_CRand.Main(), !dbg !8
// CHECK:STDOUT:   %Rand.call.loc7_24 = call i32 @_CRand.Main(), !dbg !9
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %Rand.call.loc7_15, %Rand.call.loc7_24, !dbg !8
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call, !dbg !10
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: minsize nounwind optsize
// CHECK:STDOUT: define i32 @_CCallNoInlineWithOptSize.Main() #0 !dbg !11 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %NoInlineWithOz.call = call i32 @_CNoInlineWithOz.Main(), !dbg !12
// CHECK:STDOUT:   ret i32 %NoInlineWithOz.call, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: minsize nounwind optsize
// CHECK:STDOUT: define void @_CVectorizedWithOptSpeed.Main(ptr %a) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !20
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !20
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !20
// CHECK:STDOUT:   br label %while.cond, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %while.body, %entry
// CHECK:STDOUT:   %.loc18_10 = load i32, ptr %n.var, align 4, !dbg !22
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.call = icmp slt i32 %.loc18_10, 65536, !dbg !22
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Less.call, label %while.body, label %while.done, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   %.loc19_10 = load i32, ptr %n.var, align 4, !dbg !23
// CHECK:STDOUT:   %.loc19_11.array.index = getelementptr inbounds [65536 x i32], ptr %a, i32 0, i32 %.loc19_10, !dbg !24
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call = load i32, ptr %.loc19_11.array.index, align 4, !dbg !24
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call1 = mul i32 %Int.as.MulAssignWith.impl.Op.call, 2, !dbg !24
// CHECK:STDOUT:   store i32 %Int.as.MulAssignWith.impl.Op.call1, ptr %.loc19_11.array.index, align 4, !dbg !24
// CHECK:STDOUT:   call void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %n.var), !dbg !25
// CHECK:STDOUT:   br label %while.cond, !dbg !26
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   ret void, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: minsize nounwind optsize
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !28 {
// CHECK:STDOUT:   call void @"_COp:thunk.Int.9452f4c51951679b.Core:AddAssignWith.5375cfcbca6d9e35.Core.75bec4d236c9daa5"(ptr %self, i32 1), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline minsize nounwind optsize
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk.Int.9452f4c51951679b.Core:AddAssignWith.5375cfcbca6d9e35.Core.75bec4d236c9daa5"(ptr %self, i32 %other) #2 !dbg !36 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.14b8745117b2bc54:ImplicitAs.a9271c1e04015f9c.Core.64ccbb8e5d9a0b8e"(i32 %other), !dbg !42
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !43
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !43
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !43
// CHECK:STDOUT:   ret void, !dbg !43
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: minsize nounwind optsize
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.14b8745117b2bc54:ImplicitAs.a9271c1e04015f9c.Core.64ccbb8e5d9a0b8e"(i32 %self) #0 !dbg !44 {
// CHECK:STDOUT:   ret i32 %self, !dbg !50
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { minsize nounwind optsize }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline minsize nounwind optsize }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "foo.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "NoInlineWithOz", linkageName: "_CNoInlineWithOz.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{!7}
// CHECK:STDOUT: !7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !8 = !DILocation(line: 7, column: 10, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 19, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !11 = distinct !DISubprogram(name: "CallNoInlineWithOptSize", linkageName: "_CCallNoInlineWithOptSize.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !12 = !DILocation(line: 12, column: 10, scope: !11)
// CHECK:STDOUT: !13 = !DILocation(line: 12, column: 3, scope: !11)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "VectorizedWithOptSpeed", linkageName: "_CVectorizedWithOptSpeed.Main", scope: null, file: !3, line: 15, type: !15, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !18)
// CHECK:STDOUT: !15 = !DISubroutineType(types: !16)
// CHECK:STDOUT: !16 = !{null, !17}
// CHECK:STDOUT: !17 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !18 = !{!19}
// CHECK:STDOUT: !19 = !DILocalVariable(arg: 1, scope: !14, type: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 17, column: 3, scope: !14)
// CHECK:STDOUT: !21 = !DILocation(line: 18, column: 9, scope: !14)
// CHECK:STDOUT: !22 = !DILocation(line: 18, column: 10, scope: !14)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 10, scope: !14)
// CHECK:STDOUT: !24 = !DILocation(line: 19, column: 5, scope: !14)
// CHECK:STDOUT: !25 = !DILocation(line: 20, column: 5, scope: !14)
// CHECK:STDOUT: !26 = !DILocation(line: 18, column: 3, scope: !14)
// CHECK:STDOUT: !27 = !DILocation(line: 15, column: 1, scope: !14)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.9452f4c51951679b.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !29, line: 341, type: !30, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !32)
// CHECK:STDOUT: !29 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !30 = !DISubroutineType(types: !31)
// CHECK:STDOUT: !31 = !{null, !7}
// CHECK:STDOUT: !32 = !{!33}
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !28, type: !7)
// CHECK:STDOUT: !34 = !DILocation(line: 343, column: 5, scope: !28)
// CHECK:STDOUT: !35 = !DILocation(line: 341, column: 3, scope: !28)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.9452f4c51951679b.Core:AddAssignWith.5375cfcbca6d9e35.Core.75bec4d236c9daa5", scope: null, file: !29, line: 277, type: !37, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !39)
// CHECK:STDOUT: !37 = !DISubroutineType(types: !38)
// CHECK:STDOUT: !38 = !{null, !7, !7}
// CHECK:STDOUT: !39 = !{!40, !41}
// CHECK:STDOUT: !40 = !DILocalVariable(arg: 1, scope: !36, type: !7)
// CHECK:STDOUT: !41 = !DILocalVariable(arg: 2, scope: !36, type: !7)
// CHECK:STDOUT: !42 = !DILocation(line: 4294967295, scope: !36)
// CHECK:STDOUT: !43 = !DILocation(line: 277, column: 3, scope: !36)
// CHECK:STDOUT: !44 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.14b8745117b2bc54:ImplicitAs.a9271c1e04015f9c.Core.64ccbb8e5d9a0b8e", scope: null, file: !45, line: 32, type: !46, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !48)
// CHECK:STDOUT: !45 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !46 = !DISubroutineType(types: !47)
// CHECK:STDOUT: !47 = !{!7, !7}
// CHECK:STDOUT: !48 = !{!49}
// CHECK:STDOUT: !49 = !DILocalVariable(arg: 1, scope: !44, type: !7)
// CHECK:STDOUT: !50 = !DILocation(line: 32, column: 38, scope: !44)
