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
// CHECK:STDOUT:   %Rand.call.loc7_15 = call i32 @_CRand.Main(), !dbg !7
// CHECK:STDOUT:   %Rand.call.loc7_24 = call i32 @_CRand.Main(), !dbg !8
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %Rand.call.loc7_15, %Rand.call.loc7_24, !dbg !7
// CHECK:STDOUT:   ret i32 %Int.as.AddWith.impl.Op.call, !dbg !9
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: minsize nounwind optsize
// CHECK:STDOUT: define i32 @_CCallNoInlineWithOptSize.Main() #0 !dbg !10 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %NoInlineWithOz.call = call i32 @_CNoInlineWithOz.Main(), !dbg !11
// CHECK:STDOUT:   ret i32 %NoInlineWithOz.call, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: minsize nounwind optsize
// CHECK:STDOUT: define void @_CVectorizedWithOptSpeed.Main(ptr %a) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !14
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !14
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !14
// CHECK:STDOUT:   br label %while.cond, !dbg !15
// CHECK:STDOUT:
// CHECK:STDOUT: while.cond:                                       ; preds = %while.body, %entry
// CHECK:STDOUT:   %.loc18_10 = load i32, ptr %n.var, align 4, !dbg !16
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Less.call = icmp slt i32 %.loc18_10, 65536, !dbg !16
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Less.call, label %while.body, label %while.done, !dbg !15
// CHECK:STDOUT:
// CHECK:STDOUT: while.body:                                       ; preds = %while.cond
// CHECK:STDOUT:   %.loc19_10 = load i32, ptr %n.var, align 4, !dbg !17
// CHECK:STDOUT:   %.loc19_11.array.index = getelementptr inbounds [65536 x i32], ptr %a, i32 0, i32 %.loc19_10, !dbg !18
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call = load i32, ptr %.loc19_11.array.index, align 4, !dbg !18
// CHECK:STDOUT:   %Int.as.MulAssignWith.impl.Op.call1 = mul i32 %Int.as.MulAssignWith.impl.Op.call, 2, !dbg !18
// CHECK:STDOUT:   store i32 %Int.as.MulAssignWith.impl.Op.call1, ptr %.loc19_11.array.index, align 4, !dbg !18
// CHECK:STDOUT:   call void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %n.var), !dbg !19
// CHECK:STDOUT:   br label %while.cond, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: while.done:                                       ; preds = %while.cond
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: minsize nounwind optsize
// CHECK:STDOUT: define linkonce_odr void @"_COp.Int.Core:Inc.Core.be1e879c1ad406d8"(ptr %self) #0 !dbg !22 {
// CHECK:STDOUT:   call void @"_COp:thunk.Int.Core:AddAssignWith.Core.5dfb78ae56583d8e"(ptr %self, i32 1), !dbg !24
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline minsize nounwind optsize
// CHECK:STDOUT: define linkonce_odr void @"_COp:thunk.Int.Core:AddAssignWith.Core.5dfb78ae56583d8e"(ptr %self, i32 %other) #2 !dbg !26 {
// CHECK:STDOUT:   %1 = call i32 @"_CConvert.225258f1a45e9386:ImplicitAs.Core.5450dc8e8b8e0899"(i32 %other), !dbg !27
// CHECK:STDOUT:   %2 = load i32, ptr %self, align 4, !dbg !28
// CHECK:STDOUT:   %3 = add i32 %2, %1, !dbg !28
// CHECK:STDOUT:   store i32 %3, ptr %self, align 4, !dbg !28
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: minsize nounwind optsize
// CHECK:STDOUT: define linkonce_odr i32 @"_CConvert.225258f1a45e9386:ImplicitAs.Core.5450dc8e8b8e0899"(i32 %self) #0 !dbg !29 {
// CHECK:STDOUT:   ret i32 %self, !dbg !31
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
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "foo.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "NoInlineWithOz", linkageName: "_CNoInlineWithOz.Main", scope: null, file: !3, line: 6, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 7, column: 10, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 7, column: 19, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 7, column: 3, scope: !4)
// CHECK:STDOUT: !10 = distinct !DISubprogram(name: "CallNoInlineWithOptSize", linkageName: "_CCallNoInlineWithOptSize.Main", scope: null, file: !3, line: 10, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !11 = !DILocation(line: 12, column: 10, scope: !10)
// CHECK:STDOUT: !12 = !DILocation(line: 12, column: 3, scope: !10)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "VectorizedWithOptSpeed", linkageName: "_CVectorizedWithOptSpeed.Main", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !14 = !DILocation(line: 17, column: 3, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 18, column: 9, scope: !13)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 10, scope: !13)
// CHECK:STDOUT: !17 = !DILocation(line: 19, column: 10, scope: !13)
// CHECK:STDOUT: !18 = !DILocation(line: 19, column: 5, scope: !13)
// CHECK:STDOUT: !19 = !DILocation(line: 20, column: 5, scope: !13)
// CHECK:STDOUT: !20 = !DILocation(line: 18, column: 3, scope: !13)
// CHECK:STDOUT: !21 = !DILocation(line: 15, column: 1, scope: !13)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "Op", linkageName: "_COp.Int.Core:Inc.Core.be1e879c1ad406d8", scope: null, file: !23, line: 339, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = !DIFile(filename: "{{.*}}/prelude/types/int.carbon", directory: "")
// CHECK:STDOUT: !24 = !DILocation(line: 341, column: 5, scope: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 339, column: 3, scope: !22)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Int.Core:AddAssignWith.Core.5dfb78ae56583d8e", scope: null, file: !23, line: 275, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !27 = !DILocation(line: 4294967295, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 275, column: 3, scope: !26)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.225258f1a45e9386:ImplicitAs.Core.5450dc8e8b8e0899", scope: null, file: !30, line: 24, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !30 = !DIFile(filename: "{{.*}}/prelude/operators/as.carbon", directory: "")
// CHECK:STDOUT: !31 = !DILocation(line: 24, column: 38, scope: !29)
