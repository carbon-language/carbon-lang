// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_basic.carbon

class C {
  impl as Core.Copy {
    fn Op[self: Self]() -> Self { return {}; }
  }
}

// The two specifics for recursive function F, when T is a pointer, could be
// deduplicated.
fn F[T:! Core.Copy](x: T, count: i32) -> T {
  if (count > 0) {
    return x;
  }
  return F(x, count + 1);
}

fn M() {
  var n: i32 = 0;
  var m: f64 = 1.0;
  var ptr_i32 : i32*;
  var ptr_f64 : f64*;

  F(n, 0);
  F(m, 0);
  F(ptr_i32, 0);
  F(ptr_f64, 0);

  // TODO: Crashes, hits assertion
  // `(Args.size() == FTy->getNumParams() || (FTy->isVarArg() && Args.size() > FTy->getNumParams())) && "Calling a function with bad signature!"'

  // var c: C;
  // F(c, 0);
}

// CHECK:STDOUT: ; ModuleID = 'call_recursive_basic.carbon'
// CHECK:STDOUT: source_filename = "call_recursive_basic.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @C.val.loc15_44 = internal constant {} zeroinitializer
// CHECK:STDOUT:
// CHECK:STDOUT: define void @"_COp.C.Main:Copy.Core"(ptr sret({}) %return, ptr %self) !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 1 %return, ptr align 1 @C.val.loc15_44, i64 0, i1 false), !dbg !7
// CHECK:STDOUT:   ret void, !dbg !7
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define void @_CM.Main() !dbg !8 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !9
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !10
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !11
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %n.var), !dbg !9
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !10
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_i32.var), !dbg !11
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %ptr_f64.var), !dbg !12
// CHECK:STDOUT:   %.loc34_5 = load i32, ptr %n.var, align 4, !dbg !13
// CHECK:STDOUT:   %F.call.loc34 = call i32 @_CF.Main.de631560529e9861(i32 %.loc34_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc35_5 = load double, ptr %m.var, align 8, !dbg !15
// CHECK:STDOUT:   %F.call.loc35 = call double @_CF.Main.3836ccc537683c19(double %.loc35_5, i32 0), !dbg !16
// CHECK:STDOUT:   %.loc36_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !17
// CHECK:STDOUT:   %F.call.loc36 = call ptr @_CF.Main.a194de4a3b8dbd0c(ptr %.loc36_5, i32 0), !dbg !18
// CHECK:STDOUT:   %.loc37_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !19
// CHECK:STDOUT:   %F.call.loc37 = call ptr @_CF.Main.a194de4a3b8dbd0c(ptr %.loc37_5, i32 0), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #0
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr i32 @_CF.Main.de631560529e9861(i32 %x, i32 %count) !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !23
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !24
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !25
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !26
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.de631560529e9861(i32 %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !27
// CHECK:STDOUT:   ret i32 %F.call, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr double @_CF.Main.3836ccc537683c19(double %x, i32 %count) !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !30
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !31
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !32
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !33
// CHECK:STDOUT:   %F.call = call double @_CF.Main.3836ccc537683c19(double %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !34
// CHECK:STDOUT:   ret double %F.call, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define linkonce_odr ptr @_CF.Main.a194de4a3b8dbd0c(ptr %x, i32 %count) !dbg !36 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Int.as.OrderedWith.impl.Greater.call = icmp sgt i32 %count, 0, !dbg !37
// CHECK:STDOUT:   br i1 %Int.as.OrderedWith.impl.Greater.call, label %if.then, label %if.else, !dbg !38
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !39
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %Int.as.AddWith.impl.Op.call = add i32 %count, 1, !dbg !40
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.a194de4a3b8dbd0c(ptr %x, i32 %Int.as.AddWith.impl.Op.call), !dbg !41
// CHECK:STDOUT:   ret ptr %F.call, !dbg !42
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CF.Main.a194de4a3b8dbd0c, { 1, 2, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_basic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "Op", linkageName: "_COp.C.Main:Copy.Core", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 15, column: 35, scope: !4)
// CHECK:STDOUT: !8 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 28, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !9 = !DILocation(line: 29, column: 3, scope: !8)
// CHECK:STDOUT: !10 = !DILocation(line: 30, column: 3, scope: !8)
// CHECK:STDOUT: !11 = !DILocation(line: 31, column: 3, scope: !8)
// CHECK:STDOUT: !12 = !DILocation(line: 32, column: 3, scope: !8)
// CHECK:STDOUT: !13 = !DILocation(line: 34, column: 5, scope: !8)
// CHECK:STDOUT: !14 = !DILocation(line: 34, column: 3, scope: !8)
// CHECK:STDOUT: !15 = !DILocation(line: 35, column: 5, scope: !8)
// CHECK:STDOUT: !16 = !DILocation(line: 35, column: 3, scope: !8)
// CHECK:STDOUT: !17 = !DILocation(line: 36, column: 5, scope: !8)
// CHECK:STDOUT: !18 = !DILocation(line: 36, column: 3, scope: !8)
// CHECK:STDOUT: !19 = !DILocation(line: 37, column: 5, scope: !8)
// CHECK:STDOUT: !20 = !DILocation(line: 37, column: 3, scope: !8)
// CHECK:STDOUT: !21 = !DILocation(line: 28, column: 1, scope: !8)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.de631560529e9861", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !23 = !DILocation(line: 22, column: 7, scope: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 22, column: 6, scope: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 23, column: 5, scope: !22)
// CHECK:STDOUT: !26 = !DILocation(line: 25, column: 15, scope: !22)
// CHECK:STDOUT: !27 = !DILocation(line: 25, column: 10, scope: !22)
// CHECK:STDOUT: !28 = !DILocation(line: 25, column: 3, scope: !22)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.3836ccc537683c19", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !30 = !DILocation(line: 22, column: 7, scope: !29)
// CHECK:STDOUT: !31 = !DILocation(line: 22, column: 6, scope: !29)
// CHECK:STDOUT: !32 = !DILocation(line: 23, column: 5, scope: !29)
// CHECK:STDOUT: !33 = !DILocation(line: 25, column: 15, scope: !29)
// CHECK:STDOUT: !34 = !DILocation(line: 25, column: 10, scope: !29)
// CHECK:STDOUT: !35 = !DILocation(line: 25, column: 3, scope: !29)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.a194de4a3b8dbd0c", scope: null, file: !3, line: 21, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !37 = !DILocation(line: 22, column: 7, scope: !36)
// CHECK:STDOUT: !38 = !DILocation(line: 22, column: 6, scope: !36)
// CHECK:STDOUT: !39 = !DILocation(line: 23, column: 5, scope: !36)
// CHECK:STDOUT: !40 = !DILocation(line: 25, column: 15, scope: !36)
// CHECK:STDOUT: !41 = !DILocation(line: 25, column: 10, scope: !36)
// CHECK:STDOUT: !42 = !DILocation(line: 25, column: 3, scope: !36)
