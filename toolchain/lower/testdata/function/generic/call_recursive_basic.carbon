// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/call_recursive_basic.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/call_recursive_basic.carbon

class C {}

// The two specifics for recursive function F, when T is a pointer, could be
// deduplicated.
fn F[T:! type](x: T, count: i32) -> T {
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
// CHECK:STDOUT: define void @_CM.Main() !dbg !4 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.var = alloca i32, align 4, !dbg !7
// CHECK:STDOUT:   %m.var = alloca double, align 8, !dbg !7
// CHECK:STDOUT:   %ptr_i32.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   %ptr_f64.var = alloca ptr, align 8, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 4, ptr %n.var), !dbg !7
// CHECK:STDOUT:   store i32 0, ptr %n.var, align 4, !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %m.var), !dbg !7
// CHECK:STDOUT:   store double 1.000000e+00, ptr %m.var, align 8, !dbg !8
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_i32.var), !dbg !7
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(i64 8, ptr %ptr_f64.var), !dbg !7
// CHECK:STDOUT:   %.loc28_5 = load i32, ptr %n.var, align 4, !dbg !9
// CHECK:STDOUT:   %F.call.loc28 = call i32 @_CF.Main.b88d1103f417c6d4(i32 %.loc28_5, i32 0), !dbg !10
// CHECK:STDOUT:   %.loc29_5 = load double, ptr %m.var, align 8, !dbg !11
// CHECK:STDOUT:   %F.call.loc29 = call double @_CF.Main.66be507887ceee78(double %.loc29_5, i32 0), !dbg !12
// CHECK:STDOUT:   %.loc30_5 = load ptr, ptr %ptr_i32.var, align 8, !dbg !13
// CHECK:STDOUT:   %F.call.loc30 = call ptr @_CF.Main.e8193710fd35b608(ptr %.loc30_5, i32 0), !dbg !14
// CHECK:STDOUT:   %.loc31_5 = load ptr, ptr %ptr_f64.var, align 8, !dbg !15
// CHECK:STDOUT:   %F.call.loc31 = call ptr @_CF.Main.04bf2edaaa84aa22(ptr %.loc31_5, i32 0), !dbg !16
// CHECK:STDOUT:   ret void, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #0
// CHECK:STDOUT:
// CHECK:STDOUT: define i32 @_CF.Main.b88d1103f417c6d4(i32 %x, i32 %count) !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 0, !dbg !19
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !20
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret i32 %x, !dbg !21
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !22
// CHECK:STDOUT:   %F.call = call i32 @_CF.Main.b88d1103f417c6d4(i32 %x, i32 %int.sadd), !dbg !23
// CHECK:STDOUT:   ret i32 %F.call, !dbg !24
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define double @_CF.Main.66be507887ceee78(double %x, i32 %count) !dbg !25 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 0, !dbg !26
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !27
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret double %x, !dbg !28
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !29
// CHECK:STDOUT:   %F.call = call double @_CF.Main.66be507887ceee78(double %x, i32 %int.sadd), !dbg !30
// CHECK:STDOUT:   ret double %F.call, !dbg !31
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CF.Main.e8193710fd35b608(ptr %x, i32 %count) !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 0, !dbg !33
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !34
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !35
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !36
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.e8193710fd35b608(ptr %x, i32 %int.sadd), !dbg !37
// CHECK:STDOUT:   ret ptr %F.call, !dbg !38
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: define ptr @_CF.Main.04bf2edaaa84aa22(ptr %x, i32 %count) !dbg !39 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %int.greater = icmp sgt i32 %count, 0, !dbg !40
// CHECK:STDOUT:   br i1 %int.greater, label %if.then, label %if.else, !dbg !41
// CHECK:STDOUT:
// CHECK:STDOUT: if.then:                                          ; preds = %entry
// CHECK:STDOUT:   ret ptr %x, !dbg !42
// CHECK:STDOUT:
// CHECK:STDOUT: if.else:                                          ; preds = %entry
// CHECK:STDOUT:   %int.sadd = add i32 %count, 1, !dbg !43
// CHECK:STDOUT:   %F.call = call ptr @_CF.Main.04bf2edaaa84aa22(ptr %x, i32 %int.sadd), !dbg !44
// CHECK:STDOUT:   ret ptr %F.call, !dbg !45
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1}
// CHECK:STDOUT: !llvm.dbg.cu = !{!2}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = distinct !DICompileUnit(language: DW_LANG_C, file: !3, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !3 = !DIFile(filename: "call_recursive_basic.carbon", directory: "")
// CHECK:STDOUT: !4 = distinct !DISubprogram(name: "M", linkageName: "_CM.Main", scope: null, file: !3, line: 22, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !5 = !DISubroutineType(types: !6)
// CHECK:STDOUT: !6 = !{}
// CHECK:STDOUT: !7 = !DILocation(line: 23, column: 3, scope: !4)
// CHECK:STDOUT: !8 = !DILocation(line: 24, column: 3, scope: !4)
// CHECK:STDOUT: !9 = !DILocation(line: 28, column: 5, scope: !4)
// CHECK:STDOUT: !10 = !DILocation(line: 28, column: 3, scope: !4)
// CHECK:STDOUT: !11 = !DILocation(line: 29, column: 5, scope: !4)
// CHECK:STDOUT: !12 = !DILocation(line: 29, column: 3, scope: !4)
// CHECK:STDOUT: !13 = !DILocation(line: 30, column: 5, scope: !4)
// CHECK:STDOUT: !14 = !DILocation(line: 30, column: 3, scope: !4)
// CHECK:STDOUT: !15 = !DILocation(line: 31, column: 5, scope: !4)
// CHECK:STDOUT: !16 = !DILocation(line: 31, column: 3, scope: !4)
// CHECK:STDOUT: !17 = !DILocation(line: 22, column: 1, scope: !4)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.b88d1103f417c6d4", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !19 = !DILocation(line: 16, column: 7, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 16, column: 6, scope: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 17, column: 5, scope: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 19, column: 15, scope: !18)
// CHECK:STDOUT: !23 = !DILocation(line: 19, column: 10, scope: !18)
// CHECK:STDOUT: !24 = !DILocation(line: 19, column: 3, scope: !18)
// CHECK:STDOUT: !25 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.66be507887ceee78", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !26 = !DILocation(line: 16, column: 7, scope: !25)
// CHECK:STDOUT: !27 = !DILocation(line: 16, column: 6, scope: !25)
// CHECK:STDOUT: !28 = !DILocation(line: 17, column: 5, scope: !25)
// CHECK:STDOUT: !29 = !DILocation(line: 19, column: 15, scope: !25)
// CHECK:STDOUT: !30 = !DILocation(line: 19, column: 10, scope: !25)
// CHECK:STDOUT: !31 = !DILocation(line: 19, column: 3, scope: !25)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.e8193710fd35b608", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !33 = !DILocation(line: 16, column: 7, scope: !32)
// CHECK:STDOUT: !34 = !DILocation(line: 16, column: 6, scope: !32)
// CHECK:STDOUT: !35 = !DILocation(line: 17, column: 5, scope: !32)
// CHECK:STDOUT: !36 = !DILocation(line: 19, column: 15, scope: !32)
// CHECK:STDOUT: !37 = !DILocation(line: 19, column: 10, scope: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 19, column: 3, scope: !32)
// CHECK:STDOUT: !39 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main.04bf2edaaa84aa22", scope: null, file: !3, line: 15, type: !5, spFlags: DISPFlagDefinition, unit: !2)
// CHECK:STDOUT: !40 = !DILocation(line: 16, column: 7, scope: !39)
// CHECK:STDOUT: !41 = !DILocation(line: 16, column: 6, scope: !39)
// CHECK:STDOUT: !42 = !DILocation(line: 17, column: 5, scope: !39)
// CHECK:STDOUT: !43 = !DILocation(line: 19, column: 15, scope: !39)
// CHECK:STDOUT: !44 = !DILocation(line: 19, column: 10, scope: !39)
// CHECK:STDOUT: !45 = !DILocation(line: 19, column: 3, scope: !39)
