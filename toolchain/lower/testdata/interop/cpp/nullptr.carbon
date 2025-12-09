// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/nullptr.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/nullptr.carbon

// --- simple.carbon

import Cpp inline '''
using nullptr_t = decltype(nullptr);

void TakePtr(int *p);
void TakeNullptr(nullptr_t n);

nullptr_t ReturnNullptr();
''';

fn PassNullptr() {
  Cpp.TakePtr(Cpp.nullptr);

  Cpp.TakeNullptr(Cpp.nullptr);
}

fn ReturnNullptr() -> Cpp.nullptr_t {
  return Cpp.nullptr;
}

fn ReturnNullptrCopy() -> Cpp.nullptr_t {
  var a: Cpp.nullptr_t = Cpp.nullptr;
  return a;
}

fn ReturnConvertedNullptrIndirectly() -> Core.Optional(i32*) {
  var a: Cpp.nullptr_t = Cpp.ReturnNullptr();
  return a;
}

fn ReturnConvertedNullptrDirectly() -> Core.Optional(i32*) {
  return Cpp.ReturnNullptr();
}

fn ConvertNullptrConstant() -> Core.Optional(i32*) {
  return Cpp.nullptr;
}

// CHECK:STDOUT: ; ModuleID = 'simple.carbon'
// CHECK:STDOUT: source_filename = "simple.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNullptr.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_18.2.temp = alloca ptr, align 8, !dbg !10
// CHECK:STDOUT:   %.loc14_22.1.temp = alloca ptr, align 8, !dbg !11
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4"(ptr poison), !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.2.temp), !dbg !10
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, ptr %.loc12_18.2.temp, align 8, !dbg !10
// CHECK:STDOUT:   %.loc12_18.4 = load ptr, ptr %.loc12_18.2.temp, align 8, !dbg !10
// CHECK:STDOUT:   call void @_Z7TakePtrPi(ptr %.loc12_18.4), !dbg !12
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_22.1.temp), !dbg !11
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !11
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, ptr %.loc14_22.1.temp, align 8, !dbg !11
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn.carbon_thunk(ptr %.loc14_22.1.temp), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrPi(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptr.Main() #0 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !16
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call, !dbg !17
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnNullptrCopy.Main() #0 !dbg !18 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !19
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !19
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr poison), !dbg !20
// CHECK:STDOUT:   store ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc22, ptr %a.var, align 8, !dbg !19
// CHECK:STDOUT:   %.loc23 = load ptr, ptr %a.var, align 8, !dbg !21
// CHECK:STDOUT:   %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23 = call ptr @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr %.loc23), !dbg !21
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.Copy.impl.Op.call.loc23, !dbg !22
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrIndirectly.Main() #0 !dbg !23 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !24
// CHECK:STDOUT:   %.loc27_44.1.temp = alloca ptr, align 8, !dbg !25
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !24
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc27_44.1.temp), !dbg !25
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk(ptr %.loc27_44.1.temp), !dbg !25
// CHECK:STDOUT:   %.loc27_44.2 = load ptr, ptr %.loc27_44.1.temp, align 8, !dbg !25
// CHECK:STDOUT:   store ptr %.loc27_44.2, ptr %a.var, align 8, !dbg !24
// CHECK:STDOUT:   %.loc28_10 = load ptr, ptr %a.var, align 8, !dbg !26
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4"(ptr %.loc28_10), !dbg !27
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !27
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CReturnConvertedNullptrDirectly.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_28.1.temp = alloca ptr, align 8, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_28.1.temp), !dbg !29
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk(ptr %.loc32_28.1.temp), !dbg !29
// CHECK:STDOUT:   %.loc32_28.2 = load ptr, ptr %.loc32_28.1.temp, align 8, !dbg !29
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4"(ptr %.loc32_28.2), !dbg !30
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @_CConvertNullptrConstant.Main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call = call ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4"(ptr poison), !dbg !32
// CHECK:STDOUT:   ret ptr %Cpp.nullptr_t.as.ImplicitAs.impl.Convert.call, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4"(ptr %self) #0 !dbg !33 {
// CHECK:STDOUT:   %1 = call ptr @_CNone.Optional.Core.e49c05a5c9dc9c86(), !dbg !35
// CHECK:STDOUT:   ret ptr %1, !dbg !36
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_CMake.NullptrT.CppCompat.Core()
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CNone.Optional.Core.e49c05a5c9dc9c86() #0 !dbg !37 {
// CHECK:STDOUT:   ret ptr null, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z11TakeNullptrDn.carbon_thunk(ptr %n) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %n.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %n, ptr %n.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %n.addr, align 8
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn(ptr null)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z11TakeNullptrDn(ptr) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_Z13ReturnNullptrv.carbon_thunk(ptr %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8
// CHECK:STDOUT:   %call = call ptr @_Z13ReturnNullptrv()
// CHECK:STDOUT:   store ptr %call, ptr %0, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare ptr @_Z13ReturnNullptrv() #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4", { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "simple.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "PassNullptr", linkageName: "_CPassNullptr.Main", scope: null, file: !6, line: 11, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 12, column: 15, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 14, column: 19, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 12, column: 3, scope: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 14, column: 3, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 11, column: 1, scope: !7)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "ReturnNullptr", linkageName: "_CReturnNullptr.Main", scope: null, file: !6, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 10, scope: !15)
// CHECK:STDOUT: !17 = !DILocation(line: 18, column: 3, scope: !15)
// CHECK:STDOUT: !18 = distinct !DISubprogram(name: "ReturnNullptrCopy", linkageName: "_CReturnNullptrCopy.Main", scope: null, file: !6, line: 21, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 3, scope: !18)
// CHECK:STDOUT: !20 = !DILocation(line: 22, column: 26, scope: !18)
// CHECK:STDOUT: !21 = !DILocation(line: 23, column: 10, scope: !18)
// CHECK:STDOUT: !22 = !DILocation(line: 23, column: 3, scope: !18)
// CHECK:STDOUT: !23 = distinct !DISubprogram(name: "ReturnConvertedNullptrIndirectly", linkageName: "_CReturnConvertedNullptrIndirectly.Main", scope: null, file: !6, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !24 = !DILocation(line: 27, column: 3, scope: !23)
// CHECK:STDOUT: !25 = !DILocation(line: 27, column: 26, scope: !23)
// CHECK:STDOUT: !26 = !DILocation(line: 28, column: 10, scope: !23)
// CHECK:STDOUT: !27 = !DILocation(line: 28, column: 3, scope: !23)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "ReturnConvertedNullptrDirectly", linkageName: "_CReturnConvertedNullptrDirectly.Main", scope: null, file: !6, line: 31, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !29 = !DILocation(line: 32, column: 10, scope: !28)
// CHECK:STDOUT: !30 = !DILocation(line: 32, column: 3, scope: !28)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "ConvertNullptrConstant", linkageName: "_CConvertNullptrConstant.Main", scope: null, file: !6, line: 35, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !32 = !DILocation(line: 36, column: 3, scope: !31)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4", scope: null, file: !34, line: 45, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !34 = !DIFile(filename: "{{.*}}/prelude/types/cpp/nullptr.carbon", directory: "")
// CHECK:STDOUT: !35 = !DILocation(line: 46, column: 14, scope: !33)
// CHECK:STDOUT: !36 = !DILocation(line: 46, column: 7, scope: !33)
// CHECK:STDOUT: !37 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.e49c05a5c9dc9c86", scope: null, file: !38, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !38 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !39 = !DILocation(line: 27, column: 5, scope: !37)
