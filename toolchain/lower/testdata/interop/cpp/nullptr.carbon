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
// CHECK:STDOUT: @uninit.nullptr.ref.loc12 = internal constant ptr poison
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CPassNullptr.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_18.1.temp = alloca ptr, align 8, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_18.1.temp), !dbg !10
// CHECK:STDOUT:   call void @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4"(ptr %.loc12_18.1.temp, ptr @uninit.nullptr.ref.loc12), !dbg !10
// CHECK:STDOUT:   call void @_Z7TakePtrPi(ptr %.loc12_18.1.temp), !dbg !11
// CHECK:STDOUT:   call void @_Z11TakeNullptrDn.carbon_thunk(ptr @uninit.nullptr.ref.loc12), !dbg !12
// CHECK:STDOUT:   ret void, !dbg !13
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_Z7TakePtrPi(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CReturnNullptr.Main(ptr sret(ptr) %return) #0 !dbg !14 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr %return, ptr @uninit.nullptr.ref.loc12), !dbg !15
// CHECK:STDOUT:   ret void, !dbg !16
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr sret(ptr), ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CReturnNullptrCopy.Main(ptr sret(ptr) %return) #0 !dbg !17 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !18
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !18
// CHECK:STDOUT:   call void @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr %a.var, ptr @uninit.nullptr.ref.loc12), !dbg !19
// CHECK:STDOUT:   call void @"_COp.NullptrT.CppCompat.Core:Copy.Core"(ptr %return, ptr %a.var), !dbg !20
// CHECK:STDOUT:   ret void, !dbg !21
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CReturnConvertedNullptrIndirectly.Main(ptr sret(ptr) %return) #0 !dbg !22 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %a.var = alloca ptr, align 8, !dbg !23
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %a.var), !dbg !23
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk(ptr %a.var), !dbg !24
// CHECK:STDOUT:   call void @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4"(ptr %return, ptr %a.var), !dbg !25
// CHECK:STDOUT:   ret void, !dbg !25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CReturnConvertedNullptrDirectly.Main(ptr sret(ptr) %return) #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_28.1.temp = alloca ptr, align 8, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc32_28.1.temp), !dbg !27
// CHECK:STDOUT:   call void @_Z13ReturnNullptrv.carbon_thunk(ptr %.loc32_28.1.temp), !dbg !27
// CHECK:STDOUT:   call void @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4"(ptr %return, ptr %.loc32_28.1.temp), !dbg !28
// CHECK:STDOUT:   ret void, !dbg !28
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CConvertNullptrConstant.Main(ptr sret(ptr) %return) #0 !dbg !29 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4"(ptr %return, ptr @uninit.nullptr.ref.loc12), !dbg !30
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4"(ptr sret(ptr) %return, ptr %self) #0 !dbg !31 {
// CHECK:STDOUT:   call void @_CNone.Optional.Core.e49c05a5c9dc9c86(ptr %return), !dbg !33
// CHECK:STDOUT:   ret void, !dbg !34
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMake.NullptrT.CppCompat.Core(ptr sret(ptr))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CNone.Optional.Core.e49c05a5c9dc9c86(ptr sret(ptr) %return) #0 !dbg !35 {
// CHECK:STDOUT:   store ptr null, ptr %return, align 8, !dbg !37
// CHECK:STDOUT:   ret void, !dbg !38
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
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @"_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4", { 3, 2, 1, 0 }
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
// CHECK:STDOUT: !11 = !DILocation(line: 12, column: 3, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 14, column: 3, scope: !7)
// CHECK:STDOUT: !13 = !DILocation(line: 11, column: 1, scope: !7)
// CHECK:STDOUT: !14 = distinct !DISubprogram(name: "ReturnNullptr", linkageName: "_CReturnNullptr.Main", scope: null, file: !6, line: 17, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !15 = !DILocation(line: 18, column: 10, scope: !14)
// CHECK:STDOUT: !16 = !DILocation(line: 18, column: 3, scope: !14)
// CHECK:STDOUT: !17 = distinct !DISubprogram(name: "ReturnNullptrCopy", linkageName: "_CReturnNullptrCopy.Main", scope: null, file: !6, line: 21, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !18 = !DILocation(line: 22, column: 3, scope: !17)
// CHECK:STDOUT: !19 = !DILocation(line: 22, column: 26, scope: !17)
// CHECK:STDOUT: !20 = !DILocation(line: 23, column: 10, scope: !17)
// CHECK:STDOUT: !21 = !DILocation(line: 23, column: 3, scope: !17)
// CHECK:STDOUT: !22 = distinct !DISubprogram(name: "ReturnConvertedNullptrIndirectly", linkageName: "_CReturnConvertedNullptrIndirectly.Main", scope: null, file: !6, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !23 = !DILocation(line: 27, column: 3, scope: !22)
// CHECK:STDOUT: !24 = !DILocation(line: 27, column: 26, scope: !22)
// CHECK:STDOUT: !25 = !DILocation(line: 28, column: 3, scope: !22)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "ReturnConvertedNullptrDirectly", linkageName: "_CReturnConvertedNullptrDirectly.Main", scope: null, file: !6, line: 31, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !27 = !DILocation(line: 32, column: 10, scope: !26)
// CHECK:STDOUT: !28 = !DILocation(line: 32, column: 3, scope: !26)
// CHECK:STDOUT: !29 = distinct !DISubprogram(name: "ConvertNullptrConstant", linkageName: "_CConvertNullptrConstant.Main", scope: null, file: !6, line: 35, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !30 = !DILocation(line: 36, column: 3, scope: !29)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "Convert", linkageName: "_CConvert.NullptrT.CppCompat.Core:ImplicitAs.Core.b88d1103f417c6d4", scope: null, file: !32, line: 34, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !32 = !DIFile(filename: "{{.*}}/prelude/types/cpp/nullptr.carbon", directory: "")
// CHECK:STDOUT: !33 = !DILocation(line: 35, column: 14, scope: !31)
// CHECK:STDOUT: !34 = !DILocation(line: 35, column: 7, scope: !31)
// CHECK:STDOUT: !35 = distinct !DISubprogram(name: "None", linkageName: "_CNone.Optional.Core.e49c05a5c9dc9c86", scope: null, file: !36, line: 26, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !36 = !DIFile(filename: "{{.*}}/prelude/types/optional.carbon", directory: "")
// CHECK:STDOUT: !37 = !DILocation(line: 27, column: 12, scope: !35)
// CHECK:STDOUT: !38 = !DILocation(line: 27, column: 5, scope: !35)
