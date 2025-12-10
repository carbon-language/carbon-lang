// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/constructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/constructor.carbon

// ============================================================================
// Default constructor
// ============================================================================

// --- default.h

class C {
 public:
  C() : x_(8), y_(9) {}

 private:
  int x_;
  int y_;
};

// --- import_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  let c: Cpp.C = Cpp.C.C();
}

// ============================================================================
// Copy constructor
// ============================================================================

// --- copy.h

class Copy {
 public:
  Copy();
  Copy(const Copy&);

 private:
  int n;
};

// --- call_copy.carbon

library "[[@TEST_NAME]]";

import Cpp library "copy.h";

fn Copy(c: Cpp.Copy) -> Cpp.Copy {
  return c;
}

// CHECK:STDOUT: ; ModuleID = 'import_default.carbon'
// CHECK:STDOUT: source_filename = "import_default.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.C = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC1Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_26.1.temp = alloca [8 x i8], align 1, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_26.1.temp), !dbg !10
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk(ptr %.loc7_26.1.temp), !dbg !10
// CHECK:STDOUT:   call void @_ZN1CD1Ev(ptr %.loc7_26.1.temp), !dbg !10
// CHECK:STDOUT:   ret void, !dbg !11
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1CD1Ev(ptr)
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_ZN1CC1Ev.carbon_thunk(ptr %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8
// CHECK:STDOUT:   call void @_ZN1CC1Ev(ptr nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC1Ev(ptr nonnull align 4 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr nonnull align 4 dereferenceable(8) %this1)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline nounwind optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr nonnull align 4 dereferenceable(8) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { mustprogress noinline optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { mustprogress noinline nounwind optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "import_default.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{null}
// CHECK:STDOUT: !10 = !DILocation(line: 7, column: 18, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: ; ModuleID = 'call_copy.carbon'
// CHECK:STDOUT: source_filename = "call_copy.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.Main(ptr sret([4 x i8]) %return, ptr %c) #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.1.temp = alloca [4 x i8], align 1, !dbg !13
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_10.1.temp), !dbg !13
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk(ptr %c, ptr %return), !dbg !13
// CHECK:STDOUT:   ret void, !dbg !14
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define void @"_COp:thunk.Copy.Cpp"(ptr sret([4 x i8]) %return, ptr %self) #1 !dbg !15 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk(ptr %self, ptr %return), !dbg !18
// CHECK:STDOUT:   ret void, !dbg !18
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_ZN4CopyC1ERKS_.carbon_thunk(ptr %0, ptr %return) #3 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_(ptr nonnull align 4 dereferenceable(4) %1, ptr nonnull align 4 dereferenceable(4) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN4CopyC1ERKS_(ptr nonnull align 4 dereferenceable(4), ptr nonnull align 4 dereferenceable(4)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.module.flags = !{!0, !1, !2, !3, !4}
// CHECK:STDOUT: !llvm.dbg.cu = !{!5}
// CHECK:STDOUT:
// CHECK:STDOUT: !0 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !1 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !2 = !{i32 1, !"wchar_size", i32 4}
// CHECK:STDOUT: !3 = !{i32 8, !"PIC Level", i32 0}
// CHECK:STDOUT: !4 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !5 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !6, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !6 = !DIFile(filename: "call_copy.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !11)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{!10, !10}
// CHECK:STDOUT: !10 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 8)
// CHECK:STDOUT: !11 = !{!12}
// CHECK:STDOUT: !12 = !DILocalVariable(arg: 1, scope: !7, type: !10)
// CHECK:STDOUT: !13 = !DILocation(line: 7, column: 10, scope: !7)
// CHECK:STDOUT: !14 = !DILocation(line: 7, column: 3, scope: !7)
// CHECK:STDOUT: !15 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk.Copy.Cpp", scope: null, file: !6, type: !8, spFlags: DISPFlagDefinition, unit: !5, retainedNodes: !16)
// CHECK:STDOUT: !16 = !{!17}
// CHECK:STDOUT: !17 = !DILocalVariable(arg: 1, scope: !15, type: !10)
// CHECK:STDOUT: !18 = !DILocation(line: 0, scope: !15)
