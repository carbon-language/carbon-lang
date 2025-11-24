// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/virtual_base.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/virtual_base.carbon

// --- diamond.h

struct A {
  A();
  int a;
};

struct B : virtual A {
  B();
  int b;
};

struct C : virtual A {
  C();
  int c;
};

struct D : B, C {
  int d;
};

// Layout of D (x86_64):
//
// 0:  B | 0: vptr
// 8:    | 8: int B::b
// 12: (4 bytes padding)
// 16: C | 0: vptr
// 24:   | 8: int C::c
// 28: int D::d
// 32: A | 0: int A::a
// 36: (4 bytes padding)
// 40: (end of D)

// --- use_diamond.carbon

library "[[@TEST_NAME]]";

import Cpp library "diamond.h";

fn Make() {
  // This should construct a 40-byte object, using a construtor with a `...C1`
  // mangled name for D (a complete object constructor), which should call
  // constructors with `...C2` mangled names for A, B, and C (base subobject
  // constructors).
  var d: Cpp.D = Cpp.D.D();
}

fn AccessD(d: Cpp.D) -> i32 {
  // This should read an i32 at offset 28.
  return d.d;
}

// CHECK:STDOUT: ; ModuleID = 'use_diamond.carbon'
// CHECK:STDOUT: source_filename = "use_diamond.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1DC1Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTV1D = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTT1D = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTC1D0_1B = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI1B = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS1B = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI1A = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS1A = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTC1D16_1C = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI1C = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS1C = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI1D = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS1D = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV1D = linkonce_odr dso_local unnamed_addr constant { [3 x ptr], [3 x ptr] } { [3 x ptr] [ptr inttoptr (i64 32 to ptr), ptr null, ptr @_ZTI1D], [3 x ptr] [ptr inttoptr (i64 16 to ptr), ptr inttoptr (i64 -16 to ptr), ptr @_ZTI1D] }, comdat, align 8
// CHECK:STDOUT: @_ZTT1D = linkonce_odr dso_local unnamed_addr constant [4 x ptr] [ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr], [3 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3), ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr] }, ptr @_ZTC1D0_1B, i32 0, i32 0, i32 3), ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr] }, ptr @_ZTC1D16_1C, i32 0, i32 0, i32 3), ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr], [3 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 3)], comdat, align 8
// CHECK:STDOUT: @_ZTC1D0_1B = linkonce_odr dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr inttoptr (i64 32 to ptr), ptr null, ptr @_ZTI1B] }, comdat, align 8
// CHECK:STDOUT: @_ZTI1B = linkonce_odr dso_local constant { ptr, ptr, i32, i32, ptr, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), ptr @_ZTS1B, i32 0, i32 1, ptr @_ZTI1A, i64 -6141 }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS1B = linkonce_odr dso_local constant [3 x i8] c"1B\00", comdat, align 1
// CHECK:STDOUT: @_ZTI1A = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), ptr @_ZTS1A }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS1A = linkonce_odr dso_local constant [3 x i8] c"1A\00", comdat, align 1
// CHECK:STDOUT: @_ZTC1D16_1C = linkonce_odr dso_local unnamed_addr constant { [3 x ptr] } { [3 x ptr] [ptr inttoptr (i64 16 to ptr), ptr null, ptr @_ZTI1C] }, comdat, align 8
// CHECK:STDOUT: @_ZTI1C = linkonce_odr dso_local constant { ptr, ptr, i32, i32, ptr, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), ptr @_ZTS1C, i32 0, i32 1, ptr @_ZTI1A, i64 -6141 }, comdat, align 8
// CHECK:STDOUT: @_ZTS1C = linkonce_odr dso_local constant [3 x i8] c"1C\00", comdat, align 1
// CHECK:STDOUT: @_ZTI1D = linkonce_odr dso_local constant { ptr, ptr, i32, i32, ptr, i64, ptr, i64 } { ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), ptr @_ZTS1D, i32 2, i32 2, ptr @_ZTI1B, i64 2, ptr @_ZTI1C, i64 4098 }, comdat, align 8
// CHECK:STDOUT: @_ZTS1D = linkonce_odr dso_local constant [3 x i8] c"1D\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMake.Main() #0 !dbg !7 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %d.var = alloca [40 x i8], align 1, !dbg !10
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %d.var), !dbg !10
// CHECK:STDOUT:   call void @_ZN1DC1Ev.carbon_thunk(ptr %d.var), !dbg !11
// CHECK:STDOUT:   ret void, !dbg !12
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CAccessD.Main(ptr %d) #0 !dbg !13 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc16_11.1.d = getelementptr inbounds nuw [40 x i8], ptr %d, i32 0, i32 28, !dbg !14
// CHECK:STDOUT:   %.loc16_11.2 = load i32, ptr %.loc16_11.1.d, align 4, !dbg !14
// CHECK:STDOUT:   ret i32 %.loc16_11.2, !dbg !15
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress
// CHECK:STDOUT: define dso_local void @_ZN1DC1Ev.carbon_thunk(ptr %return) #2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8
// CHECK:STDOUT:   call void @_ZN1DC1Ev(ptr nonnull align 8 dereferenceable(32) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress noinline optnone
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1DC1Ev(ptr nonnull align 8 dereferenceable(32) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %0 = getelementptr inbounds i8, ptr %this1, i64 32
// CHECK:STDOUT:   call void @_ZN1AC2Ev(ptr nonnull align 4 dereferenceable(4) %0)
// CHECK:STDOUT:   call void @_ZN1BC2Ev(ptr nonnull align 8 dereferenceable(12) %this1, ptr getelementptr inbounds ([4 x ptr], ptr @_ZTT1D, i64 0, i64 1))
// CHECK:STDOUT:   %1 = getelementptr inbounds i8, ptr %this1, i64 16
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr nonnull align 8 dereferenceable(12) %1, ptr getelementptr inbounds ([4 x ptr], ptr @_ZTT1D, i64 0, i64 2))
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr], [3 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3), ptr %this1, align 8
// CHECK:STDOUT:   %add.ptr = getelementptr inbounds i8, ptr %this1, i64 16
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr], [3 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 3), ptr %add.ptr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1AC2Ev(ptr nonnull align 4 dereferenceable(4)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1BC2Ev(ptr nonnull align 8 dereferenceable(12), ptr) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN1CC2Ev(ptr nonnull align 8 dereferenceable(12), ptr) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr], [3 x ptr] }, ptr @_ZTV1D, i32 0, i32 0, i32 3), { 1, 0 }
// CHECK:STDOUT: uselistorder ptr getelementptr inbounds inrange(-24, 0) ({ [3 x ptr], [3 x ptr] }, ptr @_ZTV1D, i32 0, i32 1, i32 3), { 1, 0 }
// CHECK:STDOUT: uselistorder ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2), { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZTI1B, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZTI1A, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZTI1C, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_ZTI1D, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #2 = { alwaysinline mustprogress "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #3 = { mustprogress noinline optnone "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="0" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
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
// CHECK:STDOUT: !6 = !DIFile(filename: "use_diamond.carbon", directory: "")
// CHECK:STDOUT: !7 = distinct !DISubprogram(name: "Make", linkageName: "_CMake.Main", scope: null, file: !6, line: 6, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !8 = !DISubroutineType(types: !9)
// CHECK:STDOUT: !9 = !{}
// CHECK:STDOUT: !10 = !DILocation(line: 11, column: 3, scope: !7)
// CHECK:STDOUT: !11 = !DILocation(line: 11, column: 18, scope: !7)
// CHECK:STDOUT: !12 = !DILocation(line: 6, column: 1, scope: !7)
// CHECK:STDOUT: !13 = distinct !DISubprogram(name: "AccessD", linkageName: "_CAccessD.Main", scope: null, file: !6, line: 14, type: !8, spFlags: DISPFlagDefinition, unit: !5)
// CHECK:STDOUT: !14 = !DILocation(line: 16, column: 10, scope: !13)
// CHECK:STDOUT: !15 = !DILocation(line: 16, column: 3, scope: !13)
