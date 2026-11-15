// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/interop/cpp/class/import/constructor.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/interop/cpp/class/import/constructor.carbon

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
  let _: Cpp.C = Cpp.C.C();
}

fn G() {
  let _: Cpp.C = ();
}

// --- implicit_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "default.h";

fn F() {
  var _: Cpp.C;
}

// --- defaulted_default.h

struct ImplicitlyDefaultedDefault {
  int x_ = 1;
};

struct ExplicitlyDefaultedDefault {
  ExplicitlyDefaultedDefault() = default;
  int x_ = 1;
};

// --- defaulted_default.carbon

library "[[@TEST_NAME]]";

import Cpp library "defaulted_default.h";

fn F() {
  let _: Cpp.ImplicitlyDefaultedDefault =
      Cpp.ImplicitlyDefaultedDefault.ImplicitlyDefaultedDefault();
  var _: Cpp.ImplicitlyDefaultedDefault;

  let _: Cpp.ExplicitlyDefaultedDefault =
      Cpp.ExplicitlyDefaultedDefault.ExplicitlyDefaultedDefault();
  var _: Cpp.ExplicitlyDefaultedDefault;
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

// ============================================================================
// C1 constructor
// ============================================================================

// --- header.h

struct Base { };
struct Derived: virtual Base {
};

// --- call_c1_in_thunk.carbon

library "[[@TEST_NAME]]";

import Cpp library "header.h";

fn F() {
  let _: Cpp.Derived = Cpp.Derived.Derived();
}

// ============================================================================
// Multiple arguments
// ============================================================================

// --- multi_argument.h

class MultiArgC {
 public:
  MultiArgC() {}
  MultiArgC(int) {}
  MultiArgC(int, int) {}
  MultiArgC(int, int, int, int = 0) {}
};

// --- call_multi_argument.carbon

library "[[@TEST_NAME]]";

import Cpp library "multi_argument.h";

fn Zero() {
  let _: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC();
  let _: Cpp.MultiArgC = ();
}

fn One() {
  let _: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC(1);
  let _: Cpp.MultiArgC = 2;
  let _: Cpp.MultiArgC = (3,);
}

fn Two() {
  let _: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC(1, 2);
  let _: Cpp.MultiArgC = (3, 4);
}

fn Three() {
  let _: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC(1, 2, 3);
  let _: Cpp.MultiArgC = (4, 5, 6);
}

fn Four() {
  let _: Cpp.MultiArgC = Cpp.MultiArgC.MultiArgC(1, 2, 3, 4);
  let _: Cpp.MultiArgC = (5, 6, 7, 8);
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'import_default.carbon'
// CHECK:STDOUT: source_filename = "import_default.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.C = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !28 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_26.1.temp = alloca [8 x i8], align 4, !dbg !29
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_26.1.temp), !dbg !29
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %.loc7_26.1.temp), !dbg !29
// CHECK:STDOUT:   ret void, !dbg !32
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !42
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !42
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !33 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_19.2.temp = alloca [8 x i8], align 4, !dbg !34
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_19.2.temp), !dbg !34
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr %.loc11_19.2.temp), !dbg !34
// CHECK:STDOUT:   ret void, !dbg !37
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !42
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !42
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !42
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4, !tbaa !45
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4, !tbaa !46
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!21}
// CHECK:STDOUT: !llvm.module.flags = !{!47, !48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!52}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !20 = !DIFile(filename: "import_default.carbon", directory: "")
// CHECK:STDOUT: !21 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !20, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !28 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !20, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !29 = !DILocation(line: 7, column: 18, scope: !28)
// CHECK:STDOUT: !32 = !DILocation(line: 6, column: 1, scope: !28)
// CHECK:STDOUT: !33 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !20, line: 10, type: !24, spFlags: DISPFlagDefinition, unit: !21)
// CHECK:STDOUT: !34 = !DILocation(line: 11, column: 18, scope: !33)
// CHECK:STDOUT: !37 = !DILocation(line: 10, column: 1, scope: !33)
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !40 = !{!"any pointer", !39, i64 0}
// CHECK:STDOUT: !41 = !{!"p1 _ZTS1C", !40, i64 0}
// CHECK:STDOUT: !42 = !{!41, !41, i64 0}
// CHECK:STDOUT: !43 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !44 = !{!"_ZTS1C", !43, i64 0, !43, i64 4}
// CHECK:STDOUT: !45 = !{!44, !43, i64 0}
// CHECK:STDOUT: !46 = !{!44, !43, i64 4}
// CHECK:STDOUT: !47 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !52 = !{!51, !43, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_default.carbon'
// CHECK:STDOUT: source_filename = "implicit_default.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.C.20 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !57 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [8 x i8], align 4, !dbg !58
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !58
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr %_.var), !dbg !58
// CHECK:STDOUT:   ret void, !dbg !60
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !42
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !42
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr sret([8 x i8]) %return) #2 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %return), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !63
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr sret([8 x i8]) %return) #0 !dbg !64 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr %return), !dbg !67
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !42
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C.20, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4, !tbaa !45
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C.20, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4, !tbaa !46
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!54}
// CHECK:STDOUT: !llvm.module.flags = !{!47, !48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!52}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !40 = !{!"any pointer", !39, i64 0}
// CHECK:STDOUT: !41 = !{!"p1 _ZTS1C", !40, i64 0}
// CHECK:STDOUT: !42 = !{!41, !41, i64 0}
// CHECK:STDOUT: !43 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !44 = !{!"_ZTS1C", !43, i64 0, !43, i64 4}
// CHECK:STDOUT: !45 = !{!44, !43, i64 0}
// CHECK:STDOUT: !46 = !{!44, !43, i64 4}
// CHECK:STDOUT: !47 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !52 = !{!51, !43, i64 0}
// CHECK:STDOUT: !53 = !DIFile(filename: "implicit_default.carbon", directory: "")
// CHECK:STDOUT: !54 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !53, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !55 = !{!25}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !57 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !53, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !58 = !DILocation(line: 7, column: 3, scope: !57)
// CHECK:STDOUT: !60 = !DILocation(line: 6, column: 1, scope: !57)
// CHECK:STDOUT: !61 = !DIFile(filename: "./default.h", directory: "")
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp", scope: null, file: !61, line: 4, type: !56, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !63 = !DILocation(line: 4, column: 3, scope: !62)
// CHECK:STDOUT: !64 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c", scope: null, file: !4, line: 9, type: !56, spFlags: DISPFlagDefinition, unit: !54)
// CHECK:STDOUT: !67 = !DILocation(line: 9, column: 28, scope: !64)
// CHECK:STDOUT: !68 = !DILocation(line: 9, column: 21, scope: !64)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'defaulted_default.carbon'
// CHECK:STDOUT: source_filename = "defaulted_default.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %struct.ImplicitlyDefaultedDefault = type { i32 }
// CHECK:STDOUT: %struct.ExplicitlyDefaultedDefault = type { i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN26ImplicitlyDefaultedDefaultC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN26ExplicitlyDefaultedDefaultC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !71 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_65.1.temp = alloca [4 x i8], align 4, !dbg !72
// CHECK:STDOUT:   %_.var.loc9 = alloca [4 x i8], align 4, !dbg !75
// CHECK:STDOUT:   %.loc12_65.1.temp = alloca [4 x i8], align 4, !dbg !77
// CHECK:STDOUT:   %_.var.loc13 = alloca [4 x i8], align 4, !dbg !80
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_65.1.temp), !dbg !72
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %.loc8_65.1.temp), !dbg !72
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc9), !dbg !75
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9"(ptr %_.var.loc9), !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_65.1.temp), !dbg !77
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %.loc12_65.1.temp), !dbg !77
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !80
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271"(ptr %_.var.loc13), !dbg !80
// CHECK:STDOUT:   ret void, !dbg !82
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !99
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !99
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %return), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !103
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !103
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !86 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %return), !dbg !87
// CHECK:STDOUT:   ret void, !dbg !87
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9"(ptr sret([4 x i8]) %return) #0 !dbg !88 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp"(ptr %return), !dbg !91
// CHECK:STDOUT:   ret void, !dbg !92
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271"(ptr sret([4 x i8]) %return) #0 !dbg !93 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp"(ptr %return), !dbg !96
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN26ImplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !99
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %struct.ImplicitlyDefaultedDefault, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %x_, align 4, !tbaa !101
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN26ExplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #5 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !103
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %struct.ExplicitlyDefaultedDefault, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %x_, align 4, !tbaa !105
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!70}
// CHECK:STDOUT: !llvm.module.flags = !{!47, !48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!52}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !40 = !{!"any pointer", !39, i64 0}
// CHECK:STDOUT: !43 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !47 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !52 = !{!51, !43, i64 0}
// CHECK:STDOUT: !55 = !{!25}
// CHECK:STDOUT: !56 = !DISubroutineType(types: !55)
// CHECK:STDOUT: !69 = !DIFile(filename: "defaulted_default.carbon", directory: "")
// CHECK:STDOUT: !70 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !69, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !71 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !69, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !70)
// CHECK:STDOUT: !72 = !DILocation(line: 8, column: 7, scope: !71)
// CHECK:STDOUT: !75 = !DILocation(line: 9, column: 3, scope: !71)
// CHECK:STDOUT: !77 = !DILocation(line: 12, column: 7, scope: !71)
// CHECK:STDOUT: !80 = !DILocation(line: 13, column: 3, scope: !71)
// CHECK:STDOUT: !82 = !DILocation(line: 6, column: 1, scope: !71)
// CHECK:STDOUT: !83 = !DIFile(filename: "./defaulted_default.h", directory: "")
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp", scope: null, file: !83, line: 2, type: !56, spFlags: DISPFlagDefinition, unit: !70)
// CHECK:STDOUT: !85 = !DILocation(line: 2, column: 8, scope: !84)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp", scope: null, file: !83, line: 7, type: !56, spFlags: DISPFlagDefinition, unit: !70)
// CHECK:STDOUT: !87 = !DILocation(line: 7, column: 3, scope: !86)
// CHECK:STDOUT: !88 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9", scope: null, file: !4, line: 9, type: !56, spFlags: DISPFlagDefinition, unit: !70)
// CHECK:STDOUT: !91 = !DILocation(line: 9, column: 28, scope: !88)
// CHECK:STDOUT: !92 = !DILocation(line: 9, column: 21, scope: !88)
// CHECK:STDOUT: !93 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271", scope: null, file: !4, line: 9, type: !56, spFlags: DISPFlagDefinition, unit: !70)
// CHECK:STDOUT: !96 = !DILocation(line: 9, column: 28, scope: !93)
// CHECK:STDOUT: !97 = !DILocation(line: 9, column: 21, scope: !93)
// CHECK:STDOUT: !98 = !{!"p1 _ZTS26ImplicitlyDefaultedDefault", !40, i64 0}
// CHECK:STDOUT: !99 = !{!98, !98, i64 0}
// CHECK:STDOUT: !100 = !{!"_ZTS26ImplicitlyDefaultedDefault", !43, i64 0}
// CHECK:STDOUT: !101 = !{!100, !43, i64 0}
// CHECK:STDOUT: !102 = !{!"p1 _ZTS26ExplicitlyDefaultedDefault", !40, i64 0}
// CHECK:STDOUT: !103 = !{!102, !102, i64 0}
// CHECK:STDOUT: !104 = !{!"_ZTS26ExplicitlyDefaultedDefault", !43, i64 0}
// CHECK:STDOUT: !105 = !{!104, !43, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_copy.carbon'
// CHECK:STDOUT: source_filename = "call_copy.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.Main(ptr sret([4 x i8]) %return, ptr %c) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.1.temp = alloca [4 x i8], align 4, !dbg !117
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_10.1.temp), !dbg !117
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr %c, ptr %return), !dbg !117
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !126
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !126
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !126
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !126
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %1, ptr noundef nonnull align 4 dereferenceable(4) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.58fe21e3ef78b0d7.Core:Copy.Cpp"(ptr sret([4 x i8]) %return, ptr %self) #2 !dbg !121 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !123
// CHECK:STDOUT:   ret void, !dbg !123
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_ZN4CopyC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4), ptr noundef nonnull align 4 dereferenceable(4)) unnamed_addr #4
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!107}
// CHECK:STDOUT: !llvm.module.flags = !{!47, !48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!52}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !25 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !40 = !{!"any pointer", !39, i64 0}
// CHECK:STDOUT: !43 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !47 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !52 = !{!51, !43, i64 0}
// CHECK:STDOUT: !106 = !DIFile(filename: "call_copy.carbon", directory: "")
// CHECK:STDOUT: !107 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !106, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !108 = !{!25, !25}
// CHECK:STDOUT: !109 = !DISubroutineType(types: !108)
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.Main", scope: null, file: !106, line: 6, type: !109, spFlags: DISPFlagDefinition, unit: !107, retainedNodes: !119)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !25)
// CHECK:STDOUT: !117 = !DILocation(line: 7, column: 10, scope: !112)
// CHECK:STDOUT: !118 = !DILocation(line: 7, column: 3, scope: !112)
// CHECK:STDOUT: !119 = !{!113}
// CHECK:STDOUT: !120 = !DIFile(filename: "./copy.h", directory: "")
// CHECK:STDOUT: !121 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.58fe21e3ef78b0d7.Core:Copy.Cpp", scope: null, file: !120, line: 5, type: !109, spFlags: DISPFlagDefinition, unit: !107, retainedNodes: !124)
// CHECK:STDOUT: !122 = !DILocalVariable(arg: 1, scope: !121, type: !25)
// CHECK:STDOUT: !123 = !DILocation(line: 5, column: 3, scope: !121)
// CHECK:STDOUT: !124 = !{!122}
// CHECK:STDOUT: !125 = !{!"p1 _ZTS4Copy", !40, i64 0}
// CHECK:STDOUT: !126 = !{!125, !125, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_c1_in_thunk.carbon'
// CHECK:STDOUT: source_filename = "call_c1_in_thunk.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN7DerivedC1Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTV7Derived = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTT7Derived = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI7Derived = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS7Derived = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTI4Base = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZTS4Base = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: @_ZTV7Derived = linkonce_odr dso_local constant { [3 x ptr] } { [3 x ptr] [ptr null, ptr null, ptr @_ZTI7Derived] }, comdat, align 8
// CHECK:STDOUT: @_ZTT7Derived = linkonce_odr dso_local unnamed_addr constant [1 x ptr] [ptr getelementptr inbounds inrange(-24, 0) (i8, ptr @_ZTV7Derived, i64 24)], comdat, align 8
// CHECK:STDOUT: @_ZTI7Derived = linkonce_odr dso_local constant { ptr, ptr, i32, i32, ptr, i64 } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 16), ptr @_ZTS7Derived, i32 0, i32 1, ptr @_ZTI4Base, i64 -6141 }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS7Derived = linkonce_odr dso_local constant [9 x i8] c"7Derived\00", comdat, align 1
// CHECK:STDOUT: @_ZTI4Base = linkonce_odr dso_local constant { ptr, ptr } { ptr getelementptr inbounds (i8, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 16), ptr @_ZTS4Base }, comdat, align 8
// CHECK:STDOUT: @_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
// CHECK:STDOUT: @_ZTS4Base = linkonce_odr dso_local constant [6 x i8] c"4Base\00", comdat, align 1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !130 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_44.1.temp = alloca [8 x i8], align 8, !dbg !131
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_44.1.temp), !dbg !131
// CHECK:STDOUT:   call void @_ZN7DerivedC1Ev.carbon_thunk.(ptr %.loc7_44.1.temp), !dbg !131
// CHECK:STDOUT:   ret void, !dbg !134
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN7DerivedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !136
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !136
// CHECK:STDOUT:   call void @_ZN7DerivedC1Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7DerivedC1Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !136
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-24, 0) (i8, ptr @_ZTV7Derived, i64 24), ptr %this1, align 8, !tbaa !138
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!129}
// CHECK:STDOUT: !llvm.module.flags = !{!47, !48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!52}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !40 = !{!"any pointer", !39, i64 0}
// CHECK:STDOUT: !43 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !47 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !52 = !{!51, !43, i64 0}
// CHECK:STDOUT: !128 = !DIFile(filename: "call_c1_in_thunk.carbon", directory: "")
// CHECK:STDOUT: !129 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !128, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !130 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !128, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !129)
// CHECK:STDOUT: !131 = !DILocation(line: 7, column: 24, scope: !130)
// CHECK:STDOUT: !134 = !DILocation(line: 6, column: 1, scope: !130)
// CHECK:STDOUT: !135 = !{!"p1 _ZTS7Derived", !40, i64 0}
// CHECK:STDOUT: !136 = !{!135, !135, i64 0}
// CHECK:STDOUT: !137 = !{!"vtable pointer", !38, i64 0}
// CHECK:STDOUT: !138 = !{!137, !137, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_multi_argument.carbon'
// CHECK:STDOUT: source_filename = "call_multi_argument.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN9MultiArgCC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN9MultiArgCC2Ei = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN9MultiArgCC2Eii = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN9MultiArgCC2Eiiii = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CZero.Main() #0 !dbg !149 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_50.1.temp = alloca [1 x i8], align 1, !dbg !150
// CHECK:STDOUT:   %.loc8_27.2.temp = alloca [1 x i8], align 1, !dbg !153
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_50.1.temp), !dbg !150
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ev.carbon_thunk.(ptr %.loc7_50.1.temp), !dbg !150
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_27.2.temp), !dbg !153
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ev.carbon_thunk_tuple.(ptr %.loc8_27.2.temp), !dbg !153
// CHECK:STDOUT:   ret void, !dbg !156
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COne.Main() #0 !dbg !157 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_51.1.temp = alloca [1 x i8], align 1, !dbg !158
// CHECK:STDOUT:   %.loc13_26.1.temp = alloca [1 x i8], align 1, !dbg !161
// CHECK:STDOUT:   %.loc14_29.2.temp = alloca [1 x i8], align 1, !dbg !164
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_51.1.temp), !dbg !158
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 1, ptr %.loc12_51.1.temp), !dbg !158
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_26.1.temp), !dbg !161
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 2, ptr %.loc13_26.1.temp), !dbg !161
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_29.2.temp), !dbg !164
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk_tuple._(i32 3, ptr %.loc14_29.2.temp), !dbg !164
// CHECK:STDOUT:   ret void, !dbg !167
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %2 = load i32, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %1, i32 noundef %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ei.carbon_thunk_tuple._(i32 noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %2 = load i32, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %1, i32 noundef %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTwo.Main() #0 !dbg !168 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_54.1.temp = alloca [1 x i8], align 1, !dbg !169
// CHECK:STDOUT:   %.loc19_31.2.temp = alloca [1 x i8], align 1, !dbg !172
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_54.1.temp), !dbg !169
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eii.carbon_thunk.__(i32 1, i32 2, ptr %.loc18_54.1.temp), !dbg !169
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_31.2.temp), !dbg !172
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eii.carbon_thunk_tuple.__(i32 3, i32 4, ptr %.loc19_31.2.temp), !dbg !172
// CHECK:STDOUT:   ret void, !dbg !175
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eii.carbon_thunk.__(i32 noundef %0, i32 noundef %1, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %3 = load i32, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %2, i32 noundef %3, i32 noundef %4)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eii.carbon_thunk_tuple.__(i32 noundef %0, i32 noundef %1, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %3 = load i32, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %2, i32 noundef %3, i32 noundef %4)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CThree.Main() #0 !dbg !176 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_57.1.temp = alloca [1 x i8], align 1, !dbg !177
// CHECK:STDOUT:   %.loc24_34.2.temp = alloca [1 x i8], align 1, !dbg !180
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc23_57.1.temp), !dbg !177
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk.___(i32 1, i32 2, i32 3, ptr %.loc23_57.1.temp), !dbg !177
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_34.2.temp), !dbg !180
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.___(i32 4, i32 5, i32 6, ptr %.loc24_34.2.temp), !dbg !180
// CHECK:STDOUT:   ret void, !dbg !183
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eiiii.carbon_thunk.___(i32 noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !194
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %3 = load ptr, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %4 = load i32, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   %5 = load i32, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   %6 = load i32, ptr %.addr2, align 4, !tbaa !194
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.___(i32 noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !194
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %3 = load ptr, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %4 = load i32, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   %5 = load i32, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   %6 = load i32, ptr %.addr2, align 4, !tbaa !194
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFour.Main() #0 !dbg !184 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_60.1.temp = alloca [1 x i8], align 1, !dbg !185
// CHECK:STDOUT:   %.loc29_37.2.temp = alloca [1 x i8], align 1, !dbg !188
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28_60.1.temp), !dbg !185
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk.____(i32 1, i32 2, i32 3, i32 4, ptr %.loc28_60.1.temp), !dbg !185
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_37.2.temp), !dbg !188
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.____(i32 5, i32 6, i32 7, i32 8, ptr %.loc29_37.2.temp), !dbg !188
// CHECK:STDOUT:   ret void, !dbg !191
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eiiii.carbon_thunk.____(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !194
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %4 = load ptr, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %5 = load i32, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   %6 = load i32, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   %7 = load i32, ptr %.addr2, align 4, !tbaa !194
// CHECK:STDOUT:   %8 = load i32, ptr %.addr3, align 4, !tbaa !194
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %4, i32 noundef %5, i32 noundef %6, i32 noundef %7, i32 noundef %8)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.____(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !194
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %4 = load ptr, ptr %return.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %5 = load i32, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   %6 = load i32, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   %7 = load i32, ptr %.addr2, align 4, !tbaa !194
// CHECK:STDOUT:   %8 = load i32, ptr %.addr3, align 4, !tbaa !194
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %4, i32 noundef %5, i32 noundef %6, i32 noundef %7, i32 noundef %8)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !193
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !193
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0, i32 noundef %1) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !193
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   %this2 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %.addr3 = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !193
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !194
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !194
// CHECK:STDOUT:   %this4 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN9MultiArgCC1Ei.carbon_thunk._, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!140}
// CHECK:STDOUT: !llvm.module.flags = !{!47, !48, !49, !50, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!52}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !{null}
// CHECK:STDOUT: !24 = !DISubroutineType(types: !23)
// CHECK:STDOUT: !38 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !39 = !{!"omnipotent char", !38, i64 0}
// CHECK:STDOUT: !40 = !{!"any pointer", !39, i64 0}
// CHECK:STDOUT: !43 = !{!"int", !39, i64 0}
// CHECK:STDOUT: !47 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !48 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !49 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !50 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !51 = !{!"__libc_errno", !43, i64 0}
// CHECK:STDOUT: !52 = !{!51, !43, i64 0}
// CHECK:STDOUT: !139 = !DIFile(filename: "call_multi_argument.carbon", directory: "")
// CHECK:STDOUT: !140 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !139, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !149 = distinct !DISubprogram(name: "Zero", linkageName: "_CZero.Main", scope: null, file: !139, line: 6, type: !24, spFlags: DISPFlagDefinition, unit: !140)
// CHECK:STDOUT: !150 = !DILocation(line: 7, column: 26, scope: !149)
// CHECK:STDOUT: !153 = !DILocation(line: 8, column: 26, scope: !149)
// CHECK:STDOUT: !156 = !DILocation(line: 6, column: 1, scope: !149)
// CHECK:STDOUT: !157 = distinct !DISubprogram(name: "One", linkageName: "_COne.Main", scope: null, file: !139, line: 11, type: !24, spFlags: DISPFlagDefinition, unit: !140)
// CHECK:STDOUT: !158 = !DILocation(line: 12, column: 26, scope: !157)
// CHECK:STDOUT: !161 = !DILocation(line: 13, column: 26, scope: !157)
// CHECK:STDOUT: !164 = !DILocation(line: 14, column: 26, scope: !157)
// CHECK:STDOUT: !167 = !DILocation(line: 11, column: 1, scope: !157)
// CHECK:STDOUT: !168 = distinct !DISubprogram(name: "Two", linkageName: "_CTwo.Main", scope: null, file: !139, line: 17, type: !24, spFlags: DISPFlagDefinition, unit: !140)
// CHECK:STDOUT: !169 = !DILocation(line: 18, column: 26, scope: !168)
// CHECK:STDOUT: !172 = !DILocation(line: 19, column: 26, scope: !168)
// CHECK:STDOUT: !175 = !DILocation(line: 17, column: 1, scope: !168)
// CHECK:STDOUT: !176 = distinct !DISubprogram(name: "Three", linkageName: "_CThree.Main", scope: null, file: !139, line: 22, type: !24, spFlags: DISPFlagDefinition, unit: !140)
// CHECK:STDOUT: !177 = !DILocation(line: 23, column: 26, scope: !176)
// CHECK:STDOUT: !180 = !DILocation(line: 24, column: 26, scope: !176)
// CHECK:STDOUT: !183 = !DILocation(line: 22, column: 1, scope: !176)
// CHECK:STDOUT: !184 = distinct !DISubprogram(name: "Four", linkageName: "_CFour.Main", scope: null, file: !139, line: 27, type: !24, spFlags: DISPFlagDefinition, unit: !140)
// CHECK:STDOUT: !185 = !DILocation(line: 28, column: 26, scope: !184)
// CHECK:STDOUT: !188 = !DILocation(line: 29, column: 26, scope: !184)
// CHECK:STDOUT: !191 = !DILocation(line: 27, column: 1, scope: !184)
// CHECK:STDOUT: !192 = !{!"p1 _ZTS9MultiArgC", !40, i64 0}
// CHECK:STDOUT: !193 = !{!192, !192, i64 0}
// CHECK:STDOUT: !194 = !{!43, !43, i64 0}
// CHECK:STDOUT:
