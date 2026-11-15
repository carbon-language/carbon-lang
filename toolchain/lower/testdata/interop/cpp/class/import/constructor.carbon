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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !26 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_26.1.temp = alloca [8 x i8], align 4, !dbg !27
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_26.1.temp), !dbg !27
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %.loc7_26.1.temp), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !40
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !40
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CG.Main() #0 !dbg !31 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_19.2.temp = alloca [8 x i8], align 4, !dbg !32
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc11_19.2.temp), !dbg !32
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr %.loc11_19.2.temp), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !40
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !40
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !40
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4, !tbaa !43
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4, !tbaa !44
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "import_default.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !27 = !DILocation(line: 7, column: 18, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 6, column: 1, scope: !26)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !18, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !32 = !DILocation(line: 11, column: 18, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 10, column: 1, scope: !31)
// CHECK:STDOUT: !36 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !37 = !{!"omnipotent char", !36, i64 0}
// CHECK:STDOUT: !38 = !{!"any pointer", !37, i64 0}
// CHECK:STDOUT: !39 = !{!"p1 _ZTS1C", !38, i64 0}
// CHECK:STDOUT: !40 = !{!39, !39, i64 0}
// CHECK:STDOUT: !41 = !{!"int", !37, i64 0}
// CHECK:STDOUT: !42 = !{!"_ZTS1C", !41, i64 0, !41, i64 4}
// CHECK:STDOUT: !43 = !{!42, !41, i64 0}
// CHECK:STDOUT: !44 = !{!42, !41, i64 4}
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !41, i64 0}
// CHECK:STDOUT: !50 = !{!49, !41, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'implicit_default.carbon'
// CHECK:STDOUT: source_filename = "implicit_default.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: %class.C.18 = type { i32, i32 }
// CHECK:STDOUT:
// CHECK:STDOUT: $_ZN1CC2Ev = comdat any
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !55 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [8 x i8], align 4, !dbg !56
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !56
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr %_.var), !dbg !56
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !40
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !40
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr sret([8 x i8]) %return) #2 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %return), !dbg !61
// CHECK:STDOUT:   ret void, !dbg !61
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr sret([8 x i8]) %return) #0 !dbg !62 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr %return), !dbg !65
// CHECK:STDOUT:   ret void, !dbg !66
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !40
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C.18, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4, !tbaa !43
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C.18, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4, !tbaa !44
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!52}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !37 = !{!"omnipotent char", !36, i64 0}
// CHECK:STDOUT: !38 = !{!"any pointer", !37, i64 0}
// CHECK:STDOUT: !39 = !{!"p1 _ZTS1C", !38, i64 0}
// CHECK:STDOUT: !40 = !{!39, !39, i64 0}
// CHECK:STDOUT: !41 = !{!"int", !37, i64 0}
// CHECK:STDOUT: !42 = !{!"_ZTS1C", !41, i64 0, !41, i64 4}
// CHECK:STDOUT: !43 = !{!42, !41, i64 0}
// CHECK:STDOUT: !44 = !{!42, !41, i64 4}
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !41, i64 0}
// CHECK:STDOUT: !50 = !{!49, !41, i64 0}
// CHECK:STDOUT: !51 = !DIFile(filename: "implicit_default.carbon", directory: "")
// CHECK:STDOUT: !52 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !51, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !53 = !{!23}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !55 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !51, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !52)
// CHECK:STDOUT: !56 = !DILocation(line: 7, column: 3, scope: !55)
// CHECK:STDOUT: !58 = !DILocation(line: 6, column: 1, scope: !55)
// CHECK:STDOUT: !59 = !DIFile(filename: "./default.h", directory: "")
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp", scope: null, file: !59, line: 4, type: !54, spFlags: DISPFlagDefinition, unit: !52)
// CHECK:STDOUT: !61 = !DILocation(line: 4, column: 3, scope: !60)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c", scope: null, file: !4, line: 9, type: !54, spFlags: DISPFlagDefinition, unit: !52)
// CHECK:STDOUT: !65 = !DILocation(line: 9, column: 28, scope: !62)
// CHECK:STDOUT: !66 = !DILocation(line: 9, column: 21, scope: !62)
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !69 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_65.1.temp = alloca [4 x i8], align 4, !dbg !70
// CHECK:STDOUT:   %_.var.loc9 = alloca [4 x i8], align 4, !dbg !73
// CHECK:STDOUT:   %.loc12_65.1.temp = alloca [4 x i8], align 4, !dbg !75
// CHECK:STDOUT:   %_.var.loc13 = alloca [4 x i8], align 4, !dbg !78
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_65.1.temp), !dbg !70
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %.loc8_65.1.temp), !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc9), !dbg !73
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9"(ptr %_.var.loc9), !dbg !73
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_65.1.temp), !dbg !75
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %.loc12_65.1.temp), !dbg !75
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !78
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271"(ptr %_.var.loc13), !dbg !78
// CHECK:STDOUT:   ret void, !dbg !80
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !97
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !97
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !82 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %return), !dbg !83
// CHECK:STDOUT:   ret void, !dbg !83
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !101
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !101
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !84 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %return), !dbg !85
// CHECK:STDOUT:   ret void, !dbg !85
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9"(ptr sret([4 x i8]) %return) #0 !dbg !86 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp"(ptr %return), !dbg !89
// CHECK:STDOUT:   ret void, !dbg !90
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271"(ptr sret([4 x i8]) %return) #0 !dbg !91 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp"(ptr %return), !dbg !94
// CHECK:STDOUT:   ret void, !dbg !95
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN26ImplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !97
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %struct.ImplicitlyDefaultedDefault, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %x_, align 4, !tbaa !99
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN26ExplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #5 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !101
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %struct.ExplicitlyDefaultedDefault, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %x_, align 4, !tbaa !103
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!68}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !37 = !{!"omnipotent char", !36, i64 0}
// CHECK:STDOUT: !38 = !{!"any pointer", !37, i64 0}
// CHECK:STDOUT: !41 = !{!"int", !37, i64 0}
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !41, i64 0}
// CHECK:STDOUT: !50 = !{!49, !41, i64 0}
// CHECK:STDOUT: !53 = !{!23}
// CHECK:STDOUT: !54 = !DISubroutineType(types: !53)
// CHECK:STDOUT: !67 = !DIFile(filename: "defaulted_default.carbon", directory: "")
// CHECK:STDOUT: !68 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !67, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !67, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !68)
// CHECK:STDOUT: !70 = !DILocation(line: 8, column: 7, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 9, column: 3, scope: !69)
// CHECK:STDOUT: !75 = !DILocation(line: 12, column: 7, scope: !69)
// CHECK:STDOUT: !78 = !DILocation(line: 13, column: 3, scope: !69)
// CHECK:STDOUT: !80 = !DILocation(line: 6, column: 1, scope: !69)
// CHECK:STDOUT: !81 = !DIFile(filename: "./defaulted_default.h", directory: "")
// CHECK:STDOUT: !82 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp", scope: null, file: !81, line: 2, type: !54, spFlags: DISPFlagDefinition, unit: !68)
// CHECK:STDOUT: !83 = !DILocation(line: 2, column: 8, scope: !82)
// CHECK:STDOUT: !84 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp", scope: null, file: !81, line: 7, type: !54, spFlags: DISPFlagDefinition, unit: !68)
// CHECK:STDOUT: !85 = !DILocation(line: 7, column: 3, scope: !84)
// CHECK:STDOUT: !86 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9", scope: null, file: !4, line: 9, type: !54, spFlags: DISPFlagDefinition, unit: !68)
// CHECK:STDOUT: !89 = !DILocation(line: 9, column: 28, scope: !86)
// CHECK:STDOUT: !90 = !DILocation(line: 9, column: 21, scope: !86)
// CHECK:STDOUT: !91 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271", scope: null, file: !4, line: 9, type: !54, spFlags: DISPFlagDefinition, unit: !68)
// CHECK:STDOUT: !94 = !DILocation(line: 9, column: 28, scope: !91)
// CHECK:STDOUT: !95 = !DILocation(line: 9, column: 21, scope: !91)
// CHECK:STDOUT: !96 = !{!"p1 _ZTS26ImplicitlyDefaultedDefault", !38, i64 0}
// CHECK:STDOUT: !97 = !{!96, !96, i64 0}
// CHECK:STDOUT: !98 = !{!"_ZTS26ImplicitlyDefaultedDefault", !41, i64 0}
// CHECK:STDOUT: !99 = !{!98, !41, i64 0}
// CHECK:STDOUT: !100 = !{!"p1 _ZTS26ExplicitlyDefaultedDefault", !38, i64 0}
// CHECK:STDOUT: !101 = !{!100, !100, i64 0}
// CHECK:STDOUT: !102 = !{!"_ZTS26ExplicitlyDefaultedDefault", !41, i64 0}
// CHECK:STDOUT: !103 = !{!102, !41, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_copy.carbon'
// CHECK:STDOUT: source_filename = "call_copy.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.Main(ptr sret([4 x i8]) %return, ptr %c) #0 !dbg !110 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.1.temp = alloca [4 x i8], align 4, !dbg !115
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_10.1.temp), !dbg !115
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr %c, ptr %return), !dbg !115
// CHECK:STDOUT:   ret void, !dbg !116
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !124
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !124
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !124
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %1, ptr noundef nonnull align 4 dereferenceable(4) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.58fe21e3ef78b0d7.Core:Copy.Cpp"(ptr sret([4 x i8]) %return, ptr %self) #2 !dbg !119 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !121
// CHECK:STDOUT:   ret void, !dbg !121
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!105}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !36 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !37 = !{!"omnipotent char", !36, i64 0}
// CHECK:STDOUT: !38 = !{!"any pointer", !37, i64 0}
// CHECK:STDOUT: !41 = !{!"int", !37, i64 0}
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !41, i64 0}
// CHECK:STDOUT: !50 = !{!49, !41, i64 0}
// CHECK:STDOUT: !104 = !DIFile(filename: "call_copy.carbon", directory: "")
// CHECK:STDOUT: !105 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !104, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !106 = !{!23, !23}
// CHECK:STDOUT: !107 = !DISubroutineType(types: !106)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.Main", scope: null, file: !104, line: 6, type: !107, spFlags: DISPFlagDefinition, unit: !105, retainedNodes: !117)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !23)
// CHECK:STDOUT: !115 = !DILocation(line: 7, column: 10, scope: !110)
// CHECK:STDOUT: !116 = !DILocation(line: 7, column: 3, scope: !110)
// CHECK:STDOUT: !117 = !{!111}
// CHECK:STDOUT: !118 = !DIFile(filename: "./copy.h", directory: "")
// CHECK:STDOUT: !119 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.58fe21e3ef78b0d7.Core:Copy.Cpp", scope: null, file: !118, line: 5, type: !107, spFlags: DISPFlagDefinition, unit: !105, retainedNodes: !122)
// CHECK:STDOUT: !120 = !DILocalVariable(arg: 1, scope: !119, type: !23)
// CHECK:STDOUT: !121 = !DILocation(line: 5, column: 3, scope: !119)
// CHECK:STDOUT: !122 = !{!120}
// CHECK:STDOUT: !123 = !{!"p1 _ZTS4Copy", !38, i64 0}
// CHECK:STDOUT: !124 = !{!123, !123, i64 0}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !128 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_44.1.temp = alloca [8 x i8], align 8, !dbg !129
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_44.1.temp), !dbg !129
// CHECK:STDOUT:   call void @_ZN7DerivedC1Ev.carbon_thunk.(ptr %.loc7_44.1.temp), !dbg !129
// CHECK:STDOUT:   ret void, !dbg !132
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN7DerivedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !134
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !134
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !134
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-24, 0) (i8, ptr @_ZTV7Derived, i64 24), ptr %this1, align 8, !tbaa !136
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!127}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !36 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !37 = !{!"omnipotent char", !36, i64 0}
// CHECK:STDOUT: !38 = !{!"any pointer", !37, i64 0}
// CHECK:STDOUT: !41 = !{!"int", !37, i64 0}
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !41, i64 0}
// CHECK:STDOUT: !50 = !{!49, !41, i64 0}
// CHECK:STDOUT: !126 = !DIFile(filename: "call_c1_in_thunk.carbon", directory: "")
// CHECK:STDOUT: !127 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !126, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !128 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !126, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !127)
// CHECK:STDOUT: !129 = !DILocation(line: 7, column: 24, scope: !128)
// CHECK:STDOUT: !132 = !DILocation(line: 6, column: 1, scope: !128)
// CHECK:STDOUT: !133 = !{!"p1 _ZTS7Derived", !38, i64 0}
// CHECK:STDOUT: !134 = !{!133, !133, i64 0}
// CHECK:STDOUT: !135 = !{!"vtable pointer", !36, i64 0}
// CHECK:STDOUT: !136 = !{!135, !135, i64 0}
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
// CHECK:STDOUT: define void @_CZero.Main() #0 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_50.1.temp = alloca [1 x i8], align 1, !dbg !148
// CHECK:STDOUT:   %.loc8_27.2.temp = alloca [1 x i8], align 1, !dbg !151
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_50.1.temp), !dbg !148
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ev.carbon_thunk.(ptr %.loc7_50.1.temp), !dbg !148
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_27.2.temp), !dbg !151
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ev.carbon_thunk_tuple.(ptr %.loc8_27.2.temp), !dbg !151
// CHECK:STDOUT:   ret void, !dbg !154
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COne.Main() #0 !dbg !155 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_51.1.temp = alloca [1 x i8], align 1, !dbg !156
// CHECK:STDOUT:   %.loc13_26.1.temp = alloca [1 x i8], align 1, !dbg !159
// CHECK:STDOUT:   %.loc14_29.2.temp = alloca [1 x i8], align 1, !dbg !162
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_51.1.temp), !dbg !156
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 1, ptr %.loc12_51.1.temp), !dbg !156
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_26.1.temp), !dbg !159
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 2, ptr %.loc13_26.1.temp), !dbg !159
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_29.2.temp), !dbg !162
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk_tuple._(i32 3, ptr %.loc14_29.2.temp), !dbg !162
// CHECK:STDOUT:   ret void, !dbg !165
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %2 = load i32, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %1, i32 noundef %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ei.carbon_thunk_tuple._(i32 noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %2 = load i32, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %1, i32 noundef %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTwo.Main() #0 !dbg !166 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_54.1.temp = alloca [1 x i8], align 1, !dbg !167
// CHECK:STDOUT:   %.loc19_31.2.temp = alloca [1 x i8], align 1, !dbg !170
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_54.1.temp), !dbg !167
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eii.carbon_thunk.__(i32 1, i32 2, ptr %.loc18_54.1.temp), !dbg !167
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_31.2.temp), !dbg !170
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eii.carbon_thunk_tuple.__(i32 3, i32 4, ptr %.loc19_31.2.temp), !dbg !170
// CHECK:STDOUT:   ret void, !dbg !173
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eii.carbon_thunk.__(i32 noundef %0, i32 noundef %1, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %3 = load i32, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !192
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
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %3 = load i32, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %2, i32 noundef %3, i32 noundef %4)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CThree.Main() #0 !dbg !174 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_57.1.temp = alloca [1 x i8], align 1, !dbg !175
// CHECK:STDOUT:   %.loc24_34.2.temp = alloca [1 x i8], align 1, !dbg !178
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc23_57.1.temp), !dbg !175
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk.___(i32 1, i32 2, i32 3, ptr %.loc23_57.1.temp), !dbg !175
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_34.2.temp), !dbg !178
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.___(i32 4, i32 5, i32 6, ptr %.loc24_34.2.temp), !dbg !178
// CHECK:STDOUT:   ret void, !dbg !181
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eiiii.carbon_thunk.___(i32 noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !192
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %3 = load ptr, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %4 = load i32, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   %5 = load i32, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   %6 = load i32, ptr %.addr2, align 4, !tbaa !192
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
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !192
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %3 = load ptr, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %4 = load i32, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   %5 = load i32, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   %6 = load i32, ptr %.addr2, align 4, !tbaa !192
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFour.Main() #0 !dbg !182 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_60.1.temp = alloca [1 x i8], align 1, !dbg !183
// CHECK:STDOUT:   %.loc29_37.2.temp = alloca [1 x i8], align 1, !dbg !186
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28_60.1.temp), !dbg !183
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk.____(i32 1, i32 2, i32 3, i32 4, ptr %.loc28_60.1.temp), !dbg !183
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_37.2.temp), !dbg !186
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.____(i32 5, i32 6, i32 7, i32 8, ptr %.loc29_37.2.temp), !dbg !186
// CHECK:STDOUT:   ret void, !dbg !189
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
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !192
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %4 = load ptr, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %5 = load i32, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   %6 = load i32, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   %7 = load i32, ptr %.addr2, align 4, !tbaa !192
// CHECK:STDOUT:   %8 = load i32, ptr %.addr3, align 4, !tbaa !192
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
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !192
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %4 = load ptr, ptr %return.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %5 = load i32, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   %6 = load i32, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   %7 = load i32, ptr %.addr2, align 4, !tbaa !192
// CHECK:STDOUT:   %8 = load i32, ptr %.addr3, align 4, !tbaa !192
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !191
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !191
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !191
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !192
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !191
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !192
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !192
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!138}
// CHECK:STDOUT: !llvm.module.flags = !{!45, !46, !47, !48, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!50}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !36 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !37 = !{!"omnipotent char", !36, i64 0}
// CHECK:STDOUT: !38 = !{!"any pointer", !37, i64 0}
// CHECK:STDOUT: !41 = !{!"int", !37, i64 0}
// CHECK:STDOUT: !45 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !46 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !47 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !48 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !49 = !{!"__libc_errno", !41, i64 0}
// CHECK:STDOUT: !50 = !{!49, !41, i64 0}
// CHECK:STDOUT: !137 = !DIFile(filename: "call_multi_argument.carbon", directory: "")
// CHECK:STDOUT: !138 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !137, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "Zero", linkageName: "_CZero.Main", scope: null, file: !137, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !138)
// CHECK:STDOUT: !148 = !DILocation(line: 7, column: 26, scope: !147)
// CHECK:STDOUT: !151 = !DILocation(line: 8, column: 26, scope: !147)
// CHECK:STDOUT: !154 = !DILocation(line: 6, column: 1, scope: !147)
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "One", linkageName: "_COne.Main", scope: null, file: !137, line: 11, type: !22, spFlags: DISPFlagDefinition, unit: !138)
// CHECK:STDOUT: !156 = !DILocation(line: 12, column: 26, scope: !155)
// CHECK:STDOUT: !159 = !DILocation(line: 13, column: 26, scope: !155)
// CHECK:STDOUT: !162 = !DILocation(line: 14, column: 26, scope: !155)
// CHECK:STDOUT: !165 = !DILocation(line: 11, column: 1, scope: !155)
// CHECK:STDOUT: !166 = distinct !DISubprogram(name: "Two", linkageName: "_CTwo.Main", scope: null, file: !137, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !138)
// CHECK:STDOUT: !167 = !DILocation(line: 18, column: 26, scope: !166)
// CHECK:STDOUT: !170 = !DILocation(line: 19, column: 26, scope: !166)
// CHECK:STDOUT: !173 = !DILocation(line: 17, column: 1, scope: !166)
// CHECK:STDOUT: !174 = distinct !DISubprogram(name: "Three", linkageName: "_CThree.Main", scope: null, file: !137, line: 22, type: !22, spFlags: DISPFlagDefinition, unit: !138)
// CHECK:STDOUT: !175 = !DILocation(line: 23, column: 26, scope: !174)
// CHECK:STDOUT: !178 = !DILocation(line: 24, column: 26, scope: !174)
// CHECK:STDOUT: !181 = !DILocation(line: 22, column: 1, scope: !174)
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "Four", linkageName: "_CFour.Main", scope: null, file: !137, line: 27, type: !22, spFlags: DISPFlagDefinition, unit: !138)
// CHECK:STDOUT: !183 = !DILocation(line: 28, column: 26, scope: !182)
// CHECK:STDOUT: !186 = !DILocation(line: 29, column: 26, scope: !182)
// CHECK:STDOUT: !189 = !DILocation(line: 27, column: 1, scope: !182)
// CHECK:STDOUT: !190 = !{!"p1 _ZTS9MultiArgC", !38, i64 0}
// CHECK:STDOUT: !191 = !{!190, !190, i64 0}
// CHECK:STDOUT: !192 = !{!41, !41, i64 0}
// CHECK:STDOUT:
