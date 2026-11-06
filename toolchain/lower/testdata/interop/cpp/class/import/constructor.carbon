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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4d2f117a5a2332f0(ptr %.loc7_26.1.temp), !dbg !27
// CHECK:STDOUT:   ret void, !dbg !30
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !47
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
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4d2f117a5a2332f0(ptr %.loc11_19.2.temp), !dbg !32
// CHECK:STDOUT:   ret void, !dbg !35
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4d2f117a5a2332f0(ptr %t) #0 !dbg !36 {
// CHECK:STDOUT:   ret void, !dbg !41
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4, !tbaa !50
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4, !tbaa !51
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.4d2f117a5a2332f0, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!52, !53, !54, !55, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "import_default.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !18, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !27 = !DILocation(line: 7, column: 18, scope: !26)
// CHECK:STDOUT: !30 = !DILocation(line: 6, column: 1, scope: !26)
// CHECK:STDOUT: !31 = distinct !DISubprogram(name: "G", linkageName: "_CG.Main", scope: null, file: !18, line: 10, type: !22, spFlags: DISPFlagDefinition, unit: !19)
// CHECK:STDOUT: !32 = !DILocation(line: 11, column: 18, scope: !31)
// CHECK:STDOUT: !35 = !DILocation(line: 10, column: 1, scope: !31)
// CHECK:STDOUT: !36 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4d2f117a5a2332f0", scope: null, file: !10, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !19, retainedNodes: !42)
// CHECK:STDOUT: !37 = !DILocalVariable(arg: 1, scope: !36, type: !23)
// CHECK:STDOUT: !41 = !DILocation(line: 32, column: 1, scope: !36)
// CHECK:STDOUT: !42 = !{!37}
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"any pointer", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"p1 _ZTS1C", !45, i64 0}
// CHECK:STDOUT: !47 = !{!46, !46, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !49 = !{!"_ZTS1C", !48, i64 0, !48, i64 4}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT: !51 = !{!49, !48, i64 4}
// CHECK:STDOUT: !52 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !53 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !55 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !56 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !57 = !{!56, !48, i64 0}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !62 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %_.var = alloca [8 x i8], align 4, !dbg !63
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var), !dbg !63
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr %_.var), !dbg !63
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4d2f117a5a2332f0(ptr %_.var), !dbg !63
// CHECK:STDOUT:   ret void, !dbg !65
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN1CC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !47
// CHECK:STDOUT:   call void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr sret([8 x i8]) %return) #2 !dbg !67 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN1CC1Ev.carbon_thunk.(ptr %return), !dbg !68
// CHECK:STDOUT:   ret void, !dbg !68
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c"(ptr sret([8 x i8]) %return) #0 !dbg !69 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp"(ptr %return), !dbg !72
// CHECK:STDOUT:   ret void, !dbg !73
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4d2f117a5a2332f0(ptr %t) #0 !dbg !74 {
// CHECK:STDOUT:   ret void, !dbg !79
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN1CC2Ev(ptr noundef nonnull align 4 dereferenceable(8) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !47
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %class.C.18, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 8, ptr %x_, align 4, !tbaa !50
// CHECK:STDOUT:   %y_ = getelementptr inbounds nuw %class.C.18, ptr %this1, i32 0, i32 1
// CHECK:STDOUT:   store i32 9, ptr %y_, align 4, !tbaa !51
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!59}
// CHECK:STDOUT: !llvm.module.flags = !{!52, !53, !54, !55, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"any pointer", !44, i64 0}
// CHECK:STDOUT: !46 = !{!"p1 _ZTS1C", !45, i64 0}
// CHECK:STDOUT: !47 = !{!46, !46, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !49 = !{!"_ZTS1C", !48, i64 0, !48, i64 4}
// CHECK:STDOUT: !50 = !{!49, !48, i64 0}
// CHECK:STDOUT: !51 = !{!49, !48, i64 4}
// CHECK:STDOUT: !52 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !53 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !55 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !56 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !57 = !{!56, !48, i64 0}
// CHECK:STDOUT: !58 = !DIFile(filename: "implicit_default.carbon", directory: "")
// CHECK:STDOUT: !59 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !58, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !60 = !{!23}
// CHECK:STDOUT: !61 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !62 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !58, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !59)
// CHECK:STDOUT: !63 = !DILocation(line: 7, column: 3, scope: !62)
// CHECK:STDOUT: !65 = !DILocation(line: 6, column: 1, scope: !62)
// CHECK:STDOUT: !66 = !DIFile(filename: "./default.h", directory: "")
// CHECK:STDOUT: !67 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.dc3dd9c349162a91.Core:C.Cpp", scope: null, file: !66, line: 4, type: !61, spFlags: DISPFlagDefinition, unit: !59)
// CHECK:STDOUT: !68 = !DILocation(line: 4, column: 3, scope: !67)
// CHECK:STDOUT: !69 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.67b28f83c7cfd60c", scope: null, file: !4, line: 9, type: !61, spFlags: DISPFlagDefinition, unit: !59)
// CHECK:STDOUT: !72 = !DILocation(line: 9, column: 28, scope: !69)
// CHECK:STDOUT: !73 = !DILocation(line: 9, column: 21, scope: !69)
// CHECK:STDOUT: !74 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4d2f117a5a2332f0", scope: null, file: !10, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !59, retainedNodes: !80)
// CHECK:STDOUT: !75 = !DILocalVariable(arg: 1, scope: !74, type: !23)
// CHECK:STDOUT: !79 = !DILocation(line: 32, column: 1, scope: !74)
// CHECK:STDOUT: !80 = !{!75}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !83 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc8_65.1.temp = alloca [4 x i8], align 4, !dbg !84
// CHECK:STDOUT:   %_.var.loc9 = alloca [4 x i8], align 4, !dbg !87
// CHECK:STDOUT:   %.loc12_65.1.temp = alloca [4 x i8], align 4, !dbg !89
// CHECK:STDOUT:   %_.var.loc13 = alloca [4 x i8], align 4, !dbg !92
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_65.1.temp), !dbg !84
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %.loc8_65.1.temp), !dbg !84
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc9), !dbg !87
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9"(ptr %_.var.loc9), !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_65.1.temp), !dbg !89
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %.loc12_65.1.temp), !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %_.var.loc13), !dbg !92
// CHECK:STDOUT:   call void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271"(ptr %_.var.loc13), !dbg !92
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d07982f4dd457e72(ptr %_.var.loc13), !dbg !92
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d07982f4dd457e72(ptr %.loc12_65.1.temp), !dbg !89
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3d110fbad655625b(ptr %_.var.loc9), !dbg !87
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.3d110fbad655625b(ptr %.loc8_65.1.temp), !dbg !84
// CHECK:STDOUT:   ret void, !dbg !94
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !125
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !125
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !96 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN26ImplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %return), !dbg !97
// CHECK:STDOUT:   ret void, !dbg !97
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !129
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !129
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp"(ptr sret([4 x i8]) %return) #2 !dbg !98 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN26ExplicitlyDefaultedDefaultC1Ev.carbon_thunk.(ptr %return), !dbg !99
// CHECK:STDOUT:   ret void, !dbg !99
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #3
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9"(ptr sret([4 x i8]) %return) #0 !dbg !100 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp"(ptr %return), !dbg !103
// CHECK:STDOUT:   ret void, !dbg !104
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @"_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271"(ptr sret([4 x i8]) %return) #0 !dbg !105 {
// CHECK:STDOUT:   call void @"_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp"(ptr %return), !dbg !108
// CHECK:STDOUT:   ret void, !dbg !109
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d07982f4dd457e72(ptr %t) #0 !dbg !110 {
// CHECK:STDOUT:   ret void, !dbg !115
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.3d110fbad655625b(ptr %t) #0 !dbg !117 {
// CHECK:STDOUT:   ret void, !dbg !122
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN26ImplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #4 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !125
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %struct.ImplicitlyDefaultedDefault, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %x_, align 4, !tbaa !127
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN26ExplicitlyDefaultedDefaultC2Ev(ptr noundef nonnull align 4 dereferenceable(4) %this) unnamed_addr #5 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !129
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   %x_ = getelementptr inbounds nuw %struct.ExplicitlyDefaultedDefault, ptr %this1, i32 0, i32 0
// CHECK:STDOUT:   store i32 1, ptr %x_, align 4, !tbaa !131
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.d07982f4dd457e72, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.3d110fbad655625b, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { alwaysinline nounwind }
// CHECK:STDOUT: attributes #3 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #4 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #5 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!82}
// CHECK:STDOUT: !llvm.module.flags = !{!52, !53, !54, !55, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !4 = !DIFile(filename: "min_prelude/parts/default.carbon", directory: "")
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"any pointer", !44, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !52 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !53 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !55 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !56 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !57 = !{!56, !48, i64 0}
// CHECK:STDOUT: !60 = !{!23}
// CHECK:STDOUT: !61 = !DISubroutineType(types: !60)
// CHECK:STDOUT: !81 = !DIFile(filename: "defaulted_default.carbon", directory: "")
// CHECK:STDOUT: !82 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !81, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !83 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !81, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !84 = !DILocation(line: 8, column: 7, scope: !83)
// CHECK:STDOUT: !87 = !DILocation(line: 9, column: 3, scope: !83)
// CHECK:STDOUT: !89 = !DILocation(line: 12, column: 7, scope: !83)
// CHECK:STDOUT: !92 = !DILocation(line: 13, column: 3, scope: !83)
// CHECK:STDOUT: !94 = !DILocation(line: 6, column: 1, scope: !83)
// CHECK:STDOUT: !95 = !DIFile(filename: "./defaulted_default.h", directory: "")
// CHECK:STDOUT: !96 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.48fb4c4d31b7d0bf.Core:ImplicitlyDefaultedDefault.Cpp", scope: null, file: !95, line: 2, type: !61, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !97 = !DILocation(line: 2, column: 8, scope: !96)
// CHECK:STDOUT: !98 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Default.f7ca6602362234ee.Core:ExplicitlyDefaultedDefault.Cpp", scope: null, file: !95, line: 7, type: !61, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !99 = !DILocation(line: 7, column: 3, scope: !98)
// CHECK:STDOUT: !100 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.c20bae04b665a8c9", scope: null, file: !4, line: 9, type: !61, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !103 = !DILocation(line: 9, column: 28, scope: !100)
// CHECK:STDOUT: !104 = !DILocation(line: 9, column: 21, scope: !100)
// CHECK:STDOUT: !105 = distinct !DISubprogram(name: "Op", linkageName: "_COp.51f3fb62cb986374:DefaultOrUnformed.Core.055ae7235dade271", scope: null, file: !4, line: 9, type: !61, spFlags: DISPFlagDefinition, unit: !82)
// CHECK:STDOUT: !108 = !DILocation(line: 9, column: 28, scope: !105)
// CHECK:STDOUT: !109 = !DILocation(line: 9, column: 21, scope: !105)
// CHECK:STDOUT: !110 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d07982f4dd457e72", scope: null, file: !10, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !116)
// CHECK:STDOUT: !111 = !DILocalVariable(arg: 1, scope: !110, type: !23)
// CHECK:STDOUT: !115 = !DILocation(line: 32, column: 1, scope: !110)
// CHECK:STDOUT: !116 = !{!111}
// CHECK:STDOUT: !117 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.3d110fbad655625b", scope: null, file: !10, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !82, retainedNodes: !123)
// CHECK:STDOUT: !118 = !DILocalVariable(arg: 1, scope: !117, type: !23)
// CHECK:STDOUT: !122 = !DILocation(line: 32, column: 1, scope: !117)
// CHECK:STDOUT: !123 = !{!118}
// CHECK:STDOUT: !124 = !{!"p1 _ZTS26ImplicitlyDefaultedDefault", !45, i64 0}
// CHECK:STDOUT: !125 = !{!124, !124, i64 0}
// CHECK:STDOUT: !126 = !{!"_ZTS26ImplicitlyDefaultedDefault", !48, i64 0}
// CHECK:STDOUT: !127 = !{!126, !48, i64 0}
// CHECK:STDOUT: !128 = !{!"p1 _ZTS26ExplicitlyDefaultedDefault", !45, i64 0}
// CHECK:STDOUT: !129 = !{!128, !128, i64 0}
// CHECK:STDOUT: !130 = !{!"_ZTS26ExplicitlyDefaultedDefault", !48, i64 0}
// CHECK:STDOUT: !131 = !{!130, !48, i64 0}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'call_copy.carbon'
// CHECK:STDOUT: source_filename = "call_copy.carbon"
// CHECK:STDOUT: target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
// CHECK:STDOUT: target triple = "x86_64-unknown-linux-gnu"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CCopy.Main(ptr sret([4 x i8]) %return, ptr %c) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_10.1.temp = alloca [4 x i8], align 4, !dbg !143
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_10.1.temp), !dbg !143
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr %c, ptr %return), !dbg !143
// CHECK:STDOUT:   ret void, !dbg !144
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca ptr, align 8
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %0, ptr %.addr, align 8, !tbaa !152
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !152
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !152
// CHECK:STDOUT:   %2 = load ptr, ptr %.addr, align 8, !tbaa !152
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_(ptr noundef nonnull align 4 dereferenceable(4) %1, ptr noundef nonnull align 4 dereferenceable(4) %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline nounwind
// CHECK:STDOUT: define weak_odr void @"_COp:thunk:Copy.58fe21e3ef78b0d7.Core:Copy.Cpp"(ptr sret([4 x i8]) %return, ptr %self) #2 !dbg !147 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_ZN4CopyC1ERKS_.carbon_thunk._(ptr %self, ptr %return), !dbg !149
// CHECK:STDOUT:   ret void, !dbg !149
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
// CHECK:STDOUT: !llvm.dbg.cu = !{!133}
// CHECK:STDOUT: !llvm.module.flags = !{!52, !53, !54, !55, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"any pointer", !44, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !52 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !53 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !55 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !56 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !57 = !{!56, !48, i64 0}
// CHECK:STDOUT: !132 = !DIFile(filename: "call_copy.carbon", directory: "")
// CHECK:STDOUT: !133 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !132, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !134 = !{!23, !23}
// CHECK:STDOUT: !135 = !DISubroutineType(types: !134)
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "Copy", linkageName: "_CCopy.Main", scope: null, file: !132, line: 6, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !145)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !23)
// CHECK:STDOUT: !143 = !DILocation(line: 7, column: 10, scope: !138)
// CHECK:STDOUT: !144 = !DILocation(line: 7, column: 3, scope: !138)
// CHECK:STDOUT: !145 = !{!139}
// CHECK:STDOUT: !146 = !DIFile(filename: "./copy.h", directory: "")
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "Op", linkageName: "_COp:thunk:Copy.58fe21e3ef78b0d7.Core:Copy.Cpp", scope: null, file: !146, line: 5, type: !135, spFlags: DISPFlagDefinition, unit: !133, retainedNodes: !150)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !23)
// CHECK:STDOUT: !149 = !DILocation(line: 5, column: 3, scope: !147)
// CHECK:STDOUT: !150 = !{!148}
// CHECK:STDOUT: !151 = !{!"p1 _ZTS4Copy", !45, i64 0}
// CHECK:STDOUT: !152 = !{!151, !151, i64 0}
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
// CHECK:STDOUT: define void @_CF.Main() #0 !dbg !156 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_44.1.temp = alloca [8 x i8], align 8, !dbg !157
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_44.1.temp), !dbg !157
// CHECK:STDOUT:   call void @_ZN7DerivedC1Ev.carbon_thunk.(ptr %.loc7_44.1.temp), !dbg !157
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8073a50a45cfdc9c(ptr %.loc7_44.1.temp), !dbg !157
// CHECK:STDOUT:   ret void, !dbg !160
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress nounwind uwtable
// CHECK:STDOUT: define internal void @_ZN7DerivedC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !169
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !169
// CHECK:STDOUT:   call void @_ZN7DerivedC1Ev(ptr noundef nonnull align 8 dereferenceable(8) %0) #0
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8073a50a45cfdc9c(ptr %t) #0 !dbg !161 {
// CHECK:STDOUT:   ret void, !dbg !166
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: inlinehint mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN7DerivedC1Ev(ptr noundef nonnull align 8 dereferenceable(8) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !169
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   store ptr getelementptr inbounds inrange(-24, 0) (i8, ptr @_ZTV7Derived, i64 24), ptr %this1, align 8, !tbaa !171
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { inlinehint mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!155}
// CHECK:STDOUT: !llvm.module.flags = !{!52, !53, !54, !55, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"any pointer", !44, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !52 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !53 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !55 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !56 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !57 = !{!56, !48, i64 0}
// CHECK:STDOUT: !154 = !DIFile(filename: "call_c1_in_thunk.carbon", directory: "")
// CHECK:STDOUT: !155 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !154, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !156 = distinct !DISubprogram(name: "F", linkageName: "_CF.Main", scope: null, file: !154, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !155)
// CHECK:STDOUT: !157 = !DILocation(line: 7, column: 24, scope: !156)
// CHECK:STDOUT: !160 = !DILocation(line: 6, column: 1, scope: !156)
// CHECK:STDOUT: !161 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8073a50a45cfdc9c", scope: null, file: !10, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !155, retainedNodes: !167)
// CHECK:STDOUT: !162 = !DILocalVariable(arg: 1, scope: !161, type: !23)
// CHECK:STDOUT: !166 = !DILocation(line: 32, column: 1, scope: !161)
// CHECK:STDOUT: !167 = !{!162}
// CHECK:STDOUT: !168 = !{!"p1 _ZTS7Derived", !45, i64 0}
// CHECK:STDOUT: !169 = !{!168, !168, i64 0}
// CHECK:STDOUT: !170 = !{!"vtable pointer", !43, i64 0}
// CHECK:STDOUT: !171 = !{!170, !170, i64 0}
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
// CHECK:STDOUT: define void @_CZero.Main() #0 !dbg !182 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc7_50.1.temp = alloca [1 x i8], align 1, !dbg !183
// CHECK:STDOUT:   %.loc8_27.2.temp = alloca [1 x i8], align 1, !dbg !186
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc7_50.1.temp), !dbg !183
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ev.carbon_thunk.(ptr %.loc7_50.1.temp), !dbg !183
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc8_27.2.temp), !dbg !186
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ev.carbon_thunk_tuple.(ptr %.loc8_27.2.temp), !dbg !186
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc8_27.2.temp), !dbg !186
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc7_50.1.temp), !dbg !183
// CHECK:STDOUT:   ret void, !dbg !189
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ev.carbon_thunk.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ev.carbon_thunk_tuple.(ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %0 = load ptr, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_COne.Main() #0 !dbg !190 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc12_51.1.temp = alloca [1 x i8], align 1, !dbg !191
// CHECK:STDOUT:   %.loc13_26.1.temp = alloca [1 x i8], align 1, !dbg !194
// CHECK:STDOUT:   %.loc14_29.2.temp = alloca [1 x i8], align 1, !dbg !197
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc12_51.1.temp), !dbg !191
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 1, ptr %.loc12_51.1.temp), !dbg !191
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc13_26.1.temp), !dbg !194
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 2, ptr %.loc13_26.1.temp), !dbg !194
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc14_29.2.temp), !dbg !197
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Ei.carbon_thunk_tuple._(i32 3, ptr %.loc14_29.2.temp), !dbg !197
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc14_29.2.temp), !dbg !197
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc13_26.1.temp), !dbg !194
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc12_51.1.temp), !dbg !191
// CHECK:STDOUT:   ret void, !dbg !200
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ei.carbon_thunk._(i32 noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %2 = load i32, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %1, i32 noundef %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Ei.carbon_thunk_tuple._(i32 noundef %0, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %1 = load ptr, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %2 = load i32, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %1, i32 noundef %2)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CTwo.Main() #0 !dbg !201 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc18_54.1.temp = alloca [1 x i8], align 1, !dbg !202
// CHECK:STDOUT:   %.loc19_31.2.temp = alloca [1 x i8], align 1, !dbg !205
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc18_54.1.temp), !dbg !202
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eii.carbon_thunk.__(i32 1, i32 2, ptr %.loc18_54.1.temp), !dbg !202
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc19_31.2.temp), !dbg !205
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eii.carbon_thunk_tuple.__(i32 3, i32 4, ptr %.loc19_31.2.temp), !dbg !205
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc19_31.2.temp), !dbg !205
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc18_54.1.temp), !dbg !202
// CHECK:STDOUT:   ret void, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eii.carbon_thunk.__(i32 noundef %0, i32 noundef %1, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %3 = load i32, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !234
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
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %2 = load ptr, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %3 = load i32, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   %4 = load i32, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eii(ptr noundef nonnull align 1 dereferenceable(1) %2, i32 noundef %3, i32 noundef %4)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CThree.Main() #0 !dbg !209 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_57.1.temp = alloca [1 x i8], align 1, !dbg !210
// CHECK:STDOUT:   %.loc24_34.2.temp = alloca [1 x i8], align 1, !dbg !213
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc23_57.1.temp), !dbg !210
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk.___(i32 1, i32 2, i32 3, ptr %.loc23_57.1.temp), !dbg !210
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc24_34.2.temp), !dbg !213
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.___(i32 4, i32 5, i32 6, ptr %.loc24_34.2.temp), !dbg !213
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc24_34.2.temp), !dbg !213
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc23_57.1.temp), !dbg !210
// CHECK:STDOUT:   ret void, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: alwaysinline mustprogress uwtable
// CHECK:STDOUT: define internal void @_ZN9MultiArgCC1Eiiii.carbon_thunk.___(i32 noundef %0, i32 noundef %1, i32 noundef %2, ptr noundef %return) #1 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   %.addr1 = alloca i32, align 4
// CHECK:STDOUT:   %.addr2 = alloca i32, align 4
// CHECK:STDOUT:   %return.addr = alloca ptr, align 8
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !234
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %3 = load ptr, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %4 = load i32, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   %5 = load i32, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   %6 = load i32, ptr %.addr2, align 4, !tbaa !234
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
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !234
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %3 = load ptr, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %4 = load i32, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   %5 = load i32, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   %6 = load i32, ptr %.addr2, align 4, !tbaa !234
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %3, i32 noundef %4, i32 noundef %5, i32 noundef %6, i32 noundef 0)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CFour.Main() #0 !dbg !217 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc28_60.1.temp = alloca [1 x i8], align 1, !dbg !218
// CHECK:STDOUT:   %.loc29_37.2.temp = alloca [1 x i8], align 1, !dbg !221
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc28_60.1.temp), !dbg !218
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk.____(i32 1, i32 2, i32 3, i32 4, ptr %.loc28_60.1.temp), !dbg !218
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %.loc29_37.2.temp), !dbg !221
// CHECK:STDOUT:   call void @_ZN9MultiArgCC1Eiiii.carbon_thunk_tuple.____(i32 5, i32 6, i32 7, i32 8, ptr %.loc29_37.2.temp), !dbg !221
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc29_37.2.temp), !dbg !221
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %.loc28_60.1.temp), !dbg !218
// CHECK:STDOUT:   ret void, !dbg !224
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
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !234
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %4 = load ptr, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %5 = load i32, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   %6 = load i32, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   %7 = load i32, ptr %.addr2, align 4, !tbaa !234
// CHECK:STDOUT:   %8 = load i32, ptr %.addr3, align 4, !tbaa !234
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
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !234
// CHECK:STDOUT:   store ptr %return, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %4 = load ptr, ptr %return.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %5 = load i32, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   %6 = load i32, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   %7 = load i32, ptr %.addr2, align 4, !tbaa !234
// CHECK:STDOUT:   %8 = load i32, ptr %.addr3, align 4, !tbaa !234
// CHECK:STDOUT:   call void @_ZN9MultiArgCC2Eiiii(ptr noundef nonnull align 1 dereferenceable(1) %4, i32 noundef %5, i32 noundef %6, i32 noundef %7, i32 noundef %8)
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #2
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.b81987abcfec64c8(ptr %t) #0 !dbg !225 {
// CHECK:STDOUT:   ret void, !dbg !230
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Ev(ptr noundef nonnull align 1 dereferenceable(1) %this) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !233
// CHECK:STDOUT:   %this1 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: mustprogress nounwind uwtable
// CHECK:STDOUT: define linkonce_odr dso_local void @_ZN9MultiArgCC2Ei(ptr noundef nonnull align 1 dereferenceable(1) %this, i32 noundef %0) unnamed_addr #3 comdat align 2 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %this.addr = alloca ptr, align 8
// CHECK:STDOUT:   %.addr = alloca i32, align 4
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !233
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !233
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !234
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
// CHECK:STDOUT:   store ptr %this, ptr %this.addr, align 8, !tbaa !233
// CHECK:STDOUT:   store i32 %0, ptr %.addr, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %1, ptr %.addr1, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %2, ptr %.addr2, align 4, !tbaa !234
// CHECK:STDOUT:   store i32 %3, ptr %.addr3, align 4, !tbaa !234
// CHECK:STDOUT:   %this4 = load ptr, ptr %this.addr, align 8
// CHECK:STDOUT:   ret void
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @_ZN9MultiArgCC1Ei.carbon_thunk._, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.b81987abcfec64c8, { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { alwaysinline mustprogress uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT: attributes #2 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT: attributes #3 = { mustprogress nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!173}
// CHECK:STDOUT: !llvm.module.flags = !{!52, !53, !54, !55, !2, !3}
// CHECK:STDOUT: !llvm.errno.tbaa = !{!57}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !10 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !21 = !{null}
// CHECK:STDOUT: !22 = !DISubroutineType(types: !21)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{null, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !43 = !{!"Simple C++ TBAA"}
// CHECK:STDOUT: !44 = !{!"omnipotent char", !43, i64 0}
// CHECK:STDOUT: !45 = !{!"any pointer", !44, i64 0}
// CHECK:STDOUT: !48 = !{!"int", !44, i64 0}
// CHECK:STDOUT: !52 = !{i32 2, !"debugger-tuning", !"gdb"}
// CHECK:STDOUT: !53 = !{i32 8, !"PIC Level", i32 2}
// CHECK:STDOUT: !54 = !{i32 7, !"PIE Level", i32 2}
// CHECK:STDOUT: !55 = !{i32 7, !"uwtable", i32 2}
// CHECK:STDOUT: !56 = !{!"__libc_errno", !48, i64 0}
// CHECK:STDOUT: !57 = !{!56, !48, i64 0}
// CHECK:STDOUT: !172 = !DIFile(filename: "call_multi_argument.carbon", directory: "")
// CHECK:STDOUT: !173 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !172, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "Zero", linkageName: "_CZero.Main", scope: null, file: !172, line: 6, type: !22, spFlags: DISPFlagDefinition, unit: !173)
// CHECK:STDOUT: !183 = !DILocation(line: 7, column: 26, scope: !182)
// CHECK:STDOUT: !186 = !DILocation(line: 8, column: 26, scope: !182)
// CHECK:STDOUT: !189 = !DILocation(line: 6, column: 1, scope: !182)
// CHECK:STDOUT: !190 = distinct !DISubprogram(name: "One", linkageName: "_COne.Main", scope: null, file: !172, line: 11, type: !22, spFlags: DISPFlagDefinition, unit: !173)
// CHECK:STDOUT: !191 = !DILocation(line: 12, column: 26, scope: !190)
// CHECK:STDOUT: !194 = !DILocation(line: 13, column: 26, scope: !190)
// CHECK:STDOUT: !197 = !DILocation(line: 14, column: 26, scope: !190)
// CHECK:STDOUT: !200 = !DILocation(line: 11, column: 1, scope: !190)
// CHECK:STDOUT: !201 = distinct !DISubprogram(name: "Two", linkageName: "_CTwo.Main", scope: null, file: !172, line: 17, type: !22, spFlags: DISPFlagDefinition, unit: !173)
// CHECK:STDOUT: !202 = !DILocation(line: 18, column: 26, scope: !201)
// CHECK:STDOUT: !205 = !DILocation(line: 19, column: 26, scope: !201)
// CHECK:STDOUT: !208 = !DILocation(line: 17, column: 1, scope: !201)
// CHECK:STDOUT: !209 = distinct !DISubprogram(name: "Three", linkageName: "_CThree.Main", scope: null, file: !172, line: 22, type: !22, spFlags: DISPFlagDefinition, unit: !173)
// CHECK:STDOUT: !210 = !DILocation(line: 23, column: 26, scope: !209)
// CHECK:STDOUT: !213 = !DILocation(line: 24, column: 26, scope: !209)
// CHECK:STDOUT: !216 = !DILocation(line: 22, column: 1, scope: !209)
// CHECK:STDOUT: !217 = distinct !DISubprogram(name: "Four", linkageName: "_CFour.Main", scope: null, file: !172, line: 27, type: !22, spFlags: DISPFlagDefinition, unit: !173)
// CHECK:STDOUT: !218 = !DILocation(line: 28, column: 26, scope: !217)
// CHECK:STDOUT: !221 = !DILocation(line: 29, column: 26, scope: !217)
// CHECK:STDOUT: !224 = !DILocation(line: 27, column: 1, scope: !217)
// CHECK:STDOUT: !225 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.b81987abcfec64c8", scope: null, file: !10, line: 32, type: !25, spFlags: DISPFlagDefinition, unit: !173, retainedNodes: !231)
// CHECK:STDOUT: !226 = !DILocalVariable(arg: 1, scope: !225, type: !23)
// CHECK:STDOUT: !230 = !DILocation(line: 32, column: 1, scope: !225)
// CHECK:STDOUT: !231 = !{!226}
// CHECK:STDOUT: !232 = !{!"p1 _ZTS9MultiArgC", !45, i64 0}
// CHECK:STDOUT: !233 = !{!232, !232, i64 0}
// CHECK:STDOUT: !234 = !{!48, !48, i64 0}
// CHECK:STDOUT:
