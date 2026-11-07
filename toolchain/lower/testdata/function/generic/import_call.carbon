// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/int.carbon
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/parts/form.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/lower/testdata/function/generic/import_call.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/lower/testdata/function/generic/import_call.carbon

// Test calls in generics and templates where the caller, specific, and/or
// resolved callee are in different files, covering expression category
// computation, return storage slot lookup, and call lowering.

// --- lib_form.carbon
library "[[@TEST_NAME]]";

import Core library "prelude/parts/form";

class Big {
  var a: i32;
  var b: i32;
  var c: i32;
}

fn FreeGetVal(generic T: type, x: T) ->? form(val T) {
  return x;
}

fn FreeGetRef(generic T: type, p: T*) -> ref T {
  return *p;
}

interface Source {
  fn GetVal(self) ->? form(val i32);
  fn GetRef(self) -> ref i32;
  fn GetBig(self) -> Big;
  fn GetGeneric(self, generic U: type, p: U*) -> ref U;
}

fn FreeGetVar(generic T: Source, x: T) -> Big {
  return x.GetBig();
}

fn ForwardVal(generic T: Source, x: T) -> i32 {
  return x.GetVal();
}

fn ForwardRef(generic T: Source, x: T) -> ref i32 {
  return x.GetRef();
}

fn ForwardBig(generic T: Source, x: T) -> Big {
  return x.GetBig();
}

fn ForwardGeneric(generic T: Source, generic U: type, x: T, p: U*) -> ref U {
  return x.GetGeneric(U, p);
}

// --- use_form.carbon
library "[[@TEST_NAME]]";

import library "lib_form";

fn ForwardFreeVal(template T: type, x: T) -> T {
  return FreeGetVal(T, x);
}

fn ForwardFreeRef(template T: type, p: T*) -> ref T {
  return FreeGetRef(T, p);
}

fn ForwardFreeVar(template T: type, x: T) -> Big {
  return FreeGetVar(T, x);
}

class MySource {
  var p: i32*;
  impl as Source {
    fn GetVal(self) ->? form(val i32) { return *self.p; }
    fn GetRef(self) -> ref i32 { return *self.p; }
    fn GetBig(self) -> Big {
      return {.a = *self.p, .b = 2, .c = 3};
    }
    fn GetGeneric(unused self, generic U: type, q: U*) -> ref U {
      return *q;
    }
  }
}

fn TestForms() -> i32 {
  var target: i32 = ForwardFreeVal(i32, 1);
  let ref r: i32 = ForwardFreeRef(i32, &target);

  var s: MySource = {.p = &target};
  var b1: Big = ForwardFreeVar(MySource, s);
  let v2: i32 = ForwardVal(MySource, s);
  ForwardRef(MySource, s) = r;
  var b2: Big = ForwardBig(MySource, s);
  ForwardGeneric(MySource, i32, s, &b2.a) = v2;

  return b1.a;
}

// --- base_init.carbon
library "[[@TEST_NAME]]";

class Big {
  var a: i32;
  var b: i32;
  var c: i32;
}

interface ValSource {
  fn Get(self) -> i32;
}

eval fn MakeImportedConstBig() -> Big {
  return {.a = 10, .b = 20, .c = 30};
}

fn MakeGenericBig(unused generic U: type, x: i32) -> Big {
  return {.a = x, .b = 2, .c = 3};
}

class ImportedMaker {
  var v: i32;
  fn MakeFromSource(self, generic V: ValSource, src: V) -> Big {
    return {.a = self.v, .b = src.Get(), .c = 3};
  }
  eval fn MakeStaticConst() -> Big {
    return {.a = 70, .b = 80, .c = 90};
  }
}

// --- lib_init.carbon
library "[[@TEST_NAME]]";

import library "base_init";

class Outer {
  var runtime_val: i32;
  var const_big: Big;
}

eval fn MakeLocalConstBig() -> Big {
  return {.a = 40, .b = 50, .c = 60};
}

fn BuildOuter(generic T: ValSource, x: T) -> Outer {
  return {.runtime_val = x.Get(), .const_big = MakeImportedConstBig()};
}

fn BuildOuterLocal(generic T: ValSource, x: T) -> Outer {
  return {.runtime_val = x.Get(), .const_big = MakeLocalConstBig()};
}

// --- use_init.carbon
library "[[@TEST_NAME]]";

import library "base_init";
import library "lib_init";

class Maker {
  var v: i32;
  impl as ValSource {
    fn Get(self) -> i32 { return self.v; }
  }
}

fn ReturnTemplateDest(template Dest: type, x: i32) -> Dest {
  return MakeGenericBig(i32, x);
}

fn CallTemplateMethod(template T: type, generic U: ValSource, x: T, y: U) -> Big {
  return x.MakeFromSource(U, y);
}

fn BuildTemplateOuter(template T: type, x: i32) -> Outer {
  return {.runtime_val = x, .const_big = T.MakeStaticConst()};
}

fn TestInit(x: i32) -> i32 {
  var m: Maker = {.v = x};
  var im: ImportedMaker = {.v = x};
  var o1: Outer = BuildOuter(Maker, m);
  var o2: Outer = BuildOuterLocal(Maker, m);
  var b1: Big = ReturnTemplateDest(Big, o1.const_big.a);
  var b2: Big = CallTemplateMethod(ImportedMaker, Maker, im, m);
  var o3: Outer = BuildTemplateOuter(ImportedMaker, o2.const_big.c);
  b2.b = b1.a;
  return o3.const_big.a;
}

// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib_form.carbon'
// CHECK:STDOUT: source_filename = "lib_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!19}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !18 = !DIFile(filename: "lib_form.carbon", directory: "")
// CHECK:STDOUT: !19 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !18, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_form.carbon'
// CHECK:STDOUT: source_filename = "use_form.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CGetVal.MySource.Main:Source.Main"(ptr %self) #0 !dbg !32 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc20_53.1.p = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !37
// CHECK:STDOUT:   %.loc20_53.2 = load ptr, ptr %.loc20_53.1.p, align 8, !dbg !37
// CHECK:STDOUT:   %.loc20_48.2 = load i32, ptr %.loc20_53.2, align 4, !dbg !38
// CHECK:STDOUT:   ret i32 %.loc20_48.2, !dbg !39
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define ptr @"_CGetRef.MySource.Main:Source.Main"(ptr %self) #0 !dbg !41 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc21_46.1.p = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !46
// CHECK:STDOUT:   %.loc21_46.2 = load ptr, ptr %.loc21_46.1.p, align 8, !dbg !46
// CHECK:STDOUT:   ret ptr %.loc21_46.2, !dbg !48
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @"_CGetBig.MySource.Main:Source.Main"(ptr sret({ i32, i32, i32 }) %return, ptr %self) #0 !dbg !50 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc23_25.1.p = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !55
// CHECK:STDOUT:   %.loc23_25.2 = load ptr, ptr %.loc23_25.1.p, align 8, !dbg !55
// CHECK:STDOUT:   %.loc23_20.2 = load i32, ptr %.loc23_25.2, align 4, !dbg !56
// CHECK:STDOUT:   %.loc23_43.2.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !57
// CHECK:STDOUT:   store i32 %.loc23_20.2, ptr %.loc23_43.2.a, align 4, !dbg !57
// CHECK:STDOUT:   %.loc23_43.5.b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !57
// CHECK:STDOUT:   %.loc23_43.8.c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !57
// CHECK:STDOUT:   store i32 2, ptr %.loc23_43.5.b, align 4, !dbg !57
// CHECK:STDOUT:   store i32 3, ptr %.loc23_43.8.c, align 4, !dbg !57
// CHECK:STDOUT:   ret void, !dbg !58
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestForms.Main() #0 !dbg !60 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %target.var = alloca i32, align 4, !dbg !63
// CHECK:STDOUT:   %s.var = alloca { ptr }, align 8, !dbg !73
// CHECK:STDOUT:   %b1.var = alloca { i32, i32, i32 }, align 4, !dbg !78
// CHECK:STDOUT:   %b2.var = alloca { i32, i32, i32 }, align 4, !dbg !89
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %target.var), !dbg !63
// CHECK:STDOUT:   %ForwardFreeVal.call = call i32 @_CForwardFreeVal.Main.55456748a8a2eb70(i32 1), !dbg !64
// CHECK:STDOUT:   store i32 %ForwardFreeVal.call, ptr %target.var, align 4, !dbg !63
// CHECK:STDOUT:   %ForwardFreeRef.call = call ptr @_CForwardFreeRef.Main.55456748a8a2eb70(ptr %target.var), !dbg !70
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %s.var), !dbg !73
// CHECK:STDOUT:   %.loc35_34.2.p = getelementptr inbounds nuw { ptr }, ptr %s.var, i32 0, i32 0, !dbg !76
// CHECK:STDOUT:   store ptr %target.var, ptr %.loc35_34.2.p, align 8, !dbg !76
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b1.var), !dbg !78
// CHECK:STDOUT:   call void @_CForwardFreeVar.Main.0a0992ca8c4e9c06(ptr %b1.var, ptr %s.var), !dbg !80
// CHECK:STDOUT:   %ForwardVal.call = call i32 @_CForwardVal.Main.a51aacd80b073269(ptr %s.var), !dbg !83
// CHECK:STDOUT:   %ForwardRef.call = call ptr @_CForwardRef.Main.a51aacd80b073269(ptr %s.var), !dbg !87
// CHECK:STDOUT:   %.loc38_29 = load i32, ptr %ForwardFreeRef.call, align 4, !dbg !88
// CHECK:STDOUT:   store i32 %.loc38_29, ptr %ForwardRef.call, align 4, !dbg !87
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b2.var), !dbg !89
// CHECK:STDOUT:   call void @_CForwardBig.Main.a51aacd80b073269(ptr %b2.var, ptr %s.var), !dbg !91
// CHECK:STDOUT:   %.loc40_39.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b2.var, i32 0, i32 0, !dbg !94
// CHECK:STDOUT:   %ForwardGeneric.call = call ptr @_CForwardGeneric.Main.6e320fc0252c31a0(ptr %s.var, ptr %.loc40_39.a), !dbg !96
// CHECK:STDOUT:   store i32 %ForwardVal.call, ptr %ForwardGeneric.call, align 4, !dbg !96
// CHECK:STDOUT:   %.loc42_12.1.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b1.var, i32 0, i32 0, !dbg !100
// CHECK:STDOUT:   %.loc42_12.2 = load i32, ptr %.loc42_12.1.a, align 4, !dbg !100
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0ee170c3ec748bce(ptr %b2.var), !dbg !89
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0ee170c3ec748bce(ptr %b1.var), !dbg !78
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.c562a8b1c20d3f73(ptr %s.var), !dbg !73
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %target.var), !dbg !63
// CHECK:STDOUT:   ret i32 %.loc42_12.2, !dbg !103
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.df522884bb4ad941:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !104 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc39_3.1.c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !106
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc39_3.1.c), !dbg !106
// CHECK:STDOUT:   %.loc39_3.4.b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !106
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc39_3.4.b), !dbg !106
// CHECK:STDOUT:   %.loc39_3.7.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !106
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc39_3.7.a), !dbg !106
// CHECK:STDOUT:   ret void, !dbg !106
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b77f5f4f1cfd462c:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !108 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc39_3.1.c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !110
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc39_3.1.c), !dbg !110
// CHECK:STDOUT:   %.loc39_3.4.b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !110
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc39_3.4.b), !dbg !110
// CHECK:STDOUT:   %.loc39_3.7.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !110
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc39_3.7.a), !dbg !110
// CHECK:STDOUT:   ret void, !dbg !110
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.40e36d95453e0094:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !112 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_3.1.p = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !114
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %.loc35_3.1.p), !dbg !114
// CHECK:STDOUT:   ret void, !dbg !114
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.573397898da0c9fd:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !116 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc35_3.1.p = getelementptr inbounds nuw { ptr }, ptr %self, i32 0, i32 0, !dbg !118
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %.loc35_3.1.p), !dbg !118
// CHECK:STDOUT:   ret void, !dbg !118
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CForwardFreeVal.Main.55456748a8a2eb70(i32 %x) #0 !dbg !120 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %FreeGetVal.call = call i32 @_CFreeGetVal.Main.55456748a8a2eb70(i32 %x), !dbg !126
// CHECK:STDOUT:   ret i32 %FreeGetVal.call, !dbg !127
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CForwardFreeRef.Main.55456748a8a2eb70(ptr %p) #0 !dbg !129 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %FreeGetRef.call = call ptr @_CFreeGetRef.Main.55456748a8a2eb70(ptr %p), !dbg !135
// CHECK:STDOUT:   ret ptr %FreeGetRef.call, !dbg !136
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CForwardFreeVar.Main.0a0992ca8c4e9c06(ptr sret({ i32, i32, i32 }) %return, ptr %x) #0 !dbg !138 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CFreeGetVar.Main.a51aacd80b073269(ptr %return, ptr %x), !dbg !144
// CHECK:STDOUT:   ret void, !dbg !145
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CForwardVal.Main.a51aacd80b073269(ptr %x) #0 !dbg !147 {
// CHECK:STDOUT:   %1 = call i32 @"_CGetVal.MySource.Main:Source.Main"(ptr %x), !dbg !152
// CHECK:STDOUT:   ret i32 %1, !dbg !153
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CForwardRef.Main.a51aacd80b073269(ptr %x) #0 !dbg !155 {
// CHECK:STDOUT:   %1 = call ptr @"_CGetRef.MySource.Main:Source.Main"(ptr %x), !dbg !160
// CHECK:STDOUT:   ret ptr %1, !dbg !161
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CForwardBig.Main.a51aacd80b073269(ptr sret({ i32, i32, i32 }) %return, ptr %x) #0 !dbg !163 {
// CHECK:STDOUT:   call void @"_CGetBig.MySource.Main:Source.Main"(ptr %return, ptr %x), !dbg !168
// CHECK:STDOUT:   ret void, !dbg !169
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CForwardGeneric.Main.6e320fc0252c31a0(ptr %x, ptr %p) #0 !dbg !171 {
// CHECK:STDOUT:   %1 = call ptr @"_CGetGeneric.MySource.Main:Source.Main.55456748a8a2eb70"(ptr %x, ptr %p), !dbg !178
// CHECK:STDOUT:   ret ptr %1, !dbg !180
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0ee170c3ec748bce(ptr %t) #0 !dbg !182 {
// CHECK:STDOUT:   call void @"_COp.b77f5f4f1cfd462c:core.SubobjectDestroy.Core"(ptr %t), !dbg !186
// CHECK:STDOUT:   ret void, !dbg !187
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.c562a8b1c20d3f73(ptr %t) #0 !dbg !189 {
// CHECK:STDOUT:   call void @"_COp.573397898da0c9fd:core.SubobjectDestroy.Core"(ptr %t), !dbg !193
// CHECK:STDOUT:   ret void, !dbg !194
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !196 {
// CHECK:STDOUT:   ret void, !dbg !201
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.4f29d15a20894e9b(ptr %t) #0 !dbg !203 {
// CHECK:STDOUT:   ret void, !dbg !208
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr i32 @_CFreeGetVal.Main.55456748a8a2eb70(i32 %x) #0 !dbg !210 {
// CHECK:STDOUT:   ret i32 %x, !dbg !216
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @_CFreeGetRef.Main.55456748a8a2eb70(ptr %p) #0 !dbg !218 {
// CHECK:STDOUT:   ret ptr %p, !dbg !225
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CFreeGetVar.Main.a51aacd80b073269(ptr sret({ i32, i32, i32 }) %return, ptr %x) #0 !dbg !227 {
// CHECK:STDOUT:   call void @"_CGetBig.MySource.Main:Source.Main"(ptr %return, ptr %x), !dbg !232
// CHECK:STDOUT:   ret void, !dbg !233
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr ptr @"_CGetGeneric.MySource.Main:Source.Main.55456748a8a2eb70"(ptr %self, ptr %q) #0 !dbg !235 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   ret ptr %q, !dbg !244
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0ee170c3ec748bce, { 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.4f29d15a20894e9b, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!22}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !12 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !18 = !DIFile(filename: "lib_form.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !21 = !DIFile(filename: "use_form.carbon", directory: "")
// CHECK:STDOUT: !22 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !21, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!23, !23}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !28 = !{!20}
// CHECK:STDOUT: !29 = !DISubroutineType(types: !28)
// CHECK:STDOUT: !30 = !{null, !23}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !32 = distinct !DISubprogram(name: "GetVal", linkageName: "_CGetVal.MySource.Main:Source.Main", scope: null, file: !21, line: 20, type: !25, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !40)
// CHECK:STDOUT: !33 = !DILocalVariable(arg: 1, scope: !32, type: !23)
// CHECK:STDOUT: !37 = !DILocation(line: 20, column: 49, scope: !32)
// CHECK:STDOUT: !38 = !DILocation(line: 20, column: 48, scope: !32)
// CHECK:STDOUT: !39 = !DILocation(line: 20, column: 41, scope: !32)
// CHECK:STDOUT: !40 = !{!33}
// CHECK:STDOUT: !41 = distinct !DISubprogram(name: "GetRef", linkageName: "_CGetRef.MySource.Main:Source.Main", scope: null, file: !21, line: 21, type: !27, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !49)
// CHECK:STDOUT: !42 = !DILocalVariable(arg: 1, scope: !41, type: !23)
// CHECK:STDOUT: !46 = !DILocation(line: 21, column: 42, scope: !41)
// CHECK:STDOUT: !48 = !DILocation(line: 21, column: 34, scope: !41)
// CHECK:STDOUT: !49 = !{!42}
// CHECK:STDOUT: !50 = distinct !DISubprogram(name: "GetBig", linkageName: "_CGetBig.MySource.Main:Source.Main", scope: null, file: !21, line: 22, type: !27, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !59)
// CHECK:STDOUT: !51 = !DILocalVariable(arg: 1, scope: !50, type: !23)
// CHECK:STDOUT: !55 = !DILocation(line: 23, column: 21, scope: !50)
// CHECK:STDOUT: !56 = !DILocation(line: 23, column: 20, scope: !50)
// CHECK:STDOUT: !57 = !DILocation(line: 23, column: 14, scope: !50)
// CHECK:STDOUT: !58 = !DILocation(line: 23, column: 7, scope: !50)
// CHECK:STDOUT: !59 = !{!51}
// CHECK:STDOUT: !60 = distinct !DISubprogram(name: "TestForms", linkageName: "_CTestForms.Main", scope: null, file: !21, line: 31, type: !29, spFlags: DISPFlagDefinition, unit: !22)
// CHECK:STDOUT: !63 = !DILocation(line: 32, column: 3, scope: !60)
// CHECK:STDOUT: !64 = !DILocation(line: 32, column: 21, scope: !60)
// CHECK:STDOUT: !65 = !{!20, !20}
// CHECK:STDOUT: !66 = !DISubroutineType(types: !65)
// CHECK:STDOUT: !70 = !DILocation(line: 33, column: 20, scope: !60)
// CHECK:STDOUT: !73 = !DILocation(line: 35, column: 3, scope: !60)
// CHECK:STDOUT: !76 = !DILocation(line: 35, column: 21, scope: !60)
// CHECK:STDOUT: !78 = !DILocation(line: 36, column: 3, scope: !60)
// CHECK:STDOUT: !80 = !DILocation(line: 36, column: 17, scope: !60)
// CHECK:STDOUT: !83 = !DILocation(line: 37, column: 17, scope: !60)
// CHECK:STDOUT: !87 = !DILocation(line: 38, column: 3, scope: !60)
// CHECK:STDOUT: !88 = !DILocation(line: 38, column: 29, scope: !60)
// CHECK:STDOUT: !89 = !DILocation(line: 39, column: 3, scope: !60)
// CHECK:STDOUT: !91 = !DILocation(line: 39, column: 17, scope: !60)
// CHECK:STDOUT: !94 = !DILocation(line: 40, column: 37, scope: !60)
// CHECK:STDOUT: !96 = !DILocation(line: 40, column: 3, scope: !60)
// CHECK:STDOUT: !97 = !{!23, !23, !23}
// CHECK:STDOUT: !98 = !DISubroutineType(types: !97)
// CHECK:STDOUT: !100 = !DILocation(line: 42, column: 10, scope: !60)
// CHECK:STDOUT: !101 = !{null, !20}
// CHECK:STDOUT: !102 = !DISubroutineType(types: !101)
// CHECK:STDOUT: !103 = !DILocation(line: 42, column: 3, scope: !60)
// CHECK:STDOUT: !104 = distinct !DISubprogram(name: "Op", linkageName: "_COp.df522884bb4ad941:core.SubobjectDestroy.Core", scope: null, file: !21, line: 39, type: !31, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !107)
// CHECK:STDOUT: !105 = !DILocalVariable(arg: 1, scope: !104, type: !23)
// CHECK:STDOUT: !106 = !DILocation(line: 39, column: 3, scope: !104)
// CHECK:STDOUT: !107 = !{!105}
// CHECK:STDOUT: !108 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b77f5f4f1cfd462c:core.SubobjectDestroy.Core", scope: null, file: !21, line: 39, type: !31, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !111)
// CHECK:STDOUT: !109 = !DILocalVariable(arg: 1, scope: !108, type: !23)
// CHECK:STDOUT: !110 = !DILocation(line: 39, column: 3, scope: !108)
// CHECK:STDOUT: !111 = !{!109}
// CHECK:STDOUT: !112 = distinct !DISubprogram(name: "Op", linkageName: "_COp.40e36d95453e0094:core.SubobjectDestroy.Core", scope: null, file: !21, line: 35, type: !31, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !115)
// CHECK:STDOUT: !113 = !DILocalVariable(arg: 1, scope: !112, type: !23)
// CHECK:STDOUT: !114 = !DILocation(line: 35, column: 3, scope: !112)
// CHECK:STDOUT: !115 = !{!113}
// CHECK:STDOUT: !116 = distinct !DISubprogram(name: "Op", linkageName: "_COp.573397898da0c9fd:core.SubobjectDestroy.Core", scope: null, file: !21, line: 35, type: !31, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !119)
// CHECK:STDOUT: !117 = !DILocalVariable(arg: 1, scope: !116, type: !23)
// CHECK:STDOUT: !118 = !DILocation(line: 35, column: 3, scope: !116)
// CHECK:STDOUT: !119 = !{!117}
// CHECK:STDOUT: !120 = distinct !DISubprogram(name: "ForwardFreeVal", linkageName: "_CForwardFreeVal.Main.55456748a8a2eb70", scope: null, file: !21, line: 5, type: !66, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !128)
// CHECK:STDOUT: !121 = !DILocalVariable(arg: 1, scope: !120, type: !20)
// CHECK:STDOUT: !126 = !DILocation(line: 6, column: 10, scope: !120)
// CHECK:STDOUT: !127 = !DILocation(line: 6, column: 3, scope: !120)
// CHECK:STDOUT: !128 = !{!121}
// CHECK:STDOUT: !129 = distinct !DISubprogram(name: "ForwardFreeRef", linkageName: "_CForwardFreeRef.Main.55456748a8a2eb70", scope: null, file: !21, line: 9, type: !27, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !137)
// CHECK:STDOUT: !130 = !DILocalVariable(arg: 1, scope: !129, type: !23)
// CHECK:STDOUT: !135 = !DILocation(line: 10, column: 10, scope: !129)
// CHECK:STDOUT: !136 = !DILocation(line: 10, column: 3, scope: !129)
// CHECK:STDOUT: !137 = !{!130}
// CHECK:STDOUT: !138 = distinct !DISubprogram(name: "ForwardFreeVar", linkageName: "_CForwardFreeVar.Main.0a0992ca8c4e9c06", scope: null, file: !21, line: 13, type: !27, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !146)
// CHECK:STDOUT: !139 = !DILocalVariable(arg: 1, scope: !138, type: !23)
// CHECK:STDOUT: !144 = !DILocation(line: 14, column: 10, scope: !138)
// CHECK:STDOUT: !145 = !DILocation(line: 14, column: 3, scope: !138)
// CHECK:STDOUT: !146 = !{!139}
// CHECK:STDOUT: !147 = distinct !DISubprogram(name: "ForwardVal", linkageName: "_CForwardVal.Main.a51aacd80b073269", scope: null, file: !18, line: 30, type: !25, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !154)
// CHECK:STDOUT: !148 = !DILocalVariable(arg: 1, scope: !147, type: !23)
// CHECK:STDOUT: !152 = !DILocation(line: 31, column: 10, scope: !147)
// CHECK:STDOUT: !153 = !DILocation(line: 31, column: 3, scope: !147)
// CHECK:STDOUT: !154 = !{!148}
// CHECK:STDOUT: !155 = distinct !DISubprogram(name: "ForwardRef", linkageName: "_CForwardRef.Main.a51aacd80b073269", scope: null, file: !18, line: 34, type: !27, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !162)
// CHECK:STDOUT: !156 = !DILocalVariable(arg: 1, scope: !155, type: !23)
// CHECK:STDOUT: !160 = !DILocation(line: 35, column: 10, scope: !155)
// CHECK:STDOUT: !161 = !DILocation(line: 35, column: 3, scope: !155)
// CHECK:STDOUT: !162 = !{!156}
// CHECK:STDOUT: !163 = distinct !DISubprogram(name: "ForwardBig", linkageName: "_CForwardBig.Main.a51aacd80b073269", scope: null, file: !18, line: 38, type: !27, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !170)
// CHECK:STDOUT: !164 = !DILocalVariable(arg: 1, scope: !163, type: !23)
// CHECK:STDOUT: !168 = !DILocation(line: 39, column: 10, scope: !163)
// CHECK:STDOUT: !169 = !DILocation(line: 39, column: 3, scope: !163)
// CHECK:STDOUT: !170 = !{!164}
// CHECK:STDOUT: !171 = distinct !DISubprogram(name: "ForwardGeneric", linkageName: "_CForwardGeneric.Main.6e320fc0252c31a0", scope: null, file: !18, line: 42, type: !98, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !181)
// CHECK:STDOUT: !172 = !DILocalVariable(arg: 1, scope: !171, type: !23)
// CHECK:STDOUT: !173 = !DILocalVariable(arg: 2, scope: !171, type: !23)
// CHECK:STDOUT: !178 = !DILocation(line: 43, column: 10, scope: !171)
// CHECK:STDOUT: !180 = !DILocation(line: 43, column: 3, scope: !171)
// CHECK:STDOUT: !181 = !{!172, !173}
// CHECK:STDOUT: !182 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0ee170c3ec748bce", scope: null, file: !12, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !188)
// CHECK:STDOUT: !183 = !DILocalVariable(arg: 1, scope: !182, type: !23)
// CHECK:STDOUT: !186 = !DILocation(line: 34, column: 5, scope: !182)
// CHECK:STDOUT: !187 = !DILocation(line: 32, column: 1, scope: !182)
// CHECK:STDOUT: !188 = !{!183}
// CHECK:STDOUT: !189 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.c562a8b1c20d3f73", scope: null, file: !12, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !195)
// CHECK:STDOUT: !190 = !DILocalVariable(arg: 1, scope: !189, type: !23)
// CHECK:STDOUT: !193 = !DILocation(line: 34, column: 5, scope: !189)
// CHECK:STDOUT: !194 = !DILocation(line: 32, column: 1, scope: !189)
// CHECK:STDOUT: !195 = !{!190}
// CHECK:STDOUT: !196 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !12, line: 32, type: !102, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !202)
// CHECK:STDOUT: !197 = !DILocalVariable(arg: 1, scope: !196, type: !20)
// CHECK:STDOUT: !201 = !DILocation(line: 32, column: 1, scope: !196)
// CHECK:STDOUT: !202 = !{!197}
// CHECK:STDOUT: !203 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.4f29d15a20894e9b", scope: null, file: !12, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !209)
// CHECK:STDOUT: !204 = !DILocalVariable(arg: 1, scope: !203, type: !23)
// CHECK:STDOUT: !208 = !DILocation(line: 32, column: 1, scope: !203)
// CHECK:STDOUT: !209 = !{!204}
// CHECK:STDOUT: !210 = distinct !DISubprogram(name: "FreeGetVal", linkageName: "_CFreeGetVal.Main.55456748a8a2eb70", scope: null, file: !18, line: 11, type: !66, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !217)
// CHECK:STDOUT: !211 = !DILocalVariable(arg: 1, scope: !210, type: !20)
// CHECK:STDOUT: !216 = !DILocation(line: 12, column: 3, scope: !210)
// CHECK:STDOUT: !217 = !{!211}
// CHECK:STDOUT: !218 = distinct !DISubprogram(name: "FreeGetRef", linkageName: "_CFreeGetRef.Main.55456748a8a2eb70", scope: null, file: !18, line: 15, type: !27, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !226)
// CHECK:STDOUT: !219 = !DILocalVariable(arg: 1, scope: !218, type: !23)
// CHECK:STDOUT: !225 = !DILocation(line: 16, column: 3, scope: !218)
// CHECK:STDOUT: !226 = !{!219}
// CHECK:STDOUT: !227 = distinct !DISubprogram(name: "FreeGetVar", linkageName: "_CFreeGetVar.Main.a51aacd80b073269", scope: null, file: !18, line: 26, type: !27, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !234)
// CHECK:STDOUT: !228 = !DILocalVariable(arg: 1, scope: !227, type: !23)
// CHECK:STDOUT: !232 = !DILocation(line: 27, column: 10, scope: !227)
// CHECK:STDOUT: !233 = !DILocation(line: 27, column: 3, scope: !227)
// CHECK:STDOUT: !234 = !{!228}
// CHECK:STDOUT: !235 = distinct !DISubprogram(name: "GetGeneric", linkageName: "_CGetGeneric.MySource.Main:Source.Main.55456748a8a2eb70", scope: null, file: !21, line: 25, type: !98, spFlags: DISPFlagDefinition, unit: !22, retainedNodes: !245)
// CHECK:STDOUT: !236 = !DILocalVariable(arg: 1, scope: !235, type: !23)
// CHECK:STDOUT: !237 = !DILocalVariable(arg: 2, scope: !235, type: !23)
// CHECK:STDOUT: !244 = !DILocation(line: 26, column: 7, scope: !235)
// CHECK:STDOUT: !245 = !{!236, !237}
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'base_init.carbon'
// CHECK:STDOUT: source_filename = "base_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Big.val.561.loc14_37 = internal constant { i32, i32, i32 } { i32 10, i32 20, i32 30 }
// CHECK:STDOUT: @Big.val.b90.loc27_39 = internal constant { i32, i32, i32 } { i32 70, i32 80, i32 90 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeImportedConstBig.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !250 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc14_36.3.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !253
// CHECK:STDOUT:   %.loc14_36.6.b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !253
// CHECK:STDOUT:   %.loc14_36.9.c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !253
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @Big.val.561.loc14_37, i64 12, i1 false), !dbg !254
// CHECK:STDOUT:   ret void, !dbg !254
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeStaticConst.ImportedMaker.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !255 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_38.3.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !258
// CHECK:STDOUT:   %.loc27_38.6.b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !258
// CHECK:STDOUT:   %.loc27_38.9.c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !258
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @Big.val.b90.loc27_39, i64 12, i1 false), !dbg !259
// CHECK:STDOUT:   ret void, !dbg !259
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.memcpy.p0.p0.i64, { 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!247}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !246 = !DIFile(filename: "base_init.carbon", directory: "")
// CHECK:STDOUT: !247 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !246, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !248 = !{!23}
// CHECK:STDOUT: !249 = !DISubroutineType(types: !248)
// CHECK:STDOUT: !250 = distinct !DISubprogram(name: "MakeImportedConstBig", linkageName: "_CMakeImportedConstBig.Main", scope: null, file: !246, line: 13, type: !249, spFlags: DISPFlagDefinition, unit: !247)
// CHECK:STDOUT: !253 = !DILocation(line: 14, column: 10, scope: !250)
// CHECK:STDOUT: !254 = !DILocation(line: 14, column: 3, scope: !250)
// CHECK:STDOUT: !255 = distinct !DISubprogram(name: "MakeStaticConst", linkageName: "_CMakeStaticConst.ImportedMaker.Main", scope: null, file: !246, line: 26, type: !249, spFlags: DISPFlagDefinition, unit: !247)
// CHECK:STDOUT: !258 = !DILocation(line: 27, column: 12, scope: !255)
// CHECK:STDOUT: !259 = !DILocation(line: 27, column: 5, scope: !255)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'lib_init.carbon'
// CHECK:STDOUT: source_filename = "lib_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @Big.val.loc11_37 = internal constant { i32, i32, i32 } { i32 40, i32 50, i32 60 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define void @_CMakeLocalConstBig.Main(ptr sret({ i32, i32, i32 }) %return) #0 !dbg !262 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc11_36.3.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !265
// CHECK:STDOUT:   %.loc11_36.6.b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !265
// CHECK:STDOUT:   %.loc11_36.9.c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !265
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %return, ptr align 4 @Big.val.loc11_37, i64 12, i1 false), !dbg !266
// CHECK:STDOUT:   ret void, !dbg !266
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeImportedConstBig.Main(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!261}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !248 = !{!23}
// CHECK:STDOUT: !249 = !DISubroutineType(types: !248)
// CHECK:STDOUT: !260 = !DIFile(filename: "lib_init.carbon", directory: "")
// CHECK:STDOUT: !261 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !260, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !262 = distinct !DISubprogram(name: "MakeLocalConstBig", linkageName: "_CMakeLocalConstBig.Main", scope: null, file: !260, line: 10, type: !249, spFlags: DISPFlagDefinition, unit: !261)
// CHECK:STDOUT: !265 = !DILocation(line: 11, column: 10, scope: !262)
// CHECK:STDOUT: !266 = !DILocation(line: 11, column: 3, scope: !262)
// CHECK:STDOUT:
// CHECK:STDOUT: ; ---
// CHECK:STDOUT: ; ModuleID = 'use_init.carbon'
// CHECK:STDOUT: source_filename = "use_init.carbon"
// CHECK:STDOUT:
// CHECK:STDOUT: @const.anon = internal constant { i32, i32, i32 } { i32 40, i32 50, i32 60 }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @"_CGet.Maker.Main:ValSource.Main"(ptr %self) #0 !dbg !269 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc9_38.1.v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !274
// CHECK:STDOUT:   %.loc9_38.2 = load i32, ptr %.loc9_38.1.v, align 4, !dbg !274
// CHECK:STDOUT:   ret i32 %.loc9_38.2, !dbg !275
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define i32 @_CTestInit.Main(i32 %x) #0 !dbg !277 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %m.var = alloca { i32 }, align 4, !dbg !282
// CHECK:STDOUT:   %im.var = alloca { i32 }, align 4, !dbg !286
// CHECK:STDOUT:   %o1.var = alloca { i32, { i32, i32, i32 } }, align 4, !dbg !290
// CHECK:STDOUT:   %o2.var = alloca { i32, { i32, i32, i32 } }, align 4, !dbg !294
// CHECK:STDOUT:   %b1.var = alloca { i32, i32, i32 }, align 4, !dbg !298
// CHECK:STDOUT:   %b2.var = alloca { i32, i32, i32 }, align 4, !dbg !304
// CHECK:STDOUT:   %o3.var = alloca { i32, { i32, i32, i32 } }, align 4, !dbg !309
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %m.var), !dbg !282
// CHECK:STDOUT:   %.loc26_25.2.v = getelementptr inbounds nuw { i32 }, ptr %m.var, i32 0, i32 0, !dbg !284
// CHECK:STDOUT:   store i32 %x, ptr %.loc26_25.2.v, align 4, !dbg !284
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %im.var), !dbg !286
// CHECK:STDOUT:   %.loc27_34.2.v = getelementptr inbounds nuw { i32 }, ptr %im.var, i32 0, i32 0, !dbg !288
// CHECK:STDOUT:   store i32 %x, ptr %.loc27_34.2.v, align 4, !dbg !288
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o1.var), !dbg !290
// CHECK:STDOUT:   call void @_CBuildOuter.Main.930bcdc4607e1e7c(ptr %o1.var, ptr %m.var), !dbg !292
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o2.var), !dbg !294
// CHECK:STDOUT:   call void @_CBuildOuterLocal.Main.930bcdc4607e1e7c(ptr %o2.var, ptr %m.var), !dbg !296
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b1.var), !dbg !298
// CHECK:STDOUT:   %.loc30_43.const_big = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %o1.var, i32 0, i32 1, !dbg !299
// CHECK:STDOUT:   %.loc30_53.1.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc30_43.const_big, i32 0, i32 0, !dbg !299
// CHECK:STDOUT:   %.loc30_53.2 = load i32, ptr %.loc30_53.1.a, align 4, !dbg !299
// CHECK:STDOUT:   call void @_CReturnTemplateDest.Main.4e7cc8e2f5e96e27(ptr %b1.var, i32 %.loc30_53.2), !dbg !300
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %b2.var), !dbg !304
// CHECK:STDOUT:   call void @_CCallTemplateMethod.Main.097f88a388bbff1e(ptr %b2.var, ptr %im.var, ptr %m.var), !dbg !307
// CHECK:STDOUT:   call void @llvm.lifetime.start.p0(ptr %o3.var), !dbg !309
// CHECK:STDOUT:   %.loc32_55.const_big = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %o2.var, i32 0, i32 1, !dbg !310
// CHECK:STDOUT:   %.loc32_65.1.c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc32_55.const_big, i32 0, i32 2, !dbg !310
// CHECK:STDOUT:   %.loc32_65.2 = load i32, ptr %.loc32_65.1.c, align 4, !dbg !310
// CHECK:STDOUT:   call void @_CBuildTemplateOuter.Main.e2f71c3e03c0c53b(ptr %o3.var, i32 %.loc32_65.2), !dbg !311
// CHECK:STDOUT:   %.loc33_5.b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b2.var, i32 0, i32 1, !dbg !313
// CHECK:STDOUT:   %.loc33_12.1.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %b1.var, i32 0, i32 0, !dbg !314
// CHECK:STDOUT:   %.loc33_12.2 = load i32, ptr %.loc33_12.1.a, align 4, !dbg !314
// CHECK:STDOUT:   store i32 %.loc33_12.2, ptr %.loc33_5.b, align 4, !dbg !313
// CHECK:STDOUT:   %.loc34_12.const_big = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %o3.var, i32 0, i32 1, !dbg !315
// CHECK:STDOUT:   %.loc34_22.1.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %.loc34_12.const_big, i32 0, i32 0, !dbg !315
// CHECK:STDOUT:   %.loc34_22.2 = load i32, ptr %.loc34_22.1.a, align 4, !dbg !315
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d65f4fd31ab6d77e(ptr %o3.var), !dbg !309
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0ee170c3ec748bce(ptr %b2.var), !dbg !304
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0ee170c3ec748bce(ptr %b1.var), !dbg !298
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d65f4fd31ab6d77e(ptr %o2.var), !dbg !294
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.d65f4fd31ab6d77e(ptr %o1.var), !dbg !290
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.1009d1e53fe8aee6(ptr %im.var), !dbg !286
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.8a2301e6212711d6(ptr %m.var), !dbg !282
// CHECK:STDOUT:   ret i32 %.loc34_22.2, !dbg !316
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.df522884bb4ad941:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !318 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_3.1.c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !320
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc32_3.1.c), !dbg !320
// CHECK:STDOUT:   %.loc32_3.4.b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !320
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc32_3.4.b), !dbg !320
// CHECK:STDOUT:   %.loc32_3.7.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !320
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc32_3.7.a), !dbg !320
// CHECK:STDOUT:   ret void, !dbg !320
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.b77f5f4f1cfd462c:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !322 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_3.1.c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 2, !dbg !324
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc32_3.1.c), !dbg !324
// CHECK:STDOUT:   %.loc32_3.4.b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 1, !dbg !324
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc32_3.4.b), !dbg !324
// CHECK:STDOUT:   %.loc32_3.7.a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %self, i32 0, i32 0, !dbg !324
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc32_3.7.a), !dbg !324
// CHECK:STDOUT:   ret void, !dbg !324
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.1cab2ae95a5f1792:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !326 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_3.1.const_big = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !328
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0ee170c3ec748bce(ptr %.loc32_3.1.const_big), !dbg !328
// CHECK:STDOUT:   %.loc32_3.4.runtime_val = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %self, i32 0, i32 0, !dbg !328
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc32_3.4.runtime_val), !dbg !328
// CHECK:STDOUT:   ret void, !dbg !328
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e396a6d79af5a9e1:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !330 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc32_3.1.const_big = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %self, i32 0, i32 1, !dbg !332
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.0ee170c3ec748bce(ptr %.loc32_3.1.const_big), !dbg !332
// CHECK:STDOUT:   %.loc32_3.4.runtime_val = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %self, i32 0, i32 0, !dbg !332
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc32_3.4.runtime_val), !dbg !332
// CHECK:STDOUT:   ret void, !dbg !332
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.8ce97e948e8f3b32:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !334 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_3.1.v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !336
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc27_3.1.v), !dbg !336
// CHECK:STDOUT:   ret void, !dbg !336
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.70a31268f711a106:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !338 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc27_3.1.v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !340
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc27_3.1.v), !dbg !340
// CHECK:STDOUT:   ret void, !dbg !340
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define weak_odr void @"_COp.e8c0758765c14e26:core.SubobjectDestroy.Core"(ptr %self) #0 !dbg !342 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.loc26_3.1.v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !344
// CHECK:STDOUT:   call void @_CSelfDestruct.Core.f127f30143696d51(ptr %.loc26_3.1.v), !dbg !344
// CHECK:STDOUT:   ret void, !dbg !344
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeStaticConst.ImportedMaker.Main(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.lifetime.start.p0(ptr captures(none)) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CBuildOuter.Main.930bcdc4607e1e7c(ptr sret({ i32, { i32, i32, i32 } }) %return, ptr %x) #0 !dbg !346 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.Maker.Main:ValSource.Main"(ptr %x), !dbg !351
// CHECK:STDOUT:   %const_big = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !352
// CHECK:STDOUT:   call void @_CMakeImportedConstBig.Main(ptr %const_big), !dbg !353
// CHECK:STDOUT:   %runtime_val = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !352
// CHECK:STDOUT:   store i32 %1, ptr %runtime_val, align 4, !dbg !352
// CHECK:STDOUT:   ret void, !dbg !354
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CBuildOuterLocal.Main.930bcdc4607e1e7c(ptr sret({ i32, { i32, i32, i32 } }) %return, ptr %x) #0 !dbg !356 {
// CHECK:STDOUT:   %1 = call i32 @"_CGet.Maker.Main:ValSource.Main"(ptr %x), !dbg !361
// CHECK:STDOUT:   %const_big = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !362
// CHECK:STDOUT:   %runtime_val = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !362
// CHECK:STDOUT:   store i32 %1, ptr %runtime_val, align 4, !dbg !362
// CHECK:STDOUT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %const_big, ptr align 4 @const.anon, i64 12, i1 false), !dbg !362
// CHECK:STDOUT:   ret void, !dbg !363
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CReturnTemplateDest.Main.4e7cc8e2f5e96e27(ptr sret({ i32, i32, i32 }) %return, i32 %x) #0 !dbg !365 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakeGenericBig.Main.55456748a8a2eb70(ptr %return, i32 %x), !dbg !372
// CHECK:STDOUT:   ret void, !dbg !371
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CCallTemplateMethod.Main.097f88a388bbff1e(ptr sret({ i32, i32, i32 }) %return, ptr %x, ptr %y) #0 !dbg !374 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   call void @_CMakeFromSource.ImportedMaker.Main.930bcdc4607e1e7c(ptr %return, ptr %x, ptr %y), !dbg !381
// CHECK:STDOUT:   ret void, !dbg !383
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CBuildTemplateOuter.Main.e2f71c3e03c0c53b(ptr sret({ i32, { i32, i32, i32 } }) %return, i32 %x) #0 !dbg !385 {
// CHECK:STDOUT: entry:
// CHECK:STDOUT:   %.15a388.1.const_big = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %return, i32 0, i32 1, !dbg !392
// CHECK:STDOUT:   call void @_CMakeStaticConst.ImportedMaker.Main(ptr %.15a388.1.const_big), !dbg !391
// CHECK:STDOUT:   %.loc22_61.2.runtime_val = getelementptr inbounds nuw { i32, { i32, i32, i32 } }, ptr %return, i32 0, i32 0, !dbg !392
// CHECK:STDOUT:   store i32 %x, ptr %.loc22_61.2.runtime_val, align 4, !dbg !392
// CHECK:STDOUT:   ret void, !dbg !393
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.d65f4fd31ab6d77e(ptr %t) #0 !dbg !395 {
// CHECK:STDOUT:   call void @"_COp.e396a6d79af5a9e1:core.SubobjectDestroy.Core"(ptr %t), !dbg !399
// CHECK:STDOUT:   ret void, !dbg !400
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.0ee170c3ec748bce(ptr %t) #0 !dbg !402 {
// CHECK:STDOUT:   call void @"_COp.b77f5f4f1cfd462c:core.SubobjectDestroy.Core"(ptr %t), !dbg !406
// CHECK:STDOUT:   ret void, !dbg !407
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.1009d1e53fe8aee6(ptr %t) #0 !dbg !409 {
// CHECK:STDOUT:   call void @"_COp.70a31268f711a106:core.SubobjectDestroy.Core"(ptr %t), !dbg !413
// CHECK:STDOUT:   ret void, !dbg !414
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.8a2301e6212711d6(ptr %t) #0 !dbg !416 {
// CHECK:STDOUT:   call void @"_COp.e8c0758765c14e26:core.SubobjectDestroy.Core"(ptr %t), !dbg !420
// CHECK:STDOUT:   ret void, !dbg !421
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CSelfDestruct.Core.f127f30143696d51(ptr %t) #0 !dbg !423 {
// CHECK:STDOUT:   ret void, !dbg !428
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeLocalConstBig.Main(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: declare void @_CMakeImportedConstBig.Main(ptr sret({ i32, i32, i32 }))
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
// CHECK:STDOUT: declare void @llvm.memcpy.p0.p0.i64(ptr noalias writeonly captures(none), ptr noalias readonly captures(none), i64, i1 immarg) #1
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMakeGenericBig.Main.55456748a8a2eb70(ptr sret({ i32, i32, i32 }) %return, i32 %x) #0 !dbg !430 {
// CHECK:STDOUT:   %a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !436
// CHECK:STDOUT:   store i32 %x, ptr %a, align 4, !dbg !436
// CHECK:STDOUT:   %b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !436
// CHECK:STDOUT:   %c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !436
// CHECK:STDOUT:   store i32 2, ptr %b, align 4, !dbg !436
// CHECK:STDOUT:   store i32 3, ptr %c, align 4, !dbg !436
// CHECK:STDOUT:   ret void, !dbg !437
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; Function Attrs: nounwind
// CHECK:STDOUT: define linkonce_odr void @_CMakeFromSource.ImportedMaker.Main.930bcdc4607e1e7c(ptr sret({ i32, i32, i32 }) %return, ptr %self, ptr %src) #0 !dbg !439 {
// CHECK:STDOUT:   %v = getelementptr inbounds nuw { i32 }, ptr %self, i32 0, i32 0, !dbg !446
// CHECK:STDOUT:   %1 = load i32, ptr %v, align 4, !dbg !446
// CHECK:STDOUT:   %2 = call i32 @"_CGet.Maker.Main:ValSource.Main"(ptr %src), !dbg !447
// CHECK:STDOUT:   %a = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 0, !dbg !448
// CHECK:STDOUT:   store i32 %1, ptr %a, align 4, !dbg !448
// CHECK:STDOUT:   %b = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 1, !dbg !448
// CHECK:STDOUT:   store i32 %2, ptr %b, align 4, !dbg !448
// CHECK:STDOUT:   %c = getelementptr inbounds nuw { i32, i32, i32 }, ptr %return, i32 0, i32 2, !dbg !448
// CHECK:STDOUT:   store i32 3, ptr %c, align 4, !dbg !448
// CHECK:STDOUT:   ret void, !dbg !449
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: ; uselistorder directives
// CHECK:STDOUT: uselistorder ptr @llvm.lifetime.start.p0, { 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.d65f4fd31ab6d77e, { 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.0ee170c3ec748bce, { 3, 2, 1, 0 }
// CHECK:STDOUT: uselistorder ptr @_CSelfDestruct.Core.f127f30143696d51, { 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 }
// CHECK:STDOUT:
// CHECK:STDOUT: attributes #0 = { nounwind }
// CHECK:STDOUT: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
// CHECK:STDOUT:
// CHECK:STDOUT: !llvm.dbg.cu = !{!268}
// CHECK:STDOUT: !llvm.module.flags = !{!2, !3}
// CHECK:STDOUT:
// CHECK:STDOUT: !2 = !{i32 7, !"Dwarf Version", i32 5}
// CHECK:STDOUT: !3 = !{i32 2, !"Debug Info Version", i32 3}
// CHECK:STDOUT: !12 = !DIFile(filename: "min_prelude/parts/destroy.carbon", directory: "")
// CHECK:STDOUT: !20 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
// CHECK:STDOUT: !23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
// CHECK:STDOUT: !24 = !{!20, !23}
// CHECK:STDOUT: !25 = !DISubroutineType(types: !24)
// CHECK:STDOUT: !26 = !{!23, !23}
// CHECK:STDOUT: !27 = !DISubroutineType(types: !26)
// CHECK:STDOUT: !30 = !{null, !23}
// CHECK:STDOUT: !31 = !DISubroutineType(types: !30)
// CHECK:STDOUT: !65 = !{!20, !20}
// CHECK:STDOUT: !66 = !DISubroutineType(types: !65)
// CHECK:STDOUT: !97 = !{!23, !23, !23}
// CHECK:STDOUT: !98 = !DISubroutineType(types: !97)
// CHECK:STDOUT: !101 = !{null, !20}
// CHECK:STDOUT: !102 = !DISubroutineType(types: !101)
// CHECK:STDOUT: !246 = !DIFile(filename: "base_init.carbon", directory: "")
// CHECK:STDOUT: !260 = !DIFile(filename: "lib_init.carbon", directory: "")
// CHECK:STDOUT: !267 = !DIFile(filename: "use_init.carbon", directory: "")
// CHECK:STDOUT: !268 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !267, producer: "carbon", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
// CHECK:STDOUT: !269 = distinct !DISubprogram(name: "Get", linkageName: "_CGet.Maker.Main:ValSource.Main", scope: null, file: !267, line: 9, type: !25, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !276)
// CHECK:STDOUT: !270 = !DILocalVariable(arg: 1, scope: !269, type: !23)
// CHECK:STDOUT: !274 = !DILocation(line: 9, column: 34, scope: !269)
// CHECK:STDOUT: !275 = !DILocation(line: 9, column: 27, scope: !269)
// CHECK:STDOUT: !276 = !{!270}
// CHECK:STDOUT: !277 = distinct !DISubprogram(name: "TestInit", linkageName: "_CTestInit.Main", scope: null, file: !267, line: 25, type: !66, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !317)
// CHECK:STDOUT: !278 = !DILocalVariable(arg: 1, scope: !277, type: !20)
// CHECK:STDOUT: !282 = !DILocation(line: 26, column: 3, scope: !277)
// CHECK:STDOUT: !284 = !DILocation(line: 26, column: 18, scope: !277)
// CHECK:STDOUT: !286 = !DILocation(line: 27, column: 3, scope: !277)
// CHECK:STDOUT: !288 = !DILocation(line: 27, column: 27, scope: !277)
// CHECK:STDOUT: !290 = !DILocation(line: 28, column: 3, scope: !277)
// CHECK:STDOUT: !292 = !DILocation(line: 28, column: 19, scope: !277)
// CHECK:STDOUT: !294 = !DILocation(line: 29, column: 3, scope: !277)
// CHECK:STDOUT: !296 = !DILocation(line: 29, column: 19, scope: !277)
// CHECK:STDOUT: !298 = !DILocation(line: 30, column: 3, scope: !277)
// CHECK:STDOUT: !299 = !DILocation(line: 30, column: 41, scope: !277)
// CHECK:STDOUT: !300 = !DILocation(line: 30, column: 17, scope: !277)
// CHECK:STDOUT: !301 = !{!23, !20}
// CHECK:STDOUT: !302 = !DISubroutineType(types: !301)
// CHECK:STDOUT: !304 = !DILocation(line: 31, column: 3, scope: !277)
// CHECK:STDOUT: !307 = !DILocation(line: 31, column: 17, scope: !277)
// CHECK:STDOUT: !309 = !DILocation(line: 32, column: 3, scope: !277)
// CHECK:STDOUT: !310 = !DILocation(line: 32, column: 53, scope: !277)
// CHECK:STDOUT: !311 = !DILocation(line: 32, column: 19, scope: !277)
// CHECK:STDOUT: !313 = !DILocation(line: 33, column: 3, scope: !277)
// CHECK:STDOUT: !314 = !DILocation(line: 33, column: 10, scope: !277)
// CHECK:STDOUT: !315 = !DILocation(line: 34, column: 10, scope: !277)
// CHECK:STDOUT: !316 = !DILocation(line: 34, column: 3, scope: !277)
// CHECK:STDOUT: !317 = !{!278}
// CHECK:STDOUT: !318 = distinct !DISubprogram(name: "Op", linkageName: "_COp.df522884bb4ad941:core.SubobjectDestroy.Core", scope: null, file: !267, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !321)
// CHECK:STDOUT: !319 = !DILocalVariable(arg: 1, scope: !318, type: !23)
// CHECK:STDOUT: !320 = !DILocation(line: 32, column: 3, scope: !318)
// CHECK:STDOUT: !321 = !{!319}
// CHECK:STDOUT: !322 = distinct !DISubprogram(name: "Op", linkageName: "_COp.b77f5f4f1cfd462c:core.SubobjectDestroy.Core", scope: null, file: !267, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !325)
// CHECK:STDOUT: !323 = !DILocalVariable(arg: 1, scope: !322, type: !23)
// CHECK:STDOUT: !324 = !DILocation(line: 32, column: 3, scope: !322)
// CHECK:STDOUT: !325 = !{!323}
// CHECK:STDOUT: !326 = distinct !DISubprogram(name: "Op", linkageName: "_COp.1cab2ae95a5f1792:core.SubobjectDestroy.Core", scope: null, file: !267, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !329)
// CHECK:STDOUT: !327 = !DILocalVariable(arg: 1, scope: !326, type: !23)
// CHECK:STDOUT: !328 = !DILocation(line: 32, column: 3, scope: !326)
// CHECK:STDOUT: !329 = !{!327}
// CHECK:STDOUT: !330 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e396a6d79af5a9e1:core.SubobjectDestroy.Core", scope: null, file: !267, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !333)
// CHECK:STDOUT: !331 = !DILocalVariable(arg: 1, scope: !330, type: !23)
// CHECK:STDOUT: !332 = !DILocation(line: 32, column: 3, scope: !330)
// CHECK:STDOUT: !333 = !{!331}
// CHECK:STDOUT: !334 = distinct !DISubprogram(name: "Op", linkageName: "_COp.8ce97e948e8f3b32:core.SubobjectDestroy.Core", scope: null, file: !267, line: 27, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !337)
// CHECK:STDOUT: !335 = !DILocalVariable(arg: 1, scope: !334, type: !23)
// CHECK:STDOUT: !336 = !DILocation(line: 27, column: 3, scope: !334)
// CHECK:STDOUT: !337 = !{!335}
// CHECK:STDOUT: !338 = distinct !DISubprogram(name: "Op", linkageName: "_COp.70a31268f711a106:core.SubobjectDestroy.Core", scope: null, file: !267, line: 27, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !341)
// CHECK:STDOUT: !339 = !DILocalVariable(arg: 1, scope: !338, type: !23)
// CHECK:STDOUT: !340 = !DILocation(line: 27, column: 3, scope: !338)
// CHECK:STDOUT: !341 = !{!339}
// CHECK:STDOUT: !342 = distinct !DISubprogram(name: "Op", linkageName: "_COp.e8c0758765c14e26:core.SubobjectDestroy.Core", scope: null, file: !267, line: 26, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !345)
// CHECK:STDOUT: !343 = !DILocalVariable(arg: 1, scope: !342, type: !23)
// CHECK:STDOUT: !344 = !DILocation(line: 26, column: 3, scope: !342)
// CHECK:STDOUT: !345 = !{!343}
// CHECK:STDOUT: !346 = distinct !DISubprogram(name: "BuildOuter", linkageName: "_CBuildOuter.Main.930bcdc4607e1e7c", scope: null, file: !260, line: 14, type: !27, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !355)
// CHECK:STDOUT: !347 = !DILocalVariable(arg: 1, scope: !346, type: !23)
// CHECK:STDOUT: !351 = !DILocation(line: 15, column: 26, scope: !346)
// CHECK:STDOUT: !352 = !DILocation(line: 15, column: 10, scope: !346)
// CHECK:STDOUT: !353 = !DILocation(line: 15, column: 48, scope: !346)
// CHECK:STDOUT: !354 = !DILocation(line: 15, column: 3, scope: !346)
// CHECK:STDOUT: !355 = !{!347}
// CHECK:STDOUT: !356 = distinct !DISubprogram(name: "BuildOuterLocal", linkageName: "_CBuildOuterLocal.Main.930bcdc4607e1e7c", scope: null, file: !260, line: 18, type: !27, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !364)
// CHECK:STDOUT: !357 = !DILocalVariable(arg: 1, scope: !356, type: !23)
// CHECK:STDOUT: !361 = !DILocation(line: 19, column: 26, scope: !356)
// CHECK:STDOUT: !362 = !DILocation(line: 19, column: 10, scope: !356)
// CHECK:STDOUT: !363 = !DILocation(line: 19, column: 3, scope: !356)
// CHECK:STDOUT: !364 = !{!357}
// CHECK:STDOUT: !365 = distinct !DISubprogram(name: "ReturnTemplateDest", linkageName: "_CReturnTemplateDest.Main.4e7cc8e2f5e96e27", scope: null, file: !267, line: 13, type: !302, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !373)
// CHECK:STDOUT: !366 = !DILocalVariable(arg: 1, scope: !365, type: !20)
// CHECK:STDOUT: !371 = !DILocation(line: 14, column: 3, scope: !365)
// CHECK:STDOUT: !372 = !DILocation(line: 14, column: 10, scope: !365)
// CHECK:STDOUT: !373 = !{!366}
// CHECK:STDOUT: !374 = distinct !DISubprogram(name: "CallTemplateMethod", linkageName: "_CCallTemplateMethod.Main.097f88a388bbff1e", scope: null, file: !267, line: 17, type: !98, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !384)
// CHECK:STDOUT: !375 = !DILocalVariable(arg: 1, scope: !374, type: !23)
// CHECK:STDOUT: !376 = !DILocalVariable(arg: 2, scope: !374, type: !23)
// CHECK:STDOUT: !381 = !DILocation(line: 18, column: 10, scope: !374)
// CHECK:STDOUT: !383 = !DILocation(line: 18, column: 3, scope: !374)
// CHECK:STDOUT: !384 = !{!375, !376}
// CHECK:STDOUT: !385 = distinct !DISubprogram(name: "BuildTemplateOuter", linkageName: "_CBuildTemplateOuter.Main.e2f71c3e03c0c53b", scope: null, file: !267, line: 21, type: !302, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !394)
// CHECK:STDOUT: !386 = !DILocalVariable(arg: 1, scope: !385, type: !20)
// CHECK:STDOUT: !391 = !DILocation(line: 22, column: 42, scope: !385)
// CHECK:STDOUT: !392 = !DILocation(line: 22, column: 10, scope: !385)
// CHECK:STDOUT: !393 = !DILocation(line: 22, column: 3, scope: !385)
// CHECK:STDOUT: !394 = !{!386}
// CHECK:STDOUT: !395 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.d65f4fd31ab6d77e", scope: null, file: !12, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !401)
// CHECK:STDOUT: !396 = !DILocalVariable(arg: 1, scope: !395, type: !23)
// CHECK:STDOUT: !399 = !DILocation(line: 34, column: 5, scope: !395)
// CHECK:STDOUT: !400 = !DILocation(line: 32, column: 1, scope: !395)
// CHECK:STDOUT: !401 = !{!396}
// CHECK:STDOUT: !402 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.0ee170c3ec748bce", scope: null, file: !12, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !408)
// CHECK:STDOUT: !403 = !DILocalVariable(arg: 1, scope: !402, type: !23)
// CHECK:STDOUT: !406 = !DILocation(line: 34, column: 5, scope: !402)
// CHECK:STDOUT: !407 = !DILocation(line: 32, column: 1, scope: !402)
// CHECK:STDOUT: !408 = !{!403}
// CHECK:STDOUT: !409 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.1009d1e53fe8aee6", scope: null, file: !12, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !415)
// CHECK:STDOUT: !410 = !DILocalVariable(arg: 1, scope: !409, type: !23)
// CHECK:STDOUT: !413 = !DILocation(line: 34, column: 5, scope: !409)
// CHECK:STDOUT: !414 = !DILocation(line: 32, column: 1, scope: !409)
// CHECK:STDOUT: !415 = !{!410}
// CHECK:STDOUT: !416 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.8a2301e6212711d6", scope: null, file: !12, line: 32, type: !31, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !422)
// CHECK:STDOUT: !417 = !DILocalVariable(arg: 1, scope: !416, type: !23)
// CHECK:STDOUT: !420 = !DILocation(line: 34, column: 5, scope: !416)
// CHECK:STDOUT: !421 = !DILocation(line: 32, column: 1, scope: !416)
// CHECK:STDOUT: !422 = !{!417}
// CHECK:STDOUT: !423 = distinct !DISubprogram(name: "SelfDestruct", linkageName: "_CSelfDestruct.Core.f127f30143696d51", scope: null, file: !12, line: 32, type: !102, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !429)
// CHECK:STDOUT: !424 = !DILocalVariable(arg: 1, scope: !423, type: !20)
// CHECK:STDOUT: !428 = !DILocation(line: 32, column: 1, scope: !423)
// CHECK:STDOUT: !429 = !{!424}
// CHECK:STDOUT: !430 = distinct !DISubprogram(name: "MakeGenericBig", linkageName: "_CMakeGenericBig.Main.55456748a8a2eb70", scope: null, file: !246, line: 17, type: !302, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !438)
// CHECK:STDOUT: !431 = !DILocalVariable(arg: 1, scope: !430, type: !20)
// CHECK:STDOUT: !436 = !DILocation(line: 18, column: 10, scope: !430)
// CHECK:STDOUT: !437 = !DILocation(line: 18, column: 3, scope: !430)
// CHECK:STDOUT: !438 = !{!431}
// CHECK:STDOUT: !439 = distinct !DISubprogram(name: "MakeFromSource", linkageName: "_CMakeFromSource.ImportedMaker.Main.930bcdc4607e1e7c", scope: null, file: !246, line: 23, type: !98, spFlags: DISPFlagDefinition, unit: !268, retainedNodes: !450)
// CHECK:STDOUT: !440 = !DILocalVariable(arg: 1, scope: !439, type: !23)
// CHECK:STDOUT: !441 = !DILocalVariable(arg: 2, scope: !439, type: !23)
// CHECK:STDOUT: !446 = !DILocation(line: 24, column: 18, scope: !439)
// CHECK:STDOUT: !447 = !DILocation(line: 24, column: 31, scope: !439)
// CHECK:STDOUT: !448 = !DILocation(line: 24, column: 12, scope: !439)
// CHECK:STDOUT: !449 = !DILocation(line: 24, column: 5, scope: !439)
// CHECK:STDOUT: !450 = !{!440, !441}
// CHECK:STDOUT:
