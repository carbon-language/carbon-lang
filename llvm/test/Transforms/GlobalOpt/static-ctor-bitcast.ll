; RUN: opt -S -globalopt < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

%0 = type { i8*, i8* }
%1 = type { i8*, i8*, i32, i32, i8*, i64 }
%2 = type { i32, void ()* }
%struct.A = type { i32 }
%struct.B = type { i32 (...)**, i8*, [4 x i8] }

@y = global i8 0, align 1
@x = global %struct.B zeroinitializer, align 8
@_ZTV1B = weak_odr unnamed_addr constant [3 x i8*] [i8* inttoptr (i64 16 to i8*), i8* null, i8* bitcast (%1* @_ZTI1B to i8*)]
@_ZTVN10__cxxabiv121__vmi_class_type_infoE = external global i8*
@_ZTS1B = weak_odr constant [3 x i8] c"1B\00"
@_ZTVN10__cxxabiv117__class_type_infoE = external global i8*
@_ZTS1A = weak_odr constant [3 x i8] c"1A\00"
@_ZTI1A = weak_odr unnamed_addr constant %0 { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv117__class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8]* @_ZTS1A, i32 0, i32 0) }
@_ZTI1B = weak_odr unnamed_addr constant %1 { i8* bitcast (i8** getelementptr inbounds (i8** @_ZTVN10__cxxabiv121__vmi_class_type_infoE, i64 2) to i8*), i8* getelementptr inbounds ([3 x i8]* @_ZTS1B, i32 0, i32 0), i32 0, i32 1, i8* bitcast (%0* @_ZTI1A to i8*), i64 -6141 }
@_ZTT1B = weak_odr unnamed_addr constant [1 x i8*] [i8* bitcast (i8** getelementptr inbounds ([3 x i8*]* @_ZTV1B, i64 1, i64 0) to i8*)]
@llvm.global_ctors = appending global [1 x %2] [%2 { i32 65535, void ()* @_GLOBAL__I_a }]

; CHECK-NOT: __cxx_global_var_init
define internal void @__cxx_global_var_init() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  call void @_ZN1BC1Ev(%struct.B* @x)
  ret void
}

; CHECK-NOT: _ZN1BC1Ev
define linkonce_odr unnamed_addr void @_ZN1BC1Ev(%struct.B* %this) inlinehint ssp align 2 {
entry:
  %0 = bitcast %struct.B* %this to i8*
  %1 = getelementptr inbounds i8* %0, i64 16
  %2 = bitcast i8* %1 to %struct.A*
  call void @_ZN1AC2Ev(%struct.A* %2)
  %3 = bitcast %struct.B* %this to i8***
  store i8** getelementptr inbounds ([3 x i8*]* @_ZTV1B, i64 1, i64 0), i8*** %3
  ret void
}

; CHECK-NOT: _ZN1AC2Ev
define linkonce_odr unnamed_addr void @_ZN1AC2Ev(%struct.A* %this) nounwind ssp align 2 {
entry:
  %0 = ptrtoint %struct.A* %this to i64
  %sub = sub i64 %0, ptrtoint (%struct.B* @x to i64)
  %div = udiv i64 %sub, 8
  %conv = trunc i64 %div to i8
  store i8 %conv, i8* @y, align 1
  ret void
}

; CHECK-NOT: _GLOBAL__I_a
define internal void @_GLOBAL__I_a() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  call void @__cxx_global_var_init()
  ret void
}
