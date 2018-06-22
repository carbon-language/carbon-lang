; Checks if bitcasted call expression can be evaluated
; Given call expresion:
;   %struct.Foo* bitcast (%struct.Bar* ()* @_Z6getFoov to %struct.Foo* ()*)()
; We evaluate call to function _Z6getFoov and then cast the result to %structFoo*

; RUN: opt -globalopt -instcombine %s -S -o - | FileCheck %s

; CHECK:      i32 @main()
; CHECK-NEXT:   %1 = load i64, i64* inttoptr (i64 32 to i64*), align 32
; CHECK-NEXT:   %2 = trunc i64 %1 to i32
; CHECK-NEXT:   ret i32 %2
; CHECK-NOT: _GLOBAL__sub_I_main

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

%struct.Bar = type { i64, i64 }
%struct.S = type { %struct.Foo* }
%struct.Foo = type { i64, i64 }
%struct.Baz = type { i64, i64, %struct.Bar }

@instance = internal local_unnamed_addr global %struct.S zeroinitializer, align 8
@llvm.global_ctors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @_GLOBAL__sub_I_main.cpp, i8* null }]
@gBaz = available_externally dso_local local_unnamed_addr global %struct.Baz* null, align 8
@gFoo = available_externally dso_local local_unnamed_addr global %struct.Bar* null, align 8

; Function Attrs: norecurse nounwind readonly uwtable
define dso_local i32 @main() local_unnamed_addr {
  %1 = load %struct.Foo*, %struct.Foo** getelementptr inbounds (%struct.S, %struct.S* @instance, i64 0, i32 0), align 8
  %2 = getelementptr inbounds %struct.Foo, %struct.Foo* %1, i64 0, i32 0
  %3 = load i64, i64* %2, align 8
  %4 = trunc i64 %3 to i32
  ret i32 %4
}

; Function Attrs: uwtable
define internal void @_GLOBAL__sub_I_main.cpp() section ".text.startup" {
  %1 = tail call %struct.Foo* bitcast (%struct.Bar* ()* @_Z6getFoov to %struct.Foo* ()*)()
  %2 = getelementptr inbounds %struct.Foo, %struct.Foo* %1, i64 1
  store %struct.Foo* %2, %struct.Foo** getelementptr inbounds (%struct.S, %struct.S* @instance, i64 0, i32 0), align 8
  ret void
}

; Function Attrs: norecurse nounwind readonly uwtable
define available_externally dso_local %struct.Bar* @_Z6getFoov() local_unnamed_addr {
  %1 = load %struct.Bar*, %struct.Bar** @gFoo, align 8
  %2 = icmp eq %struct.Bar* %1, null
  %3 = load %struct.Baz*, %struct.Baz** @gBaz, align 8
  %4 = getelementptr inbounds %struct.Baz, %struct.Baz* %3, i64 0, i32 2
  %5 = select i1 %2, %struct.Bar* %4, %struct.Bar* %1
  ret %struct.Bar* %5
}
