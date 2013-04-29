; RUN: llc < %s -mtriple x86_64-apple-macosx10 | FileCheck %s

; CHECK: .section __DATA,__thread_init,thread_local_init_function_pointers
; CHECK: .align 3
; CHECK: .quad ___tls_init

%struct.A = type { i8 }
%struct.B = type { i32 }

@i = thread_local global i32 37, align 4
@a = thread_local global %struct.A zeroinitializer, align 1
@b = thread_local global %struct.B zeroinitializer, align 4
@z = global %struct.A zeroinitializer, align 1
@y = global %struct.B zeroinitializer, align 4
@__tls_guard = internal thread_local unnamed_addr global i1 false
@llvm.global_ctors = appending global [1 x { i32, void ()* }] [{ i32, void ()* } { i32 65535, void ()* @_GLOBAL__I_a }]
@llvm.tls_init_funcs = appending global [1 x void ()*] [void ()* @__tls_init]

@_ZTH1i = alias void ()* @__tls_init
@_ZTH1a = alias void ()* @__tls_init
@_ZTH1b = alias void ()* @__tls_init

declare void @_ZN1AC1Ev(%struct.A*)

declare void @_ZN1BC1Ei(%struct.B*, i32)

define internal void @_GLOBAL__I_a() section "__TEXT,__StaticInit,regular,pure_instructions" {
entry:
  tail call void @_ZN1AC1Ev(%struct.A* @z)
  tail call void @_ZN1BC1Ei(%struct.B* @y, i32 42)
  ret void
}

define internal void @__tls_init() {
entry:
  %.b = load i1* @__tls_guard, align 1
  store i1 true, i1* @__tls_guard, align 1
  br i1 %.b, label %exit, label %init

init:                                             ; preds = %entry
  tail call void @_ZN1AC1Ev(%struct.A* @a)
  tail call void @_ZN1BC1Ei(%struct.B* @b, i32 927)
  br label %exit

exit:                                             ; preds = %entry, %init
  ret void
}

define weak_odr hidden i32* @_ZTW1i() {
  %.b.i = load i1* @__tls_guard, align 1
  store i1 true, i1* @__tls_guard, align 1
  br i1 %.b.i, label %__tls_init.exit, label %init.i

init.i:                                           ; preds = %0
  tail call void @_ZN1AC1Ev(%struct.A* @a)
  tail call void @_ZN1BC1Ei(%struct.B* @b, i32 927)
  br label %__tls_init.exit

__tls_init.exit:                                  ; preds = %0, %init.i
  ret i32* @i
}

define weak_odr hidden %struct.A* @_ZTW1a() {
  %.b.i = load i1* @__tls_guard, align 1
  store i1 true, i1* @__tls_guard, align 1
  br i1 %.b.i, label %__tls_init.exit, label %init.i

init.i:                                           ; preds = %0
  tail call void @_ZN1AC1Ev(%struct.A* @a)
  tail call void @_ZN1BC1Ei(%struct.B* @b, i32 927)
  br label %__tls_init.exit

__tls_init.exit:                                  ; preds = %0, %init.i
  ret %struct.A* @a
}

define weak_odr hidden %struct.B* @_ZTW1b() {
  %.b.i = load i1* @__tls_guard, align 1
  store i1 true, i1* @__tls_guard, align 1
  br i1 %.b.i, label %__tls_init.exit, label %init.i

init.i:                                           ; preds = %0
  tail call void @_ZN1AC1Ev(%struct.A* @a)
  tail call void @_ZN1BC1Ei(%struct.B* @b, i32 927)
  br label %__tls_init.exit

__tls_init.exit:                                  ; preds = %0, %init.i
  ret %struct.B* @b
}
