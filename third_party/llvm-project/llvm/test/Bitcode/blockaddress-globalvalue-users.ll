; RUN: llvm-as %s -o %t.bc
; RUN: llvm-bcanalyzer -dump %t.bc | FileCheck %s
; RUN: llvm-dis %t.bc

; There's a curious case where blockaddress constants may refer to functions
; outside of the function they're used in. There's a special bitcode function
; code, FUNC_CODE_BLOCKADDR_USERS, used to signify that this is the case.

; The intent of this test is two-fold:
; 1. Ensure we do not produce BLOCKADDR_USERS bitcode function code on the first
;    fn, @repro, by accident, when @fun and @fun2 use a global value, @foo,
;    which is initialized to @repro's blockaddress constants.
; 2. Ensure we can round-trip serializing+desearlizing such case.

; CHECK: <FUNCTION_BLOCK
; CHECK-NOT: <BLOCKADDR_USERS

@foo = global i8* blockaddress(@repro, %label)

define void @repro() {
  br label %label

label:
  call void @fun()
  ret void
}

define void @fun() noinline {
  call void @f(i8** @foo)
  ret void
}

define void @fun2() noinline {
  call void @f(i8** @foo)
  ret void
}

declare void @f(i8**)