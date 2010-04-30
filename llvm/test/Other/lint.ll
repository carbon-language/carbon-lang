; RUN: opt -lint -disable-output < %s |& FileCheck %s
target datalayout = "e-p:64:64:64"

declare fastcc void @bar()

@CG = constant i32 7

define i32 @foo() noreturn {
; CHECK: Caller and callee calling convention differ
  call void @bar()
; CHECK: Null pointer dereference
  store i32 0, i32* null
; CHECK: Null pointer dereference
  %t = load i32* null
; CHECK: Undef pointer dereference
  store i32 0, i32* undef
; CHECK: Undef pointer dereference
  %u = load i32* undef
; CHECK: Memory reference address is misaligned
  %x = inttoptr i32 1 to i32*
  load i32* %x, align 4
; CHECK: Division by zero
  %sd = sdiv i32 2, 0
; CHECK: Division by zero
  %ud = udiv i32 2, 0
; CHECK: Division by zero
  %sr = srem i32 2, 0
; CHECK: Division by zero
  %ur = urem i32 2, 0
; CHECK: extractelement index out of range
  %ee = extractelement <4 x i32> zeroinitializer, i32 4
; CHECK: insertelement index out of range
  %ie = insertelement <4 x i32> zeroinitializer, i32 0, i32 4
; CHECK: Shift count out of range
  %r = lshr i32 0, 32
; CHECK: Shift count out of range
  %q = ashr i32 0, 32
; CHECK: Shift count out of range
  %l = shl i32 0, 32
; CHECK: xor(undef, undef)
  %xx = xor i32 undef, undef
; CHECK: sub(undef, undef)
  %xs = sub i32 undef, undef

; CHECK: Write to read-only memory
  store i32 8, i32* @CG
; CHECK: Write to text section
  store i32 8, i32* bitcast (i32()* @foo to i32*)
; CHECK: Load from block address
  %lb = load i32* bitcast (i8* blockaddress(@foo, %next) to i32*)
; CHECK: Call to block address
  call void()* bitcast (i8* blockaddress(@foo, %next) to void()*)()

  br label %next

next:
; CHECK: Static alloca outside of entry block
  %a = alloca i32
; CHECK: Return statement in function with noreturn attribute
  ret i32 0

foo:
  %z = add i32 0, 0
; CHECK: unreachable immediately preceded by instruction without side effects
  unreachable
}

; CHECK: Unnamed function with non-local linkage
define void @0() nounwind {
  ret void
}

; CHECK: va_start called in a non-varargs function
declare void @llvm.va_start(i8*)
define void @not_vararg(i8* %p) nounwind {
  call void @llvm.va_start(i8* %p)
  ret void
}

define void @use_indbr() {
  indirectbr i8* bitcast (i32()* @foo to i8*), [label %block]
block:
  unreachable
}
