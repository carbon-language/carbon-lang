; RUN: llvm-as < %s | llvm-dis > %t.orig
; RUN: llvm-as < %s | llvm-c-test --echo > %t.echo
; RUN: diff %t.orig %t.echo

declare void @decl()

; TODO: label, struct and metadata types
define void @types() {
  %1 = alloca half
  %2 = alloca float
  %3 = alloca double
  %4 = alloca x86_fp80
  %5 = alloca fp128
  %6 = alloca ppc_fp128
  %7 = alloca i7
  %8 = alloca void (i1)*
  %9 = alloca [3 x i22]
  %10 = alloca i328 addrspace(5)*
  %11 = alloca <5 x i23*>
  %12 = alloca x86_mmx
  ret void
}

define i32 @add(i32 %a, i32 %b) {
  %1 = add i32 %a, %b
  ret i32 %1
}

define i32 @call() {
  %1 = call i32 @add(i32 23, i32 19)
  ret i32 %1
}
