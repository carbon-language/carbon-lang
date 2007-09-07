; RUN: llvm-as < %s | llc -march=x86-64 | grep xor | count 4
; RUN: llvm-as < %s | llc -march=x86-64 -stats -info-output-file - | grep asm-printer | grep 12

declare void @bar(double %x)
declare void @barf(float %x)

define double @foo() {
  call void @bar(double 0.0)
  ret double 0.0
}
define float @foof() {
  call void @barf(float 0.0)
  ret float 0.0
}
