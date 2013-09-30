; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 %t1

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".text"
module asm ".align 16, 0x90"
module asm ".type PR14512, @function"
module asm "PR14512:.cfi_startproc"
module asm "ret"
module asm ".cfi_endproc"

declare void @PR14512()

define i32 @main(i32 %argc, i8** %argv) {
  call void @PR14512()
  ret i32 0
}
