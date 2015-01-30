; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 %t1
; RUN: llvm-nm %t2 | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"

module asm ".symver io_cancel_0_4,io_cancel@@LIBAIO_0.4"

; Even without -exported-symbol, io_cancel_0_4 should be noticed by LTOModule's
; RecordStreamer, so it shouldn't get eliminated. However, the object file will
; contain the aliased symver as well as the original.
define i32 @io_cancel_0_4() {
; CHECK: io_cancel@@LIBAIO_0.4
; CHECK: io_cancel_0_4
  ret i32 0
}
