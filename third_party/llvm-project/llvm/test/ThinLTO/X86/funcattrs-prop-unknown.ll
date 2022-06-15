; Unknown (e.g. indirect) calls returns conservative results from function propagation
; RUN: opt -thinlto-bc %s -thin-link-bitcode-file=%t1.thinlink.bc -o %t1.bc
; RUN: llvm-lto2 run -disable-thinlto-funcattrs=0 %t1.bc -o %t.o -save-temps \
; RUN:    -r %t1.bc,indirect,px -r %t1.bc,inlineasm,px -r %t1.bc,selectcallee,px -r %t1.bc,f, -r %t1.bc,g, -r %t1.bc,global,
; RUN: llvm-dis -o - %t.o.1.3.import.bc | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-NOT: ; Function Attrs:
; CHECK: define i32 @indirect(ptr nocapture %0) {
define i32 @indirect(i32 ()* nocapture) {
  %2 = tail call i32 %0()
  ret i32 %2
}

; CHECK-NOT: ; Function Attrs:
; CHECK: define ptr @inlineasm() {
define i8* @inlineasm() {
entry:
  %0 = tail call i8* asm sideeffect "lea ff_h264_cabac_tables(%rip), $0", "=&r,~{dirflag},~{fpsr},~{flags}"()
  ret i8* %0
}

; CHECK-NOT: ; Function Attrs:
; CHECK: define void @selectcallee() {
define void @selectcallee() {
    ; Test calls that aren't handled either as direct or indirect.
    call void select (i1 icmp eq (i32* @global, i32* null), void ()* @f, void ()* @g)()
    ret void
}

declare void @f()
declare void @g()
@global = extern_weak global i32
