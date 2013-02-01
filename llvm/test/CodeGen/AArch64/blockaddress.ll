; RUN: llc -mtriple=aarch64-none-linux-gnu -verify-machineinstrs < %s | FileCheck %s

@addr = global i8* null

define void @test_blockaddress() {
; CHECK: test_blockaddress:
  store volatile i8* blockaddress(@test_blockaddress, %block), i8** @addr
  %val = load volatile i8** @addr
  indirectbr i8* %val, [label %block]
; CHECK: adrp [[DEST_HI:x[0-9]+]], [[DEST_LBL:.Ltmp[0-9]+]]
; CHECK: add [[DEST:x[0-9]+]], [[DEST_HI]], #:lo12:[[DEST_LBL]]
; CHECK: str [[DEST]],
; CHECK: ldr [[NEWDEST:x[0-9]+]]
; CHECK: br [[NEWDEST]]

block:
  ret void
}
