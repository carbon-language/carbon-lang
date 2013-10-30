; RUN: llc -mtriple=x86_64-apple-darwin  -O0 < %s | FileCheck %s

@v1 = linkonce_odr global i32 32
; CHECK: .globl  _v1
; CHECK: .weak_def_can_be_hidden _v1

define i32 @f1() {
  %x = load i32 * @v1
  ret i32 %x
}

@v2 = linkonce_odr global i32 32
; CHECK: .globl  _v2
; CHECK: .weak_definition _v2

@v3 = linkonce_odr unnamed_addr global i32 32
; CHECK: .globl  _v3
; CHECK: .weak_def_can_be_hidden _v3

define i32* @f2() {
  ret i32* @v2
}

define i32* @f3() {
  ret i32* @v3
}
