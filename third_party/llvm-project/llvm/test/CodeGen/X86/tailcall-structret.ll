; RUN: llc < %s -mtriple=i686-unknown-linux -tailcallopt | FileCheck %s
define fastcc { { i8*, i8* }*, i8*} @init({ { i8*, i8* }*, i8*}, i32) {
entry:
      %2 = tail call fastcc { { i8*, i8* }*, i8* } @init({ { i8*, i8*}*, i8*} %0, i32 %1)
      ret { { i8*, i8* }*, i8*} %2
; CHECK: jmp init
}
