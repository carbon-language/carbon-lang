; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:32:32"
target triple = "i686-apple-darwin8.7.2"

@x = weak global i32 0
; CHECK: .no_dead_strip	_x

@llvm.used = appending global [1 x i8*] [ i8* bitcast (i32* @x to i8*) ]
