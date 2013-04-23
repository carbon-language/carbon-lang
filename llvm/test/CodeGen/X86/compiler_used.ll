; RUN: llc < %s -mtriple=i386-apple-darwin9 | FileCheck %s

@X = internal global i8 4
@Y = internal global i32 123
@Z = internal global i8 4

@llvm.used = appending global [1 x i8*] [ i8* @Z ], section "llvm.metadata"
@llvm.compiler_used = appending global [2 x i8*] [ i8* @X, i8* bitcast (i32* @Y to i8*)], section "llvm.metadata"

; CHECK-NOT: .no_dead_strip
; CHECK: .no_dead_strip	_Z
; CHECK-NOT: .no_dead_strip
