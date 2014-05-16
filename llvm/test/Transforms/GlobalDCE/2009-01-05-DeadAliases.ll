; RUN: opt < %s -globaldce -S > %t
; RUN: FileCheck %s < %t
; RUN: FileCheck --check-prefix=DEAD %s < %t

@A = global i32 0
; CHECK: @A = global i32 0

@D = alias internal i32* @A
; DEAD-NOT: @D

@L1 = alias i32* @A
; CHECK: @L1 = alias i32* @A

@L2 = alias internal i32* @A
; DEAD-NOT: @L2

@L3 = alias i32* @A
; CHECK: @L3 = alias i32* @A
