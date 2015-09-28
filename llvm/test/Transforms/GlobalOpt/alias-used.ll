; RUN: opt < %s -globalopt -S | FileCheck %s

@c = global i8 42

@i = internal global i8 42
; CHECK: @ia = internal global i8 42
@ia = internal alias i8, i8* @i

@llvm.used = appending global [3 x i8*] [i8* bitcast (void ()* @fa to i8*), i8* bitcast (void ()* @f to i8*), i8* @ca], section "llvm.metadata"
; CHECK-DAG: @llvm.used = appending global [3 x i8*] [i8* @ca, i8* bitcast (void ()* @fa to i8*), i8* bitcast (void ()* @f to i8*)], section "llvm.metadata"

@llvm.compiler.used = appending global [4 x i8*] [i8* bitcast (void ()* @fa3 to i8*), i8* bitcast (void ()* @fa to i8*), i8* @ia, i8* @i], section "llvm.metadata"
; CHECK-DAG: @llvm.compiler.used = appending global [2 x i8*] [i8* bitcast (void ()* @fa3 to i8*), i8* @ia], section "llvm.metadata"

@sameAsUsed = global [3 x i8*] [i8* bitcast (void ()* @fa to i8*), i8* bitcast (void ()* @f to i8*), i8* @ca]
; CHECK-DAG: @sameAsUsed = global [3 x i8*] [i8* bitcast (void ()* @f to i8*), i8* bitcast (void ()* @f to i8*), i8* @c]

@other = global i32* bitcast (void ()* @fa to i32*)
; CHECK-DAG: @other = global i32* bitcast (void ()* @f to i32*)

@fa = internal alias void (), void ()* @f
; CHECK: @fa = internal alias void (), void ()* @f

@fa2 = internal alias void (), void ()* @f
; CHECK-NOT: @fa2

@fa3 = internal alias void (), void ()* @f
; CHECK: @fa3

@ca = internal alias i8, i8* @c
; CHECK: @ca = internal alias i8, i8* @c

define void @f() {
  ret void
}

define i8* @g() {
  ret i8* bitcast (void ()* @fa to i8*);
}

define i8* @g2() {
  ret i8* bitcast (void ()* @fa2 to i8*);
}

define i8* @h() {
  ret i8* @ca
}
