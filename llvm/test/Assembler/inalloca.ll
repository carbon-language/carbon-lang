; RUN: llvm-as %s -o /dev/null

define void @a() {
entry:
  %0 = alloca i32, inalloca
  %1 = alloca [2 x i32], inalloca
  %2 = alloca i32, inalloca, i32 2
  %3 = alloca i32, inalloca, i32 2, align 16
  %4 = alloca i32, inalloca, i32 2, align 16, !foo !0
  %5 = alloca i32, i32 2, align 16, !foo !0
  %6 = alloca i32, i32 2, align 16
  ret void
}

!0 = metadata !{i32 662302, null}
!foo = !{ !0 }
