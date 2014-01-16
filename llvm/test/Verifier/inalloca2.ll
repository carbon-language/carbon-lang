; This used to be invalid, but now it's valid.  Ensure the verifier
; doesn't reject it.
; RUN: llvm-as %s -o /dev/null

declare void @doit(i64* inalloca %a)

define void @a() {
entry:
  %a = alloca [2 x i32]
  %b = bitcast [2 x i32]* %a to i64*
  call void @doit(i64* inalloca %b)
  ret void
}

define void @b() {
entry:
  %a = alloca i64
  call void @doit(i64* inalloca %a)
  call void @doit(i64* inalloca %a)
  ret void
}
