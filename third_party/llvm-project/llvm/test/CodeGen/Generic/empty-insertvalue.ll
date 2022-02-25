; RUN: llc < %s

define void @f() {
entry:
  %0 = insertvalue { [0 x { i8*, i8* }], [0 x { i8*, i64 }] } undef, [0 x { i8*, i8* }] undef, 0
  ret void
}
