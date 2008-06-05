; RUN: llvm-as < %s | llvm-dis

define { i32, i32 } @foo() {
  %res = insertvalue { i32, i32 } undef, i32 0, 0
  %res2 = insertvalue { i32, i32 } %res, i32 1, 1
  ret { i32, i32 } %res2
}

define [ 2 x i32 ] @bar() {
  %res = insertvalue [ 2 x i32 ] undef, i32 0, 0
  %res2 = insertvalue [ 2 x i32 ] %res, i32 1, 1
  ret [ 2 x i32 ] %res2
}

define i32 @main() {
  %a = call { i32, i32 }()* @foo ()
  %b = call [ 2 x i32 ]()* @bar ()
  %a.0 = extractvalue { i32, i32 } %a, 0
  %b.1 = extractvalue [ 2 x i32 ] %b, 1
  %r = add i32 %a.0, %b.1
  ret i32 %r
}
