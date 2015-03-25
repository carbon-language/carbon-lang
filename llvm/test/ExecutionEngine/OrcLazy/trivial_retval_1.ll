; RUN: lli -jit-kind=orc-lazy %s; [ $? -eq 30 ]
define i32 @baz() {
entry:
  ret i32 2
}

define i32 @bar() {
entry:
  %call = call i32 @baz()
  %mul = mul nsw i32 3, %call
  ret i32 %mul
}

define i32 @foo() {
entry:
  %call = call i32 @bar()
  %mul = mul nsw i32 5, %call
  ret i32 %mul
}

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %call = call i32 @foo()
  ret i32 %call
}
