; RUN: lli -jit-kind=orc-lazy %s
;
; Test handling of global aliases for function and variables.

@x = global i32 42, align 4
@y = alias i32, i32* @x

define i32 @foo() {
entry:
  %0 = load i32, i32* @y, align 4
  ret i32 %0
}

@bar = alias i32(), i32()* @foo

define i32 @main(i32 %argc, i8** %argv) {
entry:
  %0 = call i32() @bar()
  %1 = sub i32 %0, 42
  ret i32 %1
}
