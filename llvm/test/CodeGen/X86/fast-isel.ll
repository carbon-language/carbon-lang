; RUN: llvm-as < %s | llc -fast-isel | grep add | count 1

; This tests very minimal fast-isel functionality.

define i32 @foo(i32* %p, i32* %q) {
entry:
  %r = load i32* %p
  %s = load i32* %q
  br label %fast

fast:
  %t = add i32 %r, %s
  br label %exit

exit:
  ret i32 %t
}
