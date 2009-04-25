; RUN: llvm-as < %s | llvm-dis | llvm-as | llvm-dis

;; Verify that i16 indices work.
@x = external global {i32, i32}
@y = global i32* getelementptr ({i32, i32}* @x, i16 42, i32 0)

; see if i92 indices work too.
define i32 *@test({i32, i32}* %t, i92 %n) {
  %B = getelementptr {i32, i32}* %t, i92 %n, i32 0
  ret i32* %B
}

