; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=att | grep {(\$bar)} | count 1
; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=att | grep {(\$qux)} | count 1
; RUN: llvm-as < %s | llc -march=x86 -x86-asm-syntax=att | grep {(\$hen)} | count 1
; PR1339

@"$bar" = global i32 zeroinitializer
@"$qux" = external global i32

define i32 @"$foo"() {
  %m = load i32* @"$bar"
  %n = load i32* @"$qux"
  %t = add i32 %m, %n
  %u = call i32 @"$hen"(i32 %t)
  ret i32 %u
}

declare i32 @"$hen"(i32 %a)
