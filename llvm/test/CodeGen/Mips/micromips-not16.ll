; RUN: llc -march=mipsel -mcpu=mips32r2 -mattr=+micromips \
; RUN:   -relocation-model=pic -O3 < %s | FileCheck %s

define i32 @main() {
entry:
  %retval = alloca i32, align 4
  %x = alloca i64, align 8
  store i32 0, i32* %retval
  %0 = load i64, i64* %x, align 8
  %cmp = icmp ne i64 %0, 9223372036854775807
  br i1 %cmp, label %if.then, label %if.end

if.then:
  store i32 1, i32* %retval
  br label %return

if.end:
  store i32 0, i32* %retval
  br label %return

return:
  %1 = load i32, i32* %retval
  ret i32 %1
}

; CHECK: not16
