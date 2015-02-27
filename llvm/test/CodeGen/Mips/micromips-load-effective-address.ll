; RUN: llc %s -march=mipsel -mattr=micromips -filetype=asm \
; RUN: -relocation-model=pic -O3 -o - | FileCheck %s

define i32 @sum(i32* %x, i32* %y) nounwind uwtable {
entry:
  %x.addr = alloca i32*, align 8
  %y.addr = alloca i32*, align 8
  store i32* %x, i32** %x.addr, align 8
  store i32* %y, i32** %y.addr, align 8
  %0 = load i32*, i32** %x.addr, align 8
  %1 = load i32, i32* %0, align 4
  %2 = load i32*, i32** %y.addr, align 8
  %3 = load i32, i32* %2, align 4
  %add = add nsw i32 %1, %3
  ret i32 %add
}

define i32 @main() nounwind uwtable {
entry:
  %retval = alloca i32, align 4
  %x = alloca i32, align 4
  %y = alloca i32, align 4
  store i32 0, i32* %retval
  %call = call i32 @sum(i32* %x, i32* %y)
  ret i32 %call
}

; CHECK: addiu ${{[0-9]+}}, $sp, {{[0-9]+}}
; CHECK: addiu ${{[0-9]+}}, $sp, {{[0-9]+}}
