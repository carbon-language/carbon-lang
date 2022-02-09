; RUN: llc %s -march=mipsel -mattr=micromips -filetype=asm \
; RUN: -relocation-model=pic -O3 -o - | FileCheck %s

; Function Attrs: noinline nounwind
define void @bar(i32* %p) #0 {
entry:
  %p.addr = alloca i32*, align 4
  store i32* %p, i32** %p.addr, align 4
  %0 = load i32*, i32** %p.addr, align 4
  %1 = load i32, i32* %0, align 4
  %add = add nsw i32 7, %1
  %2 = load i32*, i32** %p.addr, align 4
  store i32 %add, i32* %2, align 4
  %3 = load i32*, i32** %p.addr, align 4
  %add.ptr = getelementptr inbounds i32, i32* %3, i32 1
  %4 = load i32, i32* %add.ptr, align 4
  %add1 = add nsw i32 7, %4
  %5 = load i32*, i32** %p.addr, align 4
  %add.ptr2 = getelementptr inbounds i32, i32* %5, i32 1
  store i32 %add1, i32* %add.ptr2, align 4
  ret void
}

; CHECK: lw16 ${{[0-9]+}}, 0($4)
; CHECK: sw16 ${{[0-9]+}}, 0($4)
; CHECK: lw16 ${{[0-9]+}}, 4(${{[0-9]+}})
; CHECK: sw16 ${{[0-9]+}}, 4(${{[0-9]+}})
