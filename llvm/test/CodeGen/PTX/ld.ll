; RUN: llc < %s -march=ptx | FileCheck %s

;CHECK: .extern .global .s32 array[];
@array = external global [10 x i32]

define ptx_device i32 @t1(i32* %p) {
entry:
;CHECK: ld.global.s32 r0, [r1];
  %x = load i32* %p
  ret i32 %x
}

define ptx_device i32 @t2(i32* %p) {
entry:
;CHECK: ld.global.s32 r0, [r1+4];
  %i = getelementptr i32* %p, i32 1
  %x = load i32* %i
  ret i32 %x
}

define ptx_device i32 @t3(i32* %p, i32 %q) {
entry:
;CHECK: shl.b32 r0, r2, 2;
;CHECK: ld.global.s32 r0, [r1+r0];
  %i = getelementptr i32* %p, i32 %q
  %x = load i32* %i
  ret i32 %x
}

define ptx_device i32 @t4() {
entry:
;CHECK: ld.global.s32 r0, [array];
  %i = getelementptr [10 x i32]* @array, i32 0, i32 0
  %x = load i32* %i
  ret i32 %x
}

define ptx_device i32 @t5() {
entry:
;CHECK: ld.global.s32 r0, [array+4];
  %i = getelementptr [10 x i32]* @array, i32 0, i32 1
  %x = load i32* %i
  ret i32 %x
}
