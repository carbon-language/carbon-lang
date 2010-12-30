; RUN: llc < %s -march=ptx | FileCheck %s

;CHECK: .extern .global .s32 array[];
@array = external global [10 x i32]

;CHECK: .extern .const .s32 array_constant[];
@array_constant = external addrspace(1) constant [10 x i32]

;CHECK: .extern .local .s32 array_local[];
@array_local = external addrspace(2) global [10 x i32]

;CHECK: .extern .shared .s32 array_shared[];
@array_shared = external addrspace(4) global [10 x i32]

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

define ptx_device i32 @t4_global() {
entry:
;CHECK: ld.global.s32 r0, [array];
  %i = getelementptr [10 x i32]* @array, i32 0, i32 0
  %x = load i32* %i
  ret i32 %x
}

define ptx_device i32 @t4_const() {
entry:
;CHECK: ld.const.s32 r0, [array_constant];
  %i = getelementptr [10 x i32] addrspace(1)* @array_constant, i32 0, i32 0
  %x = load i32 addrspace(1)* %i
  ret i32 %x
}

define ptx_device i32 @t4_local() {
entry:
;CHECK: ld.local.s32 r0, [array_local];
  %i = getelementptr [10 x i32] addrspace(2)* @array_local, i32 0, i32 0
  %x = load i32 addrspace(2)* %i
  ret i32 %x
}

define ptx_device i32 @t4_shared() {
entry:
;CHECK: ld.shared.s32 r0, [array_shared];
  %i = getelementptr [10 x i32] addrspace(4)* @array_shared, i32 0, i32 0
  %x = load i32 addrspace(4)* %i
  ret i32 %x
}

define ptx_device i32 @t5() {
entry:
;CHECK: ld.global.s32 r0, [array+4];
  %i = getelementptr [10 x i32]* @array, i32 0, i32 1
  %x = load i32* %i
  ret i32 %x
}
