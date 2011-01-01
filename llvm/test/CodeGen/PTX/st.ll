; RUN: llc < %s -march=ptx | FileCheck %s

;CHECK: .extern .global .s32 array[];
@array = external global [10 x i32]

;CHECK: .extern .const .s32 array_constant[];
@array_constant = external addrspace(1) constant [10 x i32]

;CHECK: .extern .local .s32 array_local[];
@array_local = external addrspace(2) global [10 x i32]

;CHECK: .extern .shared .s32 array_shared[];
@array_shared = external addrspace(4) global [10 x i32]

define ptx_device void @t1(i32* %p, i32 %x) {
entry:
;CHECK: st.global.s32 [r1], r2;
  store i32 %x, i32* %p
  ret void
}

define ptx_device void @t2(i32* %p, i32 %x) {
entry:
;CHECK: st.global.s32 [r1+4], r2;
  %i = getelementptr i32* %p, i32 1
  store i32 %x, i32* %i
  ret void
}

define ptx_device void @t3(i32* %p, i32 %q, i32 %x) {
;CHECK: .reg .s32 r0;
entry:
;CHECK: shl.b32 r0, r2, 2;
;CHECK: add.s32 r0, r1, r0;
;CHECK: st.global.s32 [r0], r3;
  %i = getelementptr i32* %p, i32 %q
  store i32 %x, i32* %i
  ret void
}

define ptx_device void @t4_global(i32 %x) {
entry:
;CHECK: st.global.s32 [array], r1;
  %i = getelementptr [10 x i32]* @array, i32 0, i32 0
  store i32 %x, i32* %i
  ret void
}

define ptx_device void @t4_local(i32 %x) {
entry:
;CHECK: st.local.s32 [array_local], r1;
  %i = getelementptr [10 x i32] addrspace(2)* @array_local, i32 0, i32 0
  store i32 %x, i32 addrspace(2)* %i
  ret void
}

define ptx_device void @t4_shared(i32 %x) {
entry:
;CHECK: st.shared.s32 [array_shared], r1;
  %i = getelementptr [10 x i32] addrspace(4)* @array_shared, i32 0, i32 0
  store i32 %x, i32 addrspace(4)* %i
  ret void
}

define ptx_device void @t5(i32 %x) {
entry:
;CHECK: st.global.s32 [array+4], r1;
  %i = getelementptr [10 x i32]* @array, i32 0, i32 1
  store i32 %x, i32* %i
  ret void
}
