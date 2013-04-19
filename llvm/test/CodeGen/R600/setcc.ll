;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s
;CHECK: SETE_INT T{{[0-9]+\.[XYZW], T[0-9]+\.[XYZW], T[0-9]+\.[XYZW]}}

define void @test(<4 x i32> addrspace(1)* %out, <4 x i32> addrspace(1)* %in) {
  %b_ptr = getelementptr <4 x i32> addrspace(1)* %in, i32 1
  %a = load <4 x i32> addrspace(1) * %in
  %b = load <4 x i32> addrspace(1) * %b_ptr
  %result = icmp eq <4 x i32> %a, %b
  %sext = sext <4 x i1> %result to <4 x i32>
  store <4 x i32> %sext, <4 x i32> addrspace(1)* %out
  ret void
}
