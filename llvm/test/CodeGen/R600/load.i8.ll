;RUN: llc < %s -march=r600 -mcpu=redwood | FileCheck %s

;CHECK: VTX_READ_8 T{{[0-9]+\.X, T[0-9]+\.X}}

define void @test(i32 addrspace(1)* %out, i8 addrspace(1)* %in) {
  %1 = load i8 addrspace(1)* %in
  %2 = zext i8 %1 to i32
  store i32 %2, i32 addrspace(1)* %out
  ret void
}
