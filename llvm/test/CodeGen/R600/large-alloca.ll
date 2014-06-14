; XFAIL: *
; RUN: llc -march=r600 -mcpu=SI < %s

define void @large_alloca(i32 addrspace(1)* %out, i32 %x) nounwind {
  %large = alloca [256 x i32], align 4
  %gep = getelementptr [256 x i32]* %large, i32 0, i32 255
  store i32 %x, i32* %gep
  ret void
}

