; RUN: opt < %s -instcombine -S

%myStruct = type { float, [3 x float], [4 x float], i32 }

; make sure that we are not crashing when creating an illegal type
define void @func(%myStruct addrspace(1)* nocapture %p) nounwind {
ST:
  %A = getelementptr inbounds %myStruct addrspace(1)* %p, i64 0
  %B = bitcast %myStruct addrspace(1)* %A to %myStruct*
  %C = getelementptr inbounds %myStruct* %B, i32 0, i32 1
  %D = getelementptr inbounds [3 x float]* %C, i32 0, i32 2
  %E = load float* %D, align 4
  %F = fsub float %E, undef
  ret void
}

