; RUN: opt %s -mtriple amdgcn-- -amdgpu-use-legacy-divergence-analysis -passes='print<divergence>' 2>&1 -disable-output | FileCheck %s

; CHECK: DIVERGENT:  %tmp = cmpxchg volatile
define amdgpu_kernel void @unreachable_loop(i32 %tidx) #0 {
entry:
  unreachable

unreachable_loop:                                        ; preds = %do.body.i, %if.then11
  %tmp = cmpxchg volatile i32 addrspace(1)* null, i32 0, i32 0 seq_cst seq_cst
  %cmp.i = extractvalue { i32, i1 } %tmp, 1
  br i1 %cmp.i, label %unreachable_loop, label %end

end:                                      ; preds = %do.body.i51, %atomicAdd_g_f.exit
  unreachable
}

attributes #0 = { norecurse nounwind }
