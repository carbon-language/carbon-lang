; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=hawaii -O2 -tail-dup-size=1000 -tail-dup-placement-threshold=1000 -enable-tail-merge=0 < %s | FileCheck -enable-var-scope -check-prefix=GCN %s

; Need to to trigger tail duplication this during
; MachineBlockPlacement, since calls aren't tail duplicated pre-RA.

declare void @nonconvergent_func() #0
declare void @convergent_func() #1
declare void @llvm.amdgcn.s.barrier() #1
declare void @llvm.amdgcn.ds.gws.init(i32, i32) #2
declare void @llvm.amdgcn.ds.gws.barrier(i32, i32) #2
declare void @llvm.amdgcn.ds.gws.sema.release.all(i32 %offset) #2

; barrier shouldn't be duplicated.

; GCN-LABEL: {{^}}taildup_barrier:
; GCN: s_barrier
; GCN-NOT: s_barrier
define void @taildup_barrier(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i1 %cond) #0 {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, i32 addrspace(1)* %a
  br label %call

bb2:
  store i32 1, i32 addrspace(1)* %a
  br label %call

call:
  call void @llvm.amdgcn.s.barrier()
  br label %ret

ret:
  ret void
}

; GCN-LABEL: {{^}}taildup_convergent_call:
; GCN: s_swappc_b64
; GCN-NOT: s_swappc_b64
define void @taildup_convergent_call(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i1 %cond) #1 {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, i32 addrspace(1)* %a
  br label %call

bb2:
  store i32 1, i32 addrspace(1)* %a
  br label %call

call:
  call void @convergent_func()
  br label %ret

ret:
  ret void
}

; TODO: Currently there is only one convergent call pseudo, but this
; theoretically could use a nonconvergent variant.
; GCN-LABEL: {{^}}taildup_nonconvergent_call:
; GCN: s_swappc_b64
; GCN-NOT: s_swappc_b64
define void @taildup_nonconvergent_call(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i1 %cond) #1 {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, i32 addrspace(1)* %a
  br label %call

bb2:
  store i32 1, i32 addrspace(1)* %a
  br label %call

call:
  call void @nonconvergent_func()
  br label %ret

ret:
  ret void
}

; GCN-LABEL: {{^}}taildup_convergent_tailcall:
; GCN: s_setpc_b64
; GCN-NOT: s_setpc_b64
define void @taildup_convergent_tailcall(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i1 %cond) #1 {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, i32 addrspace(1)* %a
  br label %call

bb2:
  store i32 1, i32 addrspace(1)* %a
  br label %call

call:
  tail call void @convergent_func()
  ret void
}

; GCN-LABEL: {{^}}taildup_gws_init:
; GCN: ds_gws_init
; GCN-NOT: ds_gws_init
define amdgpu_kernel void @taildup_gws_init(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i1 %cond, i32 %val, i32 %offset) #0 {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, i32 addrspace(1)* %a
  br label %call

bb2:
  store i32 1, i32 addrspace(1)* %a
  br label %call

call:
  call void @llvm.amdgcn.ds.gws.init(i32 %val, i32 %offset)
  br label %ret

ret:
  ret void
}

; GCN-LABEL: {{^}}taildup_gws_barrier:
; GCN: ds_gws_barrier
; GCN-NOT: ds_gws_barrier
define amdgpu_kernel void @taildup_gws_barrier(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i1 %cond, i32 %val, i32 %offset) #0 {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, i32 addrspace(1)* %a
  br label %call

bb2:
  store i32 1, i32 addrspace(1)* %a
  br label %call

call:
  call void @llvm.amdgcn.ds.gws.barrier(i32 %val, i32 %offset)
  br label %ret

ret:
  ret void
}

; GCN-LABEL: {{^}}taildup_gws_sema_release_all:
; GCN: ds_gws_sema_release_all
; GCN-NOT: ds_gws
define amdgpu_kernel void @taildup_gws_sema_release_all(i32 addrspace(1)* %a, i32 addrspace(1)* %b, i1 %cond, i32 %offset) #0 {
entry:
  br i1 %cond, label %bb1, label %bb2

bb1:
  store i32 0, i32 addrspace(1)* %a
  br label %call

bb2:
  store i32 1, i32 addrspace(1)* %a
  br label %call

call:
  call void @llvm.amdgcn.ds.gws.sema.release.all(i32 %offset)
  br label %ret

ret:
  ret void
}

attributes #0 = { nounwind }
attributes #1 = { nounwind convergent }
attributes #2 = { convergent inaccessiblememonly nounwind }
