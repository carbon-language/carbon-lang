declare void @llvm.AMDGPU.barrier.local() nounwind
declare void @llvm.AMDGPU.barrier.global() nounwind

define void @barrier_local() nounwind alwaysinline {
  call void @llvm.AMDGPU.barrier.local()
  ret void
}

define void @barrier_global() nounwind alwaysinline {
  call void @llvm.AMDGPU.barrier.global()
  ret void
}
