define i32 @__clc_atomic_add_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile add i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_add_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile add i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_and_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile and i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_and_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile and i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_max_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_max_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile max i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_umax_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_umax_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile umax i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_sub_addr1(i32 addrspace(1)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile sub i32 addrspace(1)* %ptr, i32 %value seq_cst
  ret i32 %0
}

define i32 @__clc_atomic_sub_addr3(i32 addrspace(3)* nocapture %ptr, i32 %value) nounwind alwaysinline {
entry:
  %0 = atomicrmw volatile sub i32 addrspace(3)* %ptr, i32 %value seq_cst
  ret i32 %0
}
