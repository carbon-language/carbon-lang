; RUN: llc -mtriple=amdgcn-amd-amdhsa --amdhsa-code-object-version=3 -filetype=obj -o - < %s | llvm-readelf --notes - | FileCheck %s

; Make sure llc does not crash for invalid opencl version metadata.

; CHECK: ---
; CHECK: amdhsa.version:
; CHECK-NEXT: - 1
; CHECK-NEXT: - 0
; CHECK: ...

!opencl.ocl.version = !{!0}
!0 = !{}
