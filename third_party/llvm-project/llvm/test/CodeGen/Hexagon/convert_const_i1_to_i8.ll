; RUN: llc -march=hexagon < %s | FileCheck %s
; CHECK-NOT: .space {{[0-9][0-9][0-9][0-9]}}
; CHECK: q{{[0-3]}} = vand(v{{[0-9]*}},r{{[0-9]*}})

define void @convert_const_i1_to_i8(<32 x i32>* %a0) #0 {
entry:
  %v0 = load <32 x i32>, <32 x i32>* %a0, align 128
  %v1 = tail call <32 x i32> @llvm.hexagon.V6.vrdelta.128B(<32 x i32> %v0, <32 x i32> undef)
  %v2 = tail call <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1> <i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false, i1 true, i1 false, i1 false>, <32 x i32> undef, <32 x i32> %v1)
  store <32 x i32> %v2, <32 x i32>* %a0, align 128
  ret void
}

declare <32 x i32> @llvm.hexagon.V6.vrdelta.128B(<32 x i32>, <32 x i32>)
declare <32 x i32> @llvm.hexagon.V6.vmux.128B(<128 x i1>, <32 x i32>, <32 x i32>)

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }
