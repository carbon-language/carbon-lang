# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv60 -mattr=+hvx -filetype=obj %s | llvm-objdump -arch=hexagon -mcpu=hexagonv60 -mattr=+hvx -d - | FileCheck %s

# CHECK: 1c2eceee { v14 = vxor(v14,{{ *}}v14) }
v14 = #0

# CHECK: 1c80c0a0 { v1:0.w = vsub(v1:0.w,v1:0.w) }
v1:0 = #0

# CHECK: 1f42c3e0 { v1:0 = vcombine(v3,v2) }
v1:0 = v3:2
