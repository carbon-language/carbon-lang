# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv62 -filetype=obj %s | llvm-objdump --mcpu=hexagonv62 -d - | FileCheck %s --check-prefix=CHECK-V62
# RUN: llvm-mc -arch=hexagon -mcpu=hexagonv65 -filetype=obj %s | llvm-objdump --mcpu=hexagonv65 -d - | FileCheck %s --check-prefix=CHECK-V65

# CHECK-V62: trap1(#0)
# CHECK-V65: trap1(r0,#0)
trap1(#0)
