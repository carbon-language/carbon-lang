# RUN: llvm-mc -arch=hexagon -filetype=obj %s | llvm-objdump -d - | FileCheck %s

{ p0 = cmp.eq(r0,#0) ; if (p0.new) dealloc_return:t }

# CHECK: { p0 = cmp.eq(r0,#0)
# CHECK: if (p0.new) dealloc_return:t }
