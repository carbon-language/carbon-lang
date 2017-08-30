# RUN: llvm-mc %p/Inputs/nested-archives.s -filetype=obj -triple=x86_64-windows-msvc -o %t1.obj
# RUN: rm -f %t1.lib %t2.lib
# RUN: llvm-ar cru %t1.lib %t1.obj
# RUN: llvm-ar cru %t2.lib %t1.lib

# RUN: llvm-mc %s -filetype=obj -triple=x86_64-windows-msvc -o %t2.obj
# RUN: lld-link -entry:main -nodefaultlib %t2.obj %t2.lib -verbose | FileCheck %s
# CHECK: Loaded nested-archives.s.tmp1.lib(nested-archives.s.tmp1.obj) for sub

.text
.global main, sub
main:
  call sub
