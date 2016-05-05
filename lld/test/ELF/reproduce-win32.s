# REQUIRES: x86, system-windows

# RUN: rm -rf %t.dir
# RUN: mkdir -p %t.dir/build
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir/build/foo.o
# RUN: cd %t.dir
# RUN: ld.lld --hash-style=gnu build/foo.o -o bar --reproduce repro
# RUN: cpio -t < repro.cpio | grep -F 'repro\%:t.dir\build\foo.o' -

.globl _start
_start:
  ret
