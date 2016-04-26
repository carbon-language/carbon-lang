# REQUIRES: x86

# RUN: rm -rf %t.dir1
# RUN: mkdir -p %t.dir1/build
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.dir1/build/foo.o
# RUN: cd %t.dir1
# RUN: ld.lld build/foo.o -o bar -shared --as-needed --reproduce repro
# RUN: diff build/foo.o repro/build/foo.o

.globl _start
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall

# RUN: FileCheck %s --check-prefix=INVOCATION < repro/invocation.txt
# INVOCATION: lld{{[^\s]*}} build/foo.o -o bar -shared --as-needed --reproduce repro

# RUN: not ld.lld build/foo.o -o bar -shared --as-needed --reproduce repro 2>&1 \
# RUN:   | FileCheck --check-prefix=ERROR %s
# ERROR: --reproduce: can't create directory
