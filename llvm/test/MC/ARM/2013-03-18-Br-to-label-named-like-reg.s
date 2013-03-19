@ RUN: llvm-mc -arch arm %s
@ CHECK: test:
@ CHECK: br r1
test:
  bl r1
