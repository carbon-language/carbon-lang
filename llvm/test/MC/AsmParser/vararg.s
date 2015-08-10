// RUN: llvm-mc -triple x86_64-linux-gnu %s | FileCheck %s
.macro ifcc arg:vararg
.if cc
            \arg
.endif
.endm

.macro ifcc2 arg0 arg1:vararg
.if cc
            movl \arg0, \arg1
.endif
.endm

.macro ifcc3 arg0, arg1:vararg
.if cc
            movl \arg0, \arg1
.endif
.endm

.macro ifcc4 arg0, arg1:vararg
.if cc
            movl \arg1, \arg0
.endif
.endm

.text

// CHECK: movl %esp, %ebp
// CHECK: subl $0, %esp
// CHECK: movl %eax, %ebx
// CHECK: movl %ecx, %ebx
// CHECK: movl %ecx, %eax
// CHECK: movl %eax, %ecx
// CHECK: movl %ecx, %eax
// CHECK: movl %eax, %ecx
.set cc,1
  ifcc  movl    %esp, %ebp
        subl $0, %esp

  ifcc2 %eax %ebx
  ifcc2 %ecx, %ebx
  ifcc3 %ecx %eax
  ifcc3 %eax, %ecx
  ifcc4 %eax %ecx  ## test
  ifcc4 %ecx, %eax ## test

// CHECK-NOT: movl
// CHECK: subl $1, %esp
.set cc,0
  ifcc  movl    %esp, %ebp
        subl $1, %esp
