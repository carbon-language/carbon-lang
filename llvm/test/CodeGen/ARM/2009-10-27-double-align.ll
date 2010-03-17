; RUN: llc < %s  -mtriple=arm-linux-gnueabi  | FileCheck %s

@.str = private constant [1 x i8] zeroinitializer, align 1

define arm_aapcscc void @g() {
entry:
;CHECK: [sp, #8]
;CHECK: [sp, #12]
;CHECK: [sp]
        tail call arm_aapcscc  void (i8*, ...)* @f(i8* getelementptr ([1 x i8]* @.str, i32 0, i32 0), i32 1, double 2.000000e+00, i32 3, double 4.000000e+00)
        ret void
}

declare arm_aapcscc void @f(i8*, ...)
