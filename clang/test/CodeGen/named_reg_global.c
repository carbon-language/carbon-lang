// RUN: %clang_cc1 -triple x86_64-linux-gnu -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple arm64-linux-gnu -S -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple armv7-linux-gnu -S -emit-llvm %s -o - | FileCheck %s

// CHECK-NOT: @sp = common global
register unsigned long current_stack_pointer asm("sp");

// CHECK: define{{.*}} i[[bits:[0-9]+]] @get_stack_pointer_addr()
// CHECK: [[ret:%[0-9]+]] = call i[[bits]] @llvm.read_register.i[[bits]](metadata !0)
// CHECK: ret i[[bits]] [[ret]]
unsigned long get_stack_pointer_addr() {
  return current_stack_pointer;
}
// CHECK: declare{{.*}} i[[bits]] @llvm.read_register.i[[bits]](metadata)

// CHECK: define{{.*}} void @set_stack_pointer_addr(i[[bits]] %addr) #0 {
// CHECK: [[sto:%[0-9]+]] = load i[[bits]]* %
// CHECK: call void @llvm.write_register.i[[bits]](metadata !0, i[[bits]] [[sto]])
// CHECK: ret void
void set_stack_pointer_addr(unsigned long addr) {
  current_stack_pointer = addr;
}
// CHECK: declare{{.*}} void @llvm.write_register.i[[bits]](metadata, i[[bits]])

// CHECK: !llvm.named.register.sp = !{!0}
// CHECK: !0 = metadata !{metadata !"sp"}
