///RUN: %clang_cc1 -triple x86_64-unknown-freebsd9.0 -fobjc-trace -fgnu-runtime -fobjc-dispatch-method=non-legacy -emit-llvm -o - %s | FileCheck %s


@interface A
+ (id)msg;
@end

void f(void) {
  [A msg];
  // CHECK: call void @objc_trace_enter(
  // CHECK: @objc_msgSend
  // CHECK: call void @objc_trace_exit(
}
