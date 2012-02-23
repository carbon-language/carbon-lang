// RUN: %clang_cc1 -triple x86_64-apple-darwin -fobjc-default-synthesize-properties -emit-llvm  %s -o - | FileCheck %s
// rdar://10907410

@protocol P
@optional
@property int auto_opt_window;
@property int no_auto_opt_window;

@required
@property int auto_req_window;
@property int no_auto_req_window; // expected-note {{property declared here}}
@end

@interface I<P>
@property int auto_opt_window;
@property int auto_req_window;
@end

@implementation I // expected-warning {{auto property synthesis will not synthesize property declared in a protocol}}
@end

// CHECK: define internal i32 @"\01-[I auto_opt_window]"(
// CHECK: define internal void @"\01-[I setAuto_opt_window:]"(
// CHECK: define internal i32 @"\01-[I auto_req_window]"(
// CHECK: define internal void @"\01-[I setAuto_req_window:]"(

// CHECK-NOT: define internal i32 @"\01-[I no_auto_opt_window]"(
// CHECK-NOT: define internal void @"\01-[I setNo_auto_opt_window:]"(
// CHECK-NOT: define internal i32 @"\01-[I no_auto_req_window]"(
// CHECK-NOT: define internal void @"\01-[I setNo_auto_req_window:]"(
