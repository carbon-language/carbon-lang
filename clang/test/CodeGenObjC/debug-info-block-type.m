// RUN: %clang_cc1 -emit-llvm -fblocks -g  -triple x86_64-apple-darwin14 -x objective-c < %s -o - | FileCheck %s
#define nil ((void*) 0)
typedef signed char BOOL;
// CHECK: ![[BOOL:[0-9]+]] = {{.*}} [ DW_TAG_typedef ] [BOOL] [line [[@LINE-1]]
// CHECK: ![[ID:[0-9]+]] = {{.*}} [ DW_TAG_typedef ] [id]

typedef BOOL (^SomeKindOfPredicate)(id obj);
// CHECK: ![[PTR:[0-9]+]]} ; [ DW_TAG_member ] [__FuncPtr]
// CHECK: ![[PTR]] = {{.*}}, ![[FNTYPE:[0-9]+]]} ; [ DW_TAG_pointer_type ]
// CHECK: ![[FNTYPE]] = {{.*}} ![[ARGS:[0-9]+]]{{.*}} ; [ DW_TAG_subroutine_type ]
// CHECK: ![[ARGS]] = !{![[BOOL]], ![[ID]]}

int main()
{
  SomeKindOfPredicate p = ^BOOL(id obj) { return obj != nil; };
  // CHECK: ![[PTR]]} ; [ DW_TAG_member ] [__FuncPtr] [line [[@LINE-1]], size 64, align 64, offset 128]
  return p(nil);
}
