// RUN: %clang_cc1 -triple x86_64-windows -fasync-exceptions -fcxx-exceptions -fexceptions -fms-extensions -x c++ -Wno-implicit-function-declaration -S -emit-llvm %s -o - | FileCheck %s

// CHECK: define dso_local void @"?crash@@YAXH@Z
// CHECK: invoke void @llvm.seh.try.begin()
// CHECK: invoke void @llvm.seh.scope.begin()
// CHECK: invoke void @llvm.seh.scope.end()

// CHECK: %[[dst:[0-9-]+]] = catchswitch within none [label %catch] unwind to caller
// CHECK: %[[dst1:[0-9-]+]] = catchpad within %[[dst]] [i8* null, i32 0, i8* null]
// CHECK: "funclet"(token %[[dst1]])

// CHECK: invoke void @llvm.seh.try.begin()
// CHECK: %[[src:[0-9-]+]] = load volatile i32, i32* %i
// CHECK-NEXT: invoke void @"?crash@@YAXH@Z"(i32 %[[src]])
// CHECK: invoke void @llvm.seh.try.end()

// *****************************************************************************
// Abstract:     Test CPP catch(...) under SEH -EHa option

void printf(...);
int volatile *NullPtr = 0;
void foo() {
  *NullPtr = 0;
}
int *pt1, *pt2, *pt3;
int g;
void crash(int i) {
  g = i;
  try {
    struct A {
      A() {
        printf(" in A ctor \n");
        if (g == 0)
          *NullPtr = 0;
      }
      ~A() {
        printf(" in A dtor \n");
      }
    } ObjA;
    if (i == 1)
      *NullPtr = 0;
  } catch (...) {
    printf(" in catch(...) funclet \n");
    if (i == 1)
      throw(i);
  }
}

int main() {
  for (int i = 0; i < 2; i++) {
    __try {
      crash(i);
    } __except (1) {
      printf(" Test CPP unwind: in except handler i = %d \n", i);
    }
  }
  return 0;
}
