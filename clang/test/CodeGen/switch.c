// RUN: clang-cc -triple i386-unknown-unknown -O3 %s -emit-llvm -o - | FileCheck %s

int foo(int i) {
  int j = 0;
  switch (i) {
  case -1:
    j = 1; break;
  case 1 :
    j = 2; break;
  case 2:
    j = 3; break;
  default:
    j = 42; break;
  }
  j = j + 1;
  return j;
}

int foo2(int i) {
  int j = 0;
  switch (i) {
  case 1 :
    j = 2; break;
  case 2 ... 10:
    j = 3; break;
  default:
    j = 42; break;
  }
  j = j + 1;
  return j;
}

int foo3(int i) {
  int j = 0;
  switch (i) {
  default:
    j = 42; break;
  case 111:
    j = 111; break;
  case 0 ... 100:
    j = 1; break;
  case 222:
    j = 222; break;
  }
  return j;
}


static int foo4(int i) {
  int j = 0;
  switch (i) {
  case 111:
    j = 111; break;
  case 0 ... 100:
    j = 1; break;
  case 222:
    j = 222; break;
  default:
    j = 42; break;
  case 501 ... 600:
    j = 5; break;
  }
  return j;
}

// CHECK: define i32 @foo4t()
// CHECK: ret i32 376
// CHECK: }
int foo4t() {
  // 111 + 1 + 222 + 42 = 376
  return foo4(111) + foo4(99) + foo4(222) + foo4(601);
}

// CHECK: define void @foo5()
// CHECK-NOT: switch
// CHECK: }
void foo5(){
    switch(0){
    default:
        if (0) {

        }
    }
}

// CHECK: define void @foo6()
// CHECK-NOT: switch
// CHECK: }
void foo6(){
    switch(0){
    }
}

// CHECK: define void @foo7()
// CHECK-NOT: switch
// CHECK: }
void foo7(){
    switch(0){
      foo7();
    }
}

