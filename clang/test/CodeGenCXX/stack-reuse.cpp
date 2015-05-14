// RUN: %clang -target armv7l-unknown-linux-gnueabihf -S %s -o - -emit-llvm -O1 -disable-llvm-optzns | FileCheck %s

// Stack should be reused when possible, no need to allocate two separate slots
// if they have disjoint lifetime.

// Sizes of objects are related to previously existed threshold of 32.  In case
// of S_large stack size is rounded to 40 bytes.

// 32B
struct S_small {
  int a[8];
};

// 36B
struct S_large {
  int a[9];
};

extern S_small foo_small();
extern S_large foo_large();
extern void bar_small(S_small*);
extern void bar_large(S_large*);

// Prevent mangling of function names.
extern "C" {

void small_rvoed_unnamed_temporary_object() {
// CHECK-LABEL: define void @small_rvoed_unnamed_temporary_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_smallv
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_smallv
// CHECK: call void @llvm.lifetime.end

  foo_small();
  foo_small();
}

void large_rvoed_unnamed_temporary_object() {
// CHECK-LABEL: define void @large_rvoed_unnamed_temporary_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_largev
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_largev
// CHECK: call void @llvm.lifetime.end

  foo_large();
  foo_large();
}

void small_rvoed_named_temporary_object() {
// CHECK-LABEL: define void @small_rvoed_named_temporary_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_smallv
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_smallv
// CHECK: call void @llvm.lifetime.end

  {
    S_small s = foo_small();
  }
  {
    S_small s = foo_small();
  }
}

void large_rvoed_named_temporary_object() {
// CHECK-LABEL: define void @large_rvoed_named_temporary_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_largev
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9foo_largev
// CHECK: call void @llvm.lifetime.end

  {
    S_large s = foo_large();
  }
  {
    S_large s = foo_large();
  }
}

void small_auto_object() {
// CHECK-LABEL: define void @small_auto_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9bar_smallP7S_small
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9bar_smallP7S_small
// CHECK: call void @llvm.lifetime.end

  {
    S_small s;
    bar_small(&s);
  }
  {
    S_small s;
    bar_small(&s);
  }
}

void large_auto_object() {
// CHECK-LABEL: define void @large_auto_object
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9bar_largeP7S_large
// CHECK: call void @llvm.lifetime.end
// CHECK: call void @llvm.lifetime.start
// CHECK: call void @_Z9bar_largeP7S_large
// CHECK: call void @llvm.lifetime.end

  {
    S_large s;
    bar_large(&s);
  }
  {
    S_large s;
    bar_large(&s);
  }
}

}
