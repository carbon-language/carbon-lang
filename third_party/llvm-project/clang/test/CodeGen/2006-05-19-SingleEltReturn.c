// Test returning a single element aggregate value containing a double.
// RUN: %clang_cc1 -triple i686-linux %s -emit-llvm -o - | FileCheck %s --check-prefix=X86_32
// RUN: %clang_cc1 %s -emit-llvm -o -

struct X {
  double D;
};

struct Y {
  struct X x;
};

struct Y bar(void);

void foo(struct Y *P) {
  *P = bar();
}

struct Y bar(void) {
  struct Y a;
  a.x.D = 0;
  return a;
}


// X86_32: define{{.*}} void @foo(%struct.Y* noundef %P)
// X86_32:   call void @bar(%struct.Y* sret(%struct.Y) align 4 %{{[^),]*}})

// X86_32: define{{.*}} void @bar(%struct.Y* noalias sret(%struct.Y) align 4 %{{[^,)]*}})
// X86_32:   ret void
