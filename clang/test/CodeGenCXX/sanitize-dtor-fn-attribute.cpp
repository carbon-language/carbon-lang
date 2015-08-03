// Test -fsanitize-memory-use-after-dtor
// RUN: %clang_cc1 -fsanitize=memory -fsanitize-memory-use-after-dtor -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -DATTRIBUTE -fsanitize=memory -fsanitize-memory-use-after-dtor -std=c++11 -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-ATTR

template <class T> class Vector {
 public:
  ~Vector() {}
};

struct No_San {
  Vector<int> v;
  No_San() { }
#ifdef ATTRIBUTE
  __attribute__((no_sanitize_memory)) ~No_San() = default;
#else
  ~No_San() = default;
#endif
};

int main() {
  No_San *ns = new No_San();
  ns->~No_San();
  return 0;
}

// Repressing the sanitization attribute results in no msan
// instrumentation of the destructor
// CHECK: define {{.*}}No_SanD1Ev{{.*}} [[ATTRIBUTE:#[0-9]+]]
// CHECK: call void {{.*}}No_SanD2Ev
// CHECK: call void @__sanitizer_dtor_callback
// CHECK: ret void

// CHECK-ATTR: define {{.*}}No_SanD1Ev{{.*}} [[ATTRIBUTE:#[0-9]+]]
// CHECK-ATTR: call void {{.*}}No_SanD2Ev
// CHECK-ATTR-NOT: call void @__sanitizer_dtor_callback
// CHECK-ATTR: ret void


// CHECK: define {{.*}}No_SanD2Ev{{.*}} [[ATTRIBUTE:#[0-9]+]]
// CHECK: call void {{.*}}Vector
// CHECK: call void @__sanitizer_dtor_callback
// CHECK: ret void

// CHECK-ATTR: define {{.*}}No_SanD2Ev{{.*}} [[ATTRIBUTE:#[0-9]+]]
// CHECK-ATTR: call void {{.*}}Vector
// CHECK-ATTR-NOT: call void @__sanitizer_dtor_callback
// CHECK-ATTR: ret void

// When attribute is repressed, the destructor does not emit any tail calls
// CHECK: attributes [[ATTRIBUTE]] = {{.*}} sanitize_memory
// CHECK-ATTR-NOT: attributes [[ATTRIBUTE]] = {{.*}} sanitize_memory
