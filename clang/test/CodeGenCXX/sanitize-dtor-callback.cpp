// Test -fsanitize-memory-use-after-dtor
// RUN: %clang_cc1 -fsanitize=memory -fsanitize-memory-use-after-dtor -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -fsanitize=memory -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s -check-prefix=NO-DTOR-CHECK

// RUN: %clang_cc1 -std=c++11 -fsanitize=memory -fsanitize-memory-use-after-dtor -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s -check-prefix=STD11
// RUN: %clang_cc1 -std=c++11 -fsanitize=memory -triple=x86_64-pc-linux -emit-llvm -o - %s | FileCheck %s -check-prefix=NO-DTOR-STD11-CHECK

struct Simple {
  ~Simple() {}
};
Simple s;
// Simple internal member is poisoned by compiler-generated dtor
// CHECK-LABEL: @_ZN6SimpleD2Ev
// CHECK: call void @__sanitizer_dtor_callback
// CHECK: ret void

// Compiling without the flag does not generate member-poisoning dtor
// NO-DTOR-CHECK-LABEL: @_ZN6SimpleD2Ev
// NO-DTOR-CHECK-NOT: call void @__sanitizer_dtor_callback
// NO-DTOR-CHECK: ret void


struct Inlined {
  inline ~Inlined() {}
};
Inlined in;
// Dtor that is inlined where invoked poisons object
// CHECK-LABEL: @_ZN7InlinedD2Ev
// CHECK: call void @__sanitizer_dtor_callback
// CHECK: ret void

// Compiling without the flag does not generate member-poisoning dtor
// NO-DTOR-CHECK-LABEL: @_ZN7InlinedD2Ev
// NO-DTOR-CHECK-NOT: call void @__sanitizer_dtor_callback
// NO-DTOR-CHECK: ret void


struct Defaulted_Trivial {
  ~Defaulted_Trivial() = default;
};
int main() {
  Defaulted_Trivial def_trivial;
}
// The compiler is explicitly signalled to handle object cleanup.
// No complex member attributes ensures that the compiler destroys
// the memory inline. However, it must still poison this memory.
// STD11-CHECK-LABEL: alloca %struct.Defaulted_Trivial
// STD11: call void @__sanitizer_dtor_callback
// STD11: ret void

// Compiling without the flag does not generate member-poisoning dtor
// NO-DTOR-STD11-CHECK-LABEL: alloca %struct.Defaulted_Trivial
// NO-DTOR-STD11-CHECK-NOT: call void @__sanitizer_dtor_callback
// NO-DTOR-STD11-CHECK: ret void


struct Defaulted_Non_Trivial {
  Simple s;
  ~Defaulted_Non_Trivial() = default;
};
Defaulted_Non_Trivial def_non_trivial;
// Explicitly compiler-generated dtor poisons object.
// By including a Simple member in the struct, the compiler is
// forced to generate a non-trivial destructor..
// STD11-CHECK-LABEL: @_ZN21Defaulted_Non_TrivialD2Ev
// STD11: call void @__sanitizer_dtor_callback
// STD11: ret void

// Compiling without the flag does not generate member-poisoning dtor
// NO-DTOR-STD11-CHECK-LABEL: @_ZN21Defaulted_Non_TrivialD2Ev
// NO-DTOR-STD11-CHECK-NOT: call void @__sanitizer_dtor_callback
// NO-DTOR-STD11-CHECK: ret void
