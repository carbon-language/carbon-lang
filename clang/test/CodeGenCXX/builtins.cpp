// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

// PR8839
extern "C" char memmove();

int main() {
  // CHECK: call signext i8 @memmove()
  return memmove();
}

// <rdar://problem/10063539>

template<int (*Compare)(const char *s1, const char *s2)>
int equal(const char *s1, const char *s2) {
  return Compare(s1, s2) == 0;
}

// CHECK: define weak_odr i32 @_Z5equalIXadL_Z16__builtin_strcmpPKcS1_EEEiS1_S1_
// CHECK: call i32 @strcmp
template int equal<&__builtin_strcmp>(const char*, const char*);

