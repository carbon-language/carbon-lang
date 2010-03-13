// RUN: %clang_cc1 -emit-llvm -triple i686-pc-linux-gnu -o - %s | FileCheck %s

template<typename T, typename U, typename Result>
struct plus {
  Result operator()(const T& t, const U& u) const;
};

template<typename T, typename U, typename Result>
Result plus<T, U, Result>::operator()(const T& t, const U& u) const {
  return t + u;
}

// CHECK: define weak_odr i32 @_ZNK4plusIillEclERKiRKl
template struct plus<int, long, long>;
