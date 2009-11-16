// RUN: clang-cc -emit-llvm -triple i686-pc-linux-gnu -o %t %s
// RUN: grep "define i32 @_ZNK4plusIillEclERKiRKl" %t | count 1

template<typename T, typename U, typename Result>
struct plus {
  Result operator()(const T& t, const U& u) const;
};

template<typename T, typename U, typename Result>
Result plus<T, U, Result>::operator()(const T& t, const U& u) const {
  return t + u;
}

template struct plus<int, long, long>;
