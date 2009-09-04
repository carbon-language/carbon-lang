// RUN: clang-cc -emit-llvm -femit-all-decls -o %t %s &&
// RUN: grep "_ZNK4plusIillEclERKiRKl" %t | count 1

// FIXME: We should not need the -femit-all-decls, because operator() should
// be emitted as an external symbol rather than with linkonce_odr linkage.
// This is a Sema problem.
template<typename T, typename U, typename Result>
struct plus {
  Result operator()(const T& t, const U& u) const;
};

template<typename T, typename U, typename Result>
Result plus<T, U, Result>::operator()(const T& t, const U& u) const {
  return t + u;
}

template struct plus<int, long, long>;
