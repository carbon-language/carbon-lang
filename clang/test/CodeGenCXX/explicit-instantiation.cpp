// RUN: clang-cc -emit-llvm -femit-all-decls -o %t %s &&
// RUN: grep "_ZNK4plusIillEclERKiRKl" %t | count 1

template<typename T, typename U, typename Result>
struct plus {
  Result operator()(const T& t, const U& u) const {
    return t + u;
  }
};

template struct plus<int, long, long>;
