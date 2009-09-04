// RUN: clang-cc -fsyntax-only %s

template<typename T>
struct __is_pod {
};

__is_pod<int> ipi;
