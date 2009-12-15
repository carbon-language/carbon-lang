// RUN: %clang_cc1 -fsyntax-only %s

template<typename T>
struct __is_pod {
};

__is_pod<int> ipi;
