// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

template<typename T, T t>
struct TestStruct {
   typedef decltype(t+2) sum_type;
};
