// RUN: clang-cc -fsyntax-only -verify %s
template<int I, int J, class T> class X { 
  static const int value = 0;
};

template<int I, int J> class X<I, J, int> { 
  static const int value = 1;
};

template<int I> class X<I, I, int> { 
  static const int value = 2;
};

int array0[X<0, 0, float>::value == 0? 1 : -1];
int array1[X<0, 1, int>::value == 1? 1 : -1];
int array2[X<0, 0, int>::value == 2? 1 : -1];
