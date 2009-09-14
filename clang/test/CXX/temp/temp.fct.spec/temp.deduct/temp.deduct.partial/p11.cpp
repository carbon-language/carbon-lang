// RUN: clang-cc -fsyntax-only %s

template <class T> T* f(int);	// #1 
template <class T, class U> T& f(U); // #2 

void g() {
  int *ip = f<int>(1);	// calls #1
}

// FIXME: test occurrences of template parameters in non-deduced contexts.
