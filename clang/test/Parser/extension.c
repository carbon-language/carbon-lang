// RUN: clang-cc %s -fsyntax-only

// Top level extension marker.

__extension__ typedef struct
{
    long long int quot; 
    long long int rem; 
}lldiv_t; 


// Compound expr __extension__ marker.
void bar() {
  __extension__ int i;
  int j;
}

