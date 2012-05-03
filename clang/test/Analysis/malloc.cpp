// RUN: %clang_cc1 -analyze -analyzer-checker=core,experimental.deadcode.UnreachableCode,experimental.core.CastSize,unix.Malloc -analyzer-store=region -verify %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);

// Test for radar://11110132.
struct Foo {
    mutable void* m_data;
    Foo(void* data) : m_data(data) {}
};
Foo aFunction() {
    return malloc(10);
}

// Assume that functions which take a function pointer can free memory even if
// they are defined in system headers and take the const pointer to the
// allocated memory. (radar://11160612)
// Test default parameter.
int const_ptr_and_callback_def_param(int, const char*, int n, void(*)(void*) = 0);
void r11160612_3() {
  char *x = (char*)malloc(12);
  const_ptr_and_callback_def_param(0, x, 12);
}
