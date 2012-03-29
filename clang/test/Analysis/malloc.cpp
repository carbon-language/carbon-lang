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
