// RUN: %clang  -Werror -fobjc-arc -fsyntax-only -fno-objc-arc -Xclang -verify %s
// rdar://8949617

void * FOO() {
    id string = @"Hello World.\n";
    void *pointer = string; // No error must be issued
    return pointer;
}
