// RUN: clang-cc -fsyntax-only -verify %s
void* a(unsigned x) {
return __builtin_return_address(0);
}

void b(unsigned x) {
return __builtin_return_address(x); // expected-error{{the level argument for a stack address builtin must be constant}}
}

void* c(unsigned x) {
return __builtin_frame_address(0);
}

void d(unsigned x) {
return __builtin_frame_address(x); // expected-error{{the level argument for a stack address builtin must be constant}}
}
