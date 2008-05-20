// RUN: clang -emit-llvm < %s
void* a(unsigned x) {
return __builtin_return_address(0);
}

void* c(unsigned x) {
return __builtin_frame_address(0);
}
// RUN: clang -emit-llvm < %s
void* a(unsigned x) {
return __builtin_return_address(0);
}

void* c(unsigned x) {
return __builtin_frame_address(0);
}
