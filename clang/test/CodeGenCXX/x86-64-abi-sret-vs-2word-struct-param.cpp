// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s
// XTARGET: x86
// PR4242
// (PR 4242 bug is on 64-bit only, test passes on x86-32 as well)

struct S {
    void* data[3];
};

struct T {
    void* data[2];
};

// CHECK: %struct.T* byval
extern "C" S fail(int, int, int, int, T t, void* p) {
    S s;
    s.data[0] = t.data[0];
    s.data[1] = t.data[1];
    s.data[2] = p;
    return s;
}

// CHECK: %struct.T* byval
extern "C" S* succeed(S* sret, int, int, int, int, T t, void* p) {
    sret->data[0] = t.data[0];
    sret->data[1] = t.data[1];
    sret->data[2] = p;
    return sret;
}
