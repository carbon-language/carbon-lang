// RUN: clang -emit-llvm %s

struct FileName {
    struct FileName *next;
} *fnhead;


struct ieeeExternal {
    struct ieeeExternal *next;
} *exthead;


void f()
{
    struct ieeeExternal *exttmp = exthead;
}
