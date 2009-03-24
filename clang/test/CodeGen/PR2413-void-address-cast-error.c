// RUN: clang-cc -emit-llvm %s -o -
void f()
{
        void *addr;
        addr = (void *)( ((long int)addr + 7L) );
}
