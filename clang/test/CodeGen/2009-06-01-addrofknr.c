// RUN: clang-cc %s -o %t -emit-llvm -verify
// PR4289

struct funcptr {
    int (*func)();
};

static int func(f)
        void *f;
{
}

int
main(int argc, char *argv[])
{
        struct funcptr fp;

        fp.func = &func;
        fp.func = func;
}

