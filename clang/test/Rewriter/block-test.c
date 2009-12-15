// RUN: %clang_cc1 -rewrite-blocks %s -fblocks -o -

static int (^block)(const void *, const void *) = (int (^)(const void *, const void *))0;
static int (*func)(int (^block)(void *, void *)) = (int (*)(int (^block)(void *, void *)))0;

typedef int (^block_T)(const void *, const void *);
typedef int (*func_T)(int (^block)(void *, void *));

void foo(const void *a, const void *b, void *c) {
    int (^block)(const void *, const void *) = (int (^)(const void *, const void *))c;
    int (*func)(int (^block)(void *, void *)) = (int (*)(int (^block)(void *, void *)))c;
}

typedef void (^test_block_t)();

int main(int argc, char **argv) {
    int a;

    void (^test_block_v)();
    void (^test_block_v2)(int, float);

    void (^test_block_v3)(void (^barg)(int));

    a = 77;
    test_block_v = ^(){ int local=1; printf("a=%d\n",a+local); };
    test_block_v();
    a++;
    test_block_v();

    __block int b;

    b = 88; 
    test_block_v2 = ^(int x, float f){ printf("b=%d\n",b); };
    test_block_v2(1,2.0);
    b++;
    test_block_v2(3,4.0);
    return 7;
}
