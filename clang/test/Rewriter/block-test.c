// RUN: clang -rewrite-blocks %s -o -

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
