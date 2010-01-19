// RUN: %clang_cc1 -x objective-c++ -fblocks -fms-extensions -rewrite-objc -o - %s

extern "C" {
extern "C" void *_Block_copy(const void *aBlock);
extern "C" void _Block_release(const void *aBlock);
}

int main() {
    __attribute__((__blocks__(byref))) int a = 42;
    int save_a = a;

    void (^b)(void) = ^{
        ((__typeof(^{ a = 2; }))_Block_copy((const void *)(^{ a = 2; })));
    };

    ((__typeof(b))_Block_copy((const void *)(b)));

    return 0;
}

