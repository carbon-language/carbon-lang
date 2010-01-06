// RUN: %clang_cc1 -fblocks -fobjc-gc -rewrite-objc %s -o -

#define nil 0
int main() {
        __weak __block id foo = nil;
        __block id foo2 = nil;
        id foo3 = nil;

        void (^myblock)() = ^{
                foo = nil;
                foo2 = nil;
                [foo3 bar];
                id foo4 = foo3;
        };
}
