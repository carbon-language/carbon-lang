// RUN: clang-cc -triple x86_64-apple-darwin10 -fobjc-gc -emit-llvm -o %t %s &&
// RUN: grep objc_assign_strongCast %t | count 3 &&
// RUN: true

struct Slice {
    void *__strong * items;
};

typedef struct Slice Slice;

@interface ISlice {
@public
    void *__strong * items;
}
@end

void foo () {
    Slice *slice;
    slice->items = 0;

    ISlice *islice;
    islice->items = 0;
}
