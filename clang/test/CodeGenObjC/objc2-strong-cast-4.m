// RUN: clang-cc -triple x86_64-apple-darwin10 -fobjc-gc -emit-llvm -o %t %s &&
// RUN: grep objc_assign_strongCast %t | count 7 &&
// RUN: true

struct Slice {
    void *__strong * items;
};

typedef struct Slice Slice;

@interface ISlice {
@public
    void *__strong * IvarItem;
}
@end

void foo (int i) {
    // storing into an array of strong pointer types.
    void *__strong* items;
    items[i] = 0;

    // storing indirectly into an array of strong pointer types.
    void *__strong* *vitems;
    *vitems[i] = 0;

    Slice *slice;
    slice->items = 0;
    // storing into a struct element of an array of strong pointer types.
    slice->items[i] = 0;

    ISlice *islice;
    islice->IvarItem = 0;
    // Storing into an ivar of an array of strong pointer types.
    islice->IvarItem[i] = (void*)0;
}
