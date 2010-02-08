// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -verify -fblocks -fobjc-gc %s

@interface Intf  {
@public
    void (^block) (id);
    __weak void (^weak_block) (id);
    void (*fptr) (id);
    __weak void (*weak_fptr) (id);
}
@end

int main() {
    Intf *observer;
    return (observer->block != observer->weak_block ||
            observer->fptr != observer->weak_fptr);
}

