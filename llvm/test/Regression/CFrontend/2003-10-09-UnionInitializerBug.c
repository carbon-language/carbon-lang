struct Foo {
    unsigned a;
    unsigned b;
    unsigned c;
};

struct Bar {
    union {
        void **a;
        struct Foo b;
    }u;
};

struct Bar test = {0};

