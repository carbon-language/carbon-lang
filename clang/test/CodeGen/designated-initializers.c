// RUN: clang %s -emit-llvm -o -
// XFAIL
struct foo {
    void *a;
    int b;
};

union { int i; float f; } u = { };

int main(int argc, char **argv)
{
  union { int i; float f; } u2 = { };
    static struct foo foo = {
        .b = 1024,
    };
}

