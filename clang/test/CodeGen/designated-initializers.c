// RUN: clang-cc -triple i386-unknown-unknown %s -emit-llvm -o - | grep "<{ i8\* null, i32 1024 }>" &&
// RUN: clang-cc -triple i386-unknown-unknown %s -emit-llvm -o - | grep "i32 0, i32 22"

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

int b[2] = {
    [1] 22
};
