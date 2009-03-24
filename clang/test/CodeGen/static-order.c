// RUN: clang-cc -emit-llvm -o - %s | not grep "zeroinitializer"

struct s {
    int a;
};

static void *v;

static struct s a;

static struct s a = {
    10
};

void *f()
{
  if (a.a)
    return v;
}
