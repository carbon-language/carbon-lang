// RUN: clang-cc -emit-llvm %s -o %t -fblocks

void foo (void(^)());

int main()
{
foo(
  ^() { }
);
}
