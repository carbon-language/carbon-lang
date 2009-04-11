// RUN: clang-cc -emit-llvm %s -o %t

void foo (void(^)());

int main()
{
foo(
  ^()
   {
   }
);
}
