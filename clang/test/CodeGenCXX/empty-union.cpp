// RUN: clang-cc -emit-llvm -o - %s

union sigval { };

union sigval sigev_value;

int main()
{
  return sizeof(sigev_value);
}
