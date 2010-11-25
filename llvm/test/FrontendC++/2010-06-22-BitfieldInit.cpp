// RUN: %llvmgxx -g -S %s
struct TEST2
{
  int subid:32;
  int :0;
};

typedef struct _TEST3
{
  TEST2 foo;
  TEST2 foo2;
} TEST3;

TEST3 test =
  {
    {0},
    {0}
  };

int main() { return 0; }
