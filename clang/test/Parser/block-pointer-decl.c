// RUN: clang -fsyntax-only -verify -parse-noop %s

struct blockStruct {
  int (^a)(float, int);
  int b;
};

int blockTaker (int (^myBlock)(int), int other_input)
{
  return 0;
}

int main (int argc, char **argv)
{
  int (^blockptr) (int);
  return 0;
}

