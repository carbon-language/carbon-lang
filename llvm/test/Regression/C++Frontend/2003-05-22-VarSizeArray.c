
int test(int Num) {
  int Arr[Num];
  Arr[2] = 0;
  return Arr[2];
}

int main() {
  printf("%d\n", test(4));
  return 0;
}
