// RUN: clang-cc -analyze -checker-cfref --analyzer-store=basic -analyzer-constraints=range --verify -fblocks %s
// RUN: clang-cc -analyze -checker-cfref --analyzer-store=region -analyzer-constraints=range --verify -fblocks %s

// <rdar://problem/6776949>
// main's 'argc' argument is always > 0
int main(int argc, char* argv[]) {
  int *p = 0;

  if (argc == 0)
    *p = 1;

  if (argc == 1)
    return 1;

  int x = 1;
  int i;
  
  for(i=1;i<argc;i++){
    p = &x;
  }

  return *p; // no-warning
}
