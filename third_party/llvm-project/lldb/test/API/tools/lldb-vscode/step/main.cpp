int function(int x) {
  if ((x % 2) == 0)
    return function(x-1) + x; // breakpoint 1
  else
    return x;
}

int main(int argc, char const *argv[]) {
  return function(2);
}
