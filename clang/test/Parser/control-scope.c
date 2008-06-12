// RUN: not clang %s -std=c90 &&
// RUN: clang %s -std=c99

int f (int z) { 
  if (z + sizeof (enum {a})) 
    return 1 + sizeof (enum {a}); 
  return 0; 
}
