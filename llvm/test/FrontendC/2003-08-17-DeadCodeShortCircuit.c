// RUN: %llvmgcc -xc %s -S -o %t.o

int test(_Bool pos, _Bool color) {
  return 0;
  return (pos && color);
}

