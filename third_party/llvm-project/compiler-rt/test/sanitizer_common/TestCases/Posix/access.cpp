// RUN: %clangxx -O0 -g %s -o %t && %run %t

#include <unistd.h>

int main(void) { return access("/dev/null", F_OK); }
