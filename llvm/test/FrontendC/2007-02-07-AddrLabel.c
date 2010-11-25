// PR947
// RUN: %llvmgcc %s -S -o - 

void foo() {
    void *ptr;
  label:
    ptr = &&label;

    goto *ptr;
  }
