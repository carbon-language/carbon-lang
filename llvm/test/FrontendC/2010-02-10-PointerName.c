// RUN: %llvmgcc %s -S -g -o - | grep DW_TAG_pointer_type | grep -v char

char i = 1;
void foo() {
  char *cp = &i;
}

