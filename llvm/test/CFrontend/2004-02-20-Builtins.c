// RUN: %llvmgcc -O3 -xc %s -c -o - | llvm-dis | not grep builtin

void zsqrtxxx(float num) {
   num = sqrt(num);
}

