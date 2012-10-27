// RUN: %clang -### -S -o %t %s              2>&1 | not grep -w -- -fopenmp
// RUN: %clang -### -S -o %t %s -fopenmp     2>&1 | grep -w -- -fopenmp
