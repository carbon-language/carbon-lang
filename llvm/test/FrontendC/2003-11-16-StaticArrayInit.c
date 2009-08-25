// RUN: %llvmgcc -S %s -o - | llvm-as -o /dev/null

void bar () {
 static char x[10];
 static char *xend = x + 10;
}


