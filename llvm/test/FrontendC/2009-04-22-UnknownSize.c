// RUN: not %llvmgcc -O1 %s -S -o /dev/null |& grep {error: storage size}
// PR2958
static struct foo s;
struct foo *p = &s;
