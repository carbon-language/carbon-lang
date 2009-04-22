// RUN: not %llvmgcc -O1 %s -S |& grep {error: storage size}
// PR2958
static struct foo s;
struct foo *p = &s;
