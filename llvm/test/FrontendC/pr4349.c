// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep svars2 | grep {\\\[2 x \\\[2 x i8\\\]\\\]}
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep svars2 | grep {, i\[\[:digit:\]\]\\+ 1)} | count 1
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep svars3 | grep {\\\[2 x i16\\\]}
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep svars3 | grep {, i\[\[:digit:\]\]\\+ 1)} | count 1
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep svars4 | grep {\\\[2 x \\\[2 x i8\\\]\\\]} | count 1
// RUN: %llvmgcc %s -S -emit-llvm -O0 -o - | grep svars4 | grep {, i\[\[:digit:\]\]\\+ 1, i\[\[:digit:\]\]\\+ 1)} | count 1
// PR 4349

union reg
{
    unsigned char b[2][2];
    unsigned short w[2];
    unsigned int d;
};
struct cpu
{
    union reg pc;
};
extern struct cpu cpu;
struct svar
{
    void *ptr;
};
struct svar svars1[] =
{
    { &((cpu.pc).w[0]) }
};
struct svar svars2[] =
{
    { &((cpu.pc).b[0][1]) }
};
struct svar svars3[] =
{
    { &((cpu.pc).w[1]) }
};
struct svar svars4[] =
{
    { &((cpu.pc).b[1][1]) }
};
