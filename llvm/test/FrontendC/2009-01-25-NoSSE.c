// RUN: $llvmgcc -m64 -O1 -march=core2 -mno-sse %s -S -o - | not grep xmm
// PR3402
// This is a test for x86/x86-64, add your target below if it FAILs.
// XFAIL: alpha|ia64|arm|powerpc|sparc 
// reverted
// XFAIL: *
typedef unsigned long __kernel_size_t;
typedef __kernel_size_t size_t;
void *memset(void *s, int c, size_t n);

typedef unsigned char cc_t;
typedef unsigned int speed_t;
typedef unsigned int tcflag_t;

struct ktermios {
 tcflag_t c_iflag;
 tcflag_t c_oflag;
 tcflag_t c_cflag;
 tcflag_t c_lflag;
 cc_t c_line;
 cc_t c_cc[19];
 speed_t c_ispeed;
 speed_t c_ospeed;
};
void bar(struct ktermios*);
void foo()
{
    struct ktermios termios;
    memset(&termios, 0, sizeof(termios));
    bar(&termios);
}

