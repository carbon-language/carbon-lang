// RUN: clang %s -emit-llvm

static int staticfun(void);
int (*staticuse1)(void) = staticfun;
static int staticfun() {return 1;}
int (*staticuse2)(void) = staticfun;
