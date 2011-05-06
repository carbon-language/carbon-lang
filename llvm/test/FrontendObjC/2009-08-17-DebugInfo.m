// This is a regression test on debug info to make sure that we can set a
// breakpoint on a objective message.
// RUN: %llvmgcc -S -O0 -g %s -o - | llc -disable-cfi -o %t.s -O0
// RUN: %compile_c %t.s -o %t.o
// RUN: %link %t.o -o %t.exe -framework Foundation
// RUN: echo {break randomFunc\n} > %t.in 
// RUN: gdb -q -batch -n -x %t.in %t.exe | tee %t.out | \
// RUN:   grep {Breakpoint 1 at 0x.*: file .*2009-08-17-DebugInfo.m, line 21}
// XTARGET: darwin
@interface MyClass
{
 int my;
}
+ init;
- randomFunc;
@end

@implementation MyClass
+ init {
}
- randomFunc { my = 42; }
@end

int main() {
  id o = [MyClass init];
  [o randomFunc];
  return 0;
}
