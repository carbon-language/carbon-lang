// RUN: %llvmgcc -S -O0 -g %s -o - | llvm-as | \
// RUN:     llc --disable-fp-elim -o %t.s -O0 
// RUN: grep id %t.s | grep DW_AT_name
@interface A
-(id) blah;
@end

@implementation A
-(id)blah {
  int i = 1;
  i++;
  return i;
}
@end
