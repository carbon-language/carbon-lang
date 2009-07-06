// Insure __block_holder_tmp is allocated on the stack.  Darwin only.
// RUN: %llvmgxx %s -S -O2 -o - | egrep {__block_holder_tmp.*alloca}
// XFAIL: *
// XTARGET: darwin
// <rdar://problem/5865221>
// END.
extern void fubar_dispatch_sync(void (^PP)(void));
void fubar() {
  __block void *voodoo;
 fubar_dispatch_sync(^(void){voodoo=0;});
}
