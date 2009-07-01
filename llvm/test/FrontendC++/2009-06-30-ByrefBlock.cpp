// Insure __block_holder_tmp is allocated on the stack.
// RUN: %llvmgxx %s -S -O2 -o - | egrep {__block_holder_tmp.*alloca}
// <rdar://problem/5865221>
extern void fubar_dispatch_sync(void (^PP)(void));
void fubar() {
  __block void *voodoo;
 fubar_dispatch_sync(^(void){voodoo=0;});
}
