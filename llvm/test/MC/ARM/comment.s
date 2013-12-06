@ Tests to check that '@' does not get lexed as an identifier for arm
@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi  | FileCheck %s
@ RUN: llvm-mc %s -triple=armv7-linux-gnueabi 2>&1 | FileCheck %s --check-prefix=ERROR

foo:
  bl boo@plt should be ignored
  bl goo@plt
  .long bar@got to parse this as a comment
  .long baz@got
  add r0, r0@ignore this extra junk

@CHECK-LABEL: foo:
@CHECK: bl boo
@CHECK-NOT: @
@CHECK: bl goo
@CHECK-NOT: @
@CHECK: .long bar
@CHECK-NOT: @
@CHECK: .long baz
@CHECK-NOT: @
@CHECK: add r0, r0
@CHECK-NOT: @

@ERROR-NOT: error:
