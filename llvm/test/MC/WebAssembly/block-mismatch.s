# RUN: not llvm-mc -triple=wasm32-unknown-unknown %s -o - 2>&1 | FileCheck %s

# This tests if there are block/loop marker mismatches, the program crashes.
  .text
  .type  test0,@function
test0:
  block
  end_block
  # CHECK: End marker mismatch!
  end_block
  end_function
