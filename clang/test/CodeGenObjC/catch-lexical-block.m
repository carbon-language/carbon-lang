// RUN: %clang_cc1 -g -fobjc-exceptions -emit-llvm %s -o - | FileCheck %s
@interface Foo @end
void f0() {
  @try {
    @throw @"a";
  } @catch(Foo *e) {
  }
}

// We should have 3 lexical blocks here at the moment, including one
// for the catch block.
// CHECK: !MDLexicalBlock(
// CHECK: !MDLocalVariable(tag: DW_TAG_auto_variable
// CHECK: !MDLexicalBlock(
// CHECK: !MDLexicalBlock(
