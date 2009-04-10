// RUN: clang-cc %s -emit-llvm -o %t -fblocks &&
// RUN: grep "_Block_object_dispose" %t | count 2 &&
// RUN: grep "__copy_helper_block_" %t | count 2 &&
// RUN: grep "__destroy_helper_block_" %t | count 2 &&
// RUN: grep "__Block_byref_id_object_copy_" %t | count 0 &&
// RUN: grep "__Block_byref_id_object_dispose_" %t | count 0 &&
// RUN: grep "i32 135)" %t | count 0 &&
// RUN: grep "_Block_object_assign" %t | count 2

@interface NSDictionary @end

void test1(NSDictionary * dict) {
  ^{ (void)dict; }();
}
