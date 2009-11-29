/* RUN: clang-cc -E %s -x c++ | grep block_1
   RUN: clang-cc -E %s -x c++ | not grep block_2
   RUN: clang-cc -E %s -x c | not grep block
*/

#if true
block_1
#endif

#if false
block_2
#endif

