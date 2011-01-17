/* RUN: %llvmgcc -w -x objective-c -S %s -o - | grep {__utf16_string_1} | grep {internal unnamed_addr constant} | grep {12 x i8}
   rdar://7095855 rdar://7115749 */

void *P = @"iPod™";

