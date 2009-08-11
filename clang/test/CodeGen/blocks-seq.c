// RUN: clang-cc -fblocks -triple x86_64-apple-darwin10 -emit-llvm -o %t %s &&
// RUN: grep '%call = call i32 (...)\* @rhs()' %t | count 1 &&
// If this fails, see about sliding %4, %5, %6 and %7...
// RUN: grep '%forwarding1 = getelementptr inbounds %0\* %i, i32 0, i32 1' %t | count 1 &&
// RUN: grep '%4 = bitcast i8\*\* %forwarding1 to %0\*\*' %t | count 1 &&
// RUN: grep '%5 = load %0\*\* %4' %t | count 1 &&
// RUN: grep '%call2 = call i32 (...)\* @rhs()' %t | count 1 &&
// RUN: grep '%forwarding3 = getelementptr inbounds %0\* %i, i32 0, i32 1' %t | count 1 &&
// RUN: grep '%6 = bitcast i8\*\* %forwarding3 to %0\*\*' %t | count 1 &&
// RUN: grep '%7 = load %0\*\* %6' %t | count 1

int rhs();

void foo() {
  __block int i;
  i = rhs();
  i += rhs();
}
