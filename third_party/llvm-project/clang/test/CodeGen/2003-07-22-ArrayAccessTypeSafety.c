/* RUN: %clang_cc1  %s -emit-llvm -o - | grep -v alloca | not grep bitcast
 */

void test(int* array, long long N) {
    array[N] = N[array] = 33;
}

