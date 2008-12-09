/*
 * Test that `llvmc -clang` can link multiple files
 *
 * RUN: llvmc -clang %s %p/test_data/clang_together.c -o %t
 * RUN: ./%t | grep {Hello Clang}
 */

void f();

int main() {
    f();
    return 0;
}
