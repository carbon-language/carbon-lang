/*
 * Test basic functionality of llvmc -clang
 *
 * RUN: llvmc -clang %s -o %t
 * RUN: ./%t | grep {Hello Clang}
 */

int main() {
    printf("Hello Clang world!\n");
    return 0;
}
