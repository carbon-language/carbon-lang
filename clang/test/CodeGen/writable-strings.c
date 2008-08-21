// RUN: clang -emit-llvm -o %t -fwritable-strings %s

int main() {
    char *str = "abc";
    str[0] = '1';
    printf("%s", str);
}

