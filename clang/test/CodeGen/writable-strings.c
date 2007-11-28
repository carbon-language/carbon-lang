// RUN: clang -emit-llvm -fwritable-string %s

int main() {
    char *str = "abc";
    str[0] = '1';
    printf("%s", str);
}

