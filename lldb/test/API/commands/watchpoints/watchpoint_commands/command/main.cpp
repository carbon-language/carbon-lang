#include <stdio.h>
#include <stdint.h>

int32_t global = 0; // Watchpoint variable declaration.
int32_t cookie = 0;

static void modify(int32_t &var) {
    ++var;
}

int main(int argc, char** argv) {
    int local = 0;
    printf("&global=%p\n", &global);
    printf("about to write to 'global'...\n"); // Set break point at this line.
    for (int i = 0; i < 10; ++i)
        modify(global);

    printf("global=%d\n", global);
    printf("cookie=%d\n", cookie);
}
