// Test the -fwritable-strings option.

// RUN: %llvmgcc -O3 -S -o - -fwritable-strings %s | \
// RUN:    grep {internal global}
// RUN: %llvmgcc -O3 -S -o - %s | grep {private constant}

char *X = "foo";
