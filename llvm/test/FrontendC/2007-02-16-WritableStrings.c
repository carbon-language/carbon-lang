// Test the -fwritable-strings option.

// RUN: %llvmgcc -O3 -S -o - -fwritable-strings %s | \
// RUN:    grep {internal unnamed_addr global}
// RUN: %llvmgcc -O3 -S -o - %s | grep {private unnamed_addr constant}

char *X = "foo";
