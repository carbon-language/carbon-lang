// RUN: %llvmgcc %s -S -o -

// This struct is not 4 byte aligned becaues bit-field 
// type does not influence struct alignment.
struct U { char a; short b; int c:25; char d; } u;

