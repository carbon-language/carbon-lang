// RUN: %llvmgcc -xc %s -S -o - | not grep dead_function

extern __inline__ void dead_function() {}
