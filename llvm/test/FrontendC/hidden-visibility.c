// RUN: %llvmgcc %s -S -o - | grep {hidden global}

int X __attribute__ ((__visibility__ ("hidden"))) = 123;
