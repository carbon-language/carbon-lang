// RUN: %llvmgcc %s -S -o - | grep {hidden unnamed_addr global}

int X __attribute__ ((__visibility__ ("hidden"))) = 123;
