// RUN: %llvmgcc %s -S -g -o /dev/null
typedef void (*sigcatch_t)( struct sigcontext *);
sigcatch_t sigcatch[50] = {(sigcatch_t) 0};

