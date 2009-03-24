// RUN: clang-cc -E %s | grep '^   zzap$'

// zzap is on a new line, should be indented.
#define BLAH  zzap
   BLAH

