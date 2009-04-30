// RUN: clang-cc -triple=x86_64-apple-darwin9 -fnext-runtime -emit-llvm -o %t %s &&

// RUN: grep ji %t | count 1 &&
char *a = @encode(_Complex int);

// RUN: grep jf %t | count 1 &&
char *b = @encode(_Complex float);

// RUN: grep jd %t | count 1 &&
char *c = @encode(_Complex double);

// RUN: grep "t.00" %t | count 1 &&
char *e = @encode(__int128_t);

// RUN: grep "T.00" %t | count 1
char *f = @encode(__uint128_t);
