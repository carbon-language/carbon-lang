inline int *get_int_ptr(float *fp) {
  return fp;
}

#ifdef FATAL
void fatal(int);
void fatal(float);
#endif

// CHECK-FATAL: translation errors

// RUN: c-index-test -write-pch %t.pch -Werror %s
// RUN: not c-index-test -write-pch %t.pch -DFATAL -Werror %s 2>%t.err
// RUN: FileCheck -check-prefix=CHECK-FATAL %s < %t.err

