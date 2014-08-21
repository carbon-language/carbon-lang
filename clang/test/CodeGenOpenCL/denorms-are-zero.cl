// RUN: %clang_cc1 -S -cl-denorms-are-zero -o - %s 2>&1

// This test just checks that the -cl-denorms-are-zero argument is accepted
// by clang.  This option is currently a no-op, which is allowed by the
// OpenCL specification.
