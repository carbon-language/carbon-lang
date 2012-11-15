// Check that we can operate on files from /dev/fd.
// REQUIRES: dev-fd-fs


// Check reading from named pipes. We cat the input here instead of redirecting
// it to ensure that /dev/fd/0 is a named pipe, not just a redirected file.
//
// RUN: cat %s | %clang -x c /dev/fd/0 -E > %t
// RUN: FileCheck --check-prefix DEV-FD-INPUT < %t %s

// DEV-FD-INPUT: int x;
int x;
