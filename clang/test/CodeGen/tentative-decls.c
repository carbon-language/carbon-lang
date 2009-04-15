// RUN: clang-cc -emit-llvm -o %t %s &&

// RUN: grep '@r = common global \[1 x .*\] zeroinitializer' %t &&

int r[];
int (*a)[] = &r;

struct s0;
struct s0 x;
// RUN: grep '@x = common global .struct.s0 zeroinitializer' %t &&

struct s0 y;
// RUN: grep '@y = common global .struct.s0 zeroinitializer' %t &&
struct s0 *f0() {
  return &y;
}

struct s0 {
  int x;
};

// RUN: grep '@b = common global \[1 x .*\] zeroinitializer' %t &&
int b[];
int *f1() {
  return b;
}

// RUN: true
