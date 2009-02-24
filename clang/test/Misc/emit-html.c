// RUN: clang %s -emit-html -o -

// rdar://6562329
#line 42 "foo.c"

// PR3635
#define F(fmt, ...) fmt, ## __VA_ARGS__
int main(int argc, char **argv) {
  return F(argc, 1);
}

