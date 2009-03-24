// RUN: clang-cc %s -emit-html -o -

// rdar://6562329
#line 42 "foo.c"

// PR3635
#define F(fmt, ...) fmt, ## __VA_ARGS__
int main(int argc, char **argv) {
  return F(argc, 1);
}

// PR3798
#define FOR_ALL_FILES(f,i) i

#if 0
  FOR_ALL_FILES(f) { }
#endif

