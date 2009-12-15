// RUN: clang -cc1 -rewrite-objc %s -fblocks -o -

void I( void (^)(void));
void (^noop)(void);

void nothing();
int printf(const char*, ...);

typedef void (^T) (void);

void takeblock(T);
int takeintint(int (^C)(int)) { return C(4); }

T somefunction() {
  if (^{ })
    nothing();

  noop = ^{};

  noop = ^{printf("\nClosure\n"); };

  I(^{ });

  return ^{printf("\nClosure\n"); };
}
void test2() {
  int x = 4;

  takeblock(^{ printf("%d\n", x); });

  while (1) {
    takeblock(^{ 
        while(1) break;  // ok
      });
    break;
  }
}


void (^test3())(void) { 
  return ^{};
}

void test4() {
  void (^noop)(void) = ^{};
  void (*noop2)() = 0;
}

void myfunc(int (^block)(int)) {}

void myfunc3(const int *x);

void test5() {
  int a;

  myfunc(^(int abcd) {
      myfunc3(&a);
      return 1;
    });
}

void *X;

void test_arguments() {
  int y;
  int (^c)(char);
  (1 ? c : 0)('x');
  (1 ? 0 : c)('x');

  (1 ? c : c)('x');
}

static int global_x = 10;
void (^global_block)(void) = ^{ printf("global x is %d\n", global_x); };

typedef void (^void_block_t)(void);

static const void_block_t myBlock = ^{ };

static const void_block_t myBlock2 = ^ void(void) { }; 
