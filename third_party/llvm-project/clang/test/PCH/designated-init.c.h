static void *FooToken = &FooToken;
static void *FooTable[256] = {
    [0x3] = (void *[256]) { // 1
        [0x5b] = (void *[256]) { // 2
            [0x81] = (void *[256]) { // 3
                [0x42] = (void *[256]) { // 4
                    [0xa2] = (void *[256]) { // 5
                        [0xe] = (void *[256]) { // 6
                            [0x20] = (void *[256]) { // 7
                                [0xd7] = (void *[256]) { // 8
                                    [0x39] = (void *[256]) { // 9
                                        [0xf1] = (void *[256]) { // 10
                                            [0xa4] = (void *[256]) { // 11
                                                [0xa8] = (void *[256]) { // 12
                                                    [0x21] = (void *[256]) { // 13
                                                        [0x86] = (void *[256]) { // 14
                                                            [0x1d] = (void *[256]) { // 15
                                                                [0xdc] = (void *[256]) { // 16
                                                                    [0xa5] = (void *[256]) { // 17
                                                                        [0xef] = (void *[256]) { // 18
                                                                            [0x9] = (void *[256]) { // 19
                                                                                [0x34] = &FooToken,
                                                                            },
                                                                        },
                                                                    },
                                                                },
                                                            },
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }
};

struct P1 {
    struct Q1 {
      char a[6];
      char b[6];
    } q;
};

struct P1 l1 = {
    (struct Q1){ "foo", "bar" },
               .q.b = { "boo" },
               .q.b = { [1] = 'x' }
};

extern struct Q1 *foo();
static struct P1 test_foo() {
  struct P1 l = { *foo(),
                  .q.b = { "boo" },
                  .q.b = { [1] = 'x' }
                };
  return l;
}
