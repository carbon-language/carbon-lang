// RUN: clang %s -verify -fsyntax-only

typedef void (* fp)(void);
void foo(void);
fp a[1] = { foo };

int myArray[5] = {1, 2, 3, 4, 5};
int *myPointer2 = myArray;
int *myPointer = &(myArray[2]);


extern int x;
void *g = &x;
int *h = &x;

int test() {
int a[10];
int b[10] = a; // expected-error {{initialization with "{...}" expected}}
}


// PR2050
struct cdiff_cmd {
          const char *name;
          unsigned short argc;
          int (*handler)();
};
int cdiff_cmd_open();
struct cdiff_cmd commands[] = {
        {"OPEN", 1, &cdiff_cmd_open }
};

