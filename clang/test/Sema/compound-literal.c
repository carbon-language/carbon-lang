// RUN: clang -fsyntax-only -verify %s

struct foo { int a, b; };

static struct foo t = (struct foo){0,0}; // -expected-error {{initializer element is not constant}}
static struct foo t2 = {0,0}; 
static struct foo t3 = t2; // -expected-error {{initializer element is not constant}}
static int *p = (int []){2,4}; 
static int x = (int){1}; // -expected-error {{initializer element is not constant}} -expected-warning{{braces around scalar initializer}}

extern void fooFunc(struct foo *pfoo);

int main(int argc, char **argv) {
 fooFunc(&(struct foo){ 1, 2 });
}


