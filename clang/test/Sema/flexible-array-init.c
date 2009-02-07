// RUN: clang -fsyntax-only -verify %s
struct one {
  int a;
  int values[];
} x = {5, {1, 2, 3}};

struct one x2 = { 5, 1, 2, 3 }; // expected-error{{excess elements in struct initializer}}

void test() {
  struct one x3 = {5, {1, 2, 3}};
}

struct foo { 
  int x; 
  int y[]; // expected-note 3 {{initialized flexible array member 'y' is here}}
}; 
struct bar { struct foo z; };
     
struct foo a = { 1, { 2, 3, 4 } };        // Valid.
struct bar b = { { 1, { 2, 3, 4 } } };    // expected-error{{non-empty initialization of flexible array member inside subobject}}
struct bar c = { { 1, { } } };            // Valid.
struct foo d[1] = { { 1, { 2, 3, 4 } } };  // expected-error{{'struct foo' may not be used as an array element due to flexible array member}}

struct foo desig_foo = { .y = {2, 3, 4} };
struct bar desig_bar = { .z.y = { } };
struct bar desig_bar2 = { .z.y = { 2, 3, 4} }; // expected-error{{non-empty initialization of flexible array member inside subobject}}
struct foo design_foo2 = { .y = 2 }; // expected-error{{flexible array requires brace-enclosed initializer}}

struct point {
  int x, y;
};

struct polygon {
  int numpoints;
  struct point points[]; // expected-note{{initialized flexible array member 'points' is here}}
};
struct polygon poly = { 
  .points[2] = { 1, 2} }; // expected-error{{designator into flexible array member subobject}}
