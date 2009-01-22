// RUN: clang -fsyntax-only -verify %s

int complete_array_from_init[] = { 1, 2, [10] = 5, 1, 2, [5] = 2, 6 };

int complete_array_from_init_check[((sizeof(complete_array_from_init) / sizeof(int)) == 13)? 1 : -1];

int iarray[10] = {
  [0] = 1,
  [1 ... 5] = 2,
  [ 6 ... 6 ] = 3,
  [ 8 ... 7 ] = 4, // expected-error{{array designator range [8, 7] is empty}}
  [10] = 5,
  [-1] = 6 // expected-error{{array designator value '-1' is negative}}
};

int iarray2[10] = {
  [10] = 1, // expected-error{{array designator index (10) exceeds array bounds (10)}}
};

int iarray3[10] = {
  [5 ... 12] = 2 // expected-error{{array designator index (12) exceeds array bounds (10)}}
};

struct point {
  double x;
  double y;
};

struct point p1 = { 
  .y = 1.0, 
  x: 2.0,
  .a = 4.0, // expected-error{{field designator 'a' does not refer to any field in type 'struct point'}}
};

struct point p2 = {
  [1] = 1.0 // expected-error{{array designator cannot initialize non-array type}}
};

struct point array[10] = { 
  [0].x = 1.0,
  [1].y = 2.0,
  [2].z = 3.0, // expected-error{{field designator 'z' does not refer to any field in type 'struct point'}}
};

struct point array2[10] = {
  [10].x = 2.0, // expected-error{{array designator index (10) exceeds array bounds (10)}}
  [4 ... 5].y = 2.0,
  [4 ... 6] = { .x = 3, .y = 4.0 }
};

struct point array3[10] = {
  .x = 5 // expected-error{{field designator cannot initialize a non-struct, non-union type}}
};

struct rect {
  struct point top_left;
  struct point bottom_right;
};

struct rect window = { .top_left.x = 1.0 };

struct rect windows[] = {
  [2].top_left = { 1.0, 2.0 },
  [4].bottom_right = { .y = 1.0 },
  { { .y = 7.0, .x = 8.0 }, { .x = 5.0 } },
  [3] = { .top_left = { 1.1, 2.2 }, .bottom_right = { .y = 1.1 } }
};

int windows_size[((sizeof(windows) / sizeof(struct rect)) == 6)? 1 : -1];

struct rect windows_bad[3] = {
  [2].top_left = { { .x = 1.1 } }, // expected-error{{designator in initializer for scalar type}}
  [1].top_left = { .x = 1.1 }
};

struct gui {
  struct rect windows[10];
};

struct gui gui[] = {
  [5].windows[3].top_left.x = { 7.0 } // expected-warning{{braces around scalar initializer}}
};

struct translator {
  struct wonky { int * ptr; } wonky ;
  struct rect window;
  struct point offset;
} tran = {
  .window = { .top_left = { 1.0, 2.0 } },
  { .x = 5.0, .y = 6.0 },
  .wonky = { 0 }
};

int anint;
struct {int x,*y;} z[] = {[0].x = 2, &z[0].x};

struct outer { struct inner { int x, *y; } in, *inp; } zz[] = {
  [0].in.x = 2, &zz[0].in.x, &zz[0].in,
  0, &anint, &zz[1].in,
  [3].in = { .y = &anint, .x = 17 },
  [7].in.y = &anint, &zz[0].in,
  [4].in.y = &anint, [5].in.x = 12
};

int zz_sizecheck[sizeof(zz) / sizeof(struct outer) == 8? 1 : -1 ];

struct disklabel_ops {
  struct {} type;
  int labelsize;
};

struct disklabel_ops disklabel64_ops = {
  .labelsize = sizeof(struct disklabel_ops)
};
