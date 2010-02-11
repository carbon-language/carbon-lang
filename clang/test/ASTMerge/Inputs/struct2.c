// Matches
struct S0 {
  int field1;
  float field2;
};

struct S0 x0;

// Mismatch in field type
struct S1 {
  int field1;
  float field2;
};

struct S1 x1;

// Mismatch in tag kind.
union S2 { int i; float f; } x2;

// Missing fields
struct S3 { int i; float f; } x3;

// Extra fields
struct S4 { int i; float f; } x4;

// Bit-field matches
struct S5 { int i : 8; unsigned j : 8; } x5;

// Bit-field mismatch
struct S6 { int i : 8; unsigned j; } x6;

// Bit-field mismatch
struct S7 { int i : 8; unsigned j : 16; } x7;
