// Helper for external-defs.c test

// Tentative definitions
int x;
int x2;

// Definitions
int y = 17;
double d = 17.42;

// Should not show up
static int z;

int incomplete_array[];
int incomplete_array2[];

// FIXME: CodeGen problems prevents this from working (<rdar://problem/6762287>)
// struct S s;
