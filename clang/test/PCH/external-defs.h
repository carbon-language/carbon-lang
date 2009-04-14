// Helper for external-defs.c test

// Tentative definitions
int x;
int x2;

// FIXME: check this, once we actually serialize it
int y = 17;

// Should not show up
static int z;

int incomplete_array[];
int incomplete_array2[];

// FIXME: CodeGen problems prevents this from working (<rdar://problem/6762287>)
// struct S s;
