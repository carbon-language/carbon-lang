__attribute__((noinline))
static void NullDeref(int *ptr) {
  ptr[10]++;
}
int main() {
  NullDeref((int*)0);
}

// Check-Common: {{.*ERROR: AddressSanitizer crashed on unknown address}}
// Check-Common:   {{0x0*00028 .*pc 0x.*}}
// Check-Common: {{AddressSanitizer can not provide additional info. ABORTING}}

// atos on Mac cannot resolve the file:line info for frame 0 on the O1 level.
// It also can't extract the symbol name correctly.
// Check-Linux: {{    #0 0x.* in NullDeref.*null_deref.cc:3}}
// Check-Darwin: {{    #0 0x.* in .*NullDeref.*}}

// Check-Common: {{    #1 0x.* in main.*null_deref.cc:[67]}}
