__attribute__((noinline))
static void NullDeref(int *ptr) {
  ptr[10]++;
}
int main() {
  NullDeref((int*)0);
}

// CHECK: {{.*ERROR: AddressSanitizer crashed on unknown address 0x0*00028 .*pc 0x.*}}
// CHECK: {{AddressSanitizer can not provide additional info. ABORTING}}
// CHECK: {{    #0 0x.* in NullDeref.*null_deref.cc:3}}
// CHECK: {{    #1 0x.* in main.*null_deref.cc:[67]}}

// Darwin: {{.*ERROR: AddressSanitizer crashed on unknown address 0x0*00028 .*pc 0x.*}}
// Darwin: {{AddressSanitizer can not provide additional info. ABORTING}}
// atos cannot resolve the file:line info for frame 0 on the O1 level
// Darwin: {{    #0 0x.* in NullDeref.*}}
// Darwin: {{    #1 0x.* in main.*null_deref.cc:[67]}}
