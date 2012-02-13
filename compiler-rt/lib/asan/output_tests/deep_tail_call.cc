// Check-Common: AddressSanitizer global-buffer-overflow
int global[10];
// Check-Common: {{#0.*call4}}
void __attribute__((noinline)) call4(int i) { global[i+10]++; }
// Check-Common: {{#1.*call3}}
void __attribute__((noinline)) call3(int i) { call4(i); }
// Check-Common: {{#2.*call2}}
void __attribute__((noinline)) call2(int i) { call3(i); }
// Check-Common: {{#3.*call1}}
void __attribute__((noinline)) call1(int i) { call2(i); }
// Check-Common: {{#4.*main}}
int main(int argc, char **argv) {
  call1(argc);
  return global[0];
}
