const char *kAsanDefaultOptions="verbosity=1 foo=bar";

extern "C"
__attribute__((no_address_safety_analysis))
const char *__asan_default_options() {
  return kAsanDefaultOptions;
}

int main() {
  // Check-Common: foo=bar
  return 0;
}
