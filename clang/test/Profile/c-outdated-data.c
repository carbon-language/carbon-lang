// Test that outdated data is ignored.

// RUN: %clang_cc1 -triple x86_64-apple-macosx10.9 -main-file-name c-outdated-data.c %s -o - -emit-llvm -fprofile-instr-use=%S/Inputs/c-outdated-data.profdata | FileCheck -check-prefix=PGOUSE %s

// TODO: We should have a warning or a remark that tells us the profile data was
// discarded, rather than just checking that we fail to add metadata.

// PGOUSE-LABEL: @no_usable_data()
void no_usable_data() {
  int i = 0;

  if (i) {}

#ifdef GENERATE_OUTDATED_DATA
  if (i) {}
#endif

  // PGOUSE-NOT: br {{.*}} !prof ![0-9]+
}

int main(int argc, const char *argv[]) {
  no_usable_data();
  return 0;
}
