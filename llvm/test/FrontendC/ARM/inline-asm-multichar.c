// RUN: %llvmgcc -S -march=armv7a %s 

int t1() {
  static float k = 1.0f;
// CHECK: call void asm sideeffect "flds s15, $0 \0A", "*^Uv,~{s15}"
  __asm__ volatile ("flds s15, %[k] \n" :: [k] "Uv,m" (k) : "s15");
  return 0;
}
