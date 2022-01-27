volatile int x;

void __attribute__((noinline)) sink() {
  x++; //% self.filecheck("finish", "main.cpp", "-implicit-check-not=artificial")
  // CHECK: stop reason = step out
  // CHECK-NEXT: Stepped out past: frame #1: 0x{{[0-9a-f]+}} a.out`func3{{.*}} [artificial]
  // CHECK: frame #0: 0x{{[0-9a-f]+}} a.out`func2{{.*}}
}

void __attribute__((noinline)) func3() { sink(); /* tail */ }

void __attribute__((disable_tail_calls, noinline)) func2() { func3(); /* regular */ }

void __attribute__((noinline)) func1() { func2(); /* tail */ }

int __attribute__((disable_tail_calls)) main() {
  func1(); /* regular */
  return 0;
}
