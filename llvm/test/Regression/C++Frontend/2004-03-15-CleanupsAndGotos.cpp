// Testcase from Bug 291
struct X {
  ~X();
};

void foo() {
  X v;
  
TryAgain: 
  goto TryAgain;
}
