// RUN: %clang_cc1 %s -cxx-abi itanium -emit-llvm -o - -fblocks 
// RUN: %clang_cc1 %s -cxx-abi microsoft -fno-rtti -emit-llvm -o - -fblocks 
// Just test that this doesn't crash the compiler...

void func(void*);

struct Test
{
  virtual void use() { func((void*)this); }
  Test(Test&c) { func((void*)this); }
  Test() { func((void*)this); }
};

void useBlock(void (^)(void));

int main (void) {
  __block Test t;
  useBlock(^(void) { t.use(); });
}

