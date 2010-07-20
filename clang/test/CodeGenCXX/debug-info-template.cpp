// RUN: %clang_cc1 -emit-llvm-only -g -S %s -o - | grep "TC<int>"
template<typename T>
class TC {
public:
  TC(const TC &) {}
  TC() {}
};

TC<int> tci;
