// RUN: %llvmgcc %s -S -emit-llvm
struct TRunSoon {
  template <class P1> static void Post() {}
};

@implementation TPrivsTableViewMainController
- (void) applyToEnclosed {
  TRunSoon::Post<int>();
}
@end
