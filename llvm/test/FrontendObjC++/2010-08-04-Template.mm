// RUN: %llvmgcc %s -S
struct TRunSoon {
  template <class P1> static void Post() {}
};

@implementation TPrivsTableViewMainController
- (void) applyToEnclosed {
  TRunSoon::Post<int>();
}
@end
