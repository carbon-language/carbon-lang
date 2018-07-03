#include "Target.h"

#include <cassert>
#include <memory>

#include "MCTargetDesc/AArch64MCTargetDesc.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace exegesis {

void InitializeAArch64ExegesisTarget();

namespace {

using testing::Gt;
using testing::NotNull;
using testing::SizeIs;

class AArch64TargetTest : public ::testing::Test {
protected:
  AArch64TargetTest()
      : Target_(ExegesisTarget::lookup(llvm::Triple("aarch64-unknown-linux"))) {
    EXPECT_THAT(Target_, NotNull());
  }
  static void SetUpTestCase() { InitializeAArch64ExegesisTarget(); }

  const ExegesisTarget *const Target_;
};

TEST_F(AArch64TargetTest, SetRegToConstant) {
  // The AArch64 target currently doesn't know how to set register values
  const auto Insts = Target_->setRegToConstant(llvm::AArch64::X0);
  EXPECT_THAT(Insts, SizeIs(0));
}

} // namespace
} // namespace exegesis
