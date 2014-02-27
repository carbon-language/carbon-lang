#include "ClangTidyTest.h"
#include "llvm/LLVMTidyModule.h"
#include "gtest/gtest.h"

namespace clang {
namespace tidy {
namespace test {

TEST(NamespaceCommentCheckTest, Basic) {
  EXPECT_EQ("namespace i {\n} // namespace i",
            runCheckOnCode<NamespaceCommentCheck>("namespace i {\n}"));
}

} // namespace test
} // namespace tidy
} // namespace clang
