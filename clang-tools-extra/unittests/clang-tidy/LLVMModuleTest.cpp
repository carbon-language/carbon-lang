#include "ClangTidyTest.h"

#include "llvm/LLVMTidyModule.h"

namespace clang {
namespace tidy {

typedef ClangTidyTest<NamespaceCommentCheck> NamespaceCommentCheckTest;

TEST_F(NamespaceCommentCheckTest, Basic) {
  EXPECT_EQ("namespace i {\n} // namespace i", runCheckOn("namespace i {\n}"));
}

} // namespace tidy
} // namespace clang
