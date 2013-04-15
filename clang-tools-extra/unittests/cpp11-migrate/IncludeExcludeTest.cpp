#include "Core/IncludeExcludeInfo.h"
#include "gtest/gtest.h"

IncludeExcludeInfo IEManager(/*include=*/ "a,b/b2,c/c2/c3",
                             /*exclude=*/ "a/af.cpp,a/a2,b/b2/b2f.cpp,c/c2/c3");

TEST(IncludeExcludeTest, NoMatchOnIncludeList) {
  // If the file does not appear on the include list then it is not safe to
  // transform. Files are not safe to transform by default.
  EXPECT_FALSE(IEManager.isFileIncluded("f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("b/dir/f.cpp"));
}

TEST(IncludeExcludeTest, MatchOnIncludeList) {
  // If the file appears on only the include list then it is safe to transform.
  EXPECT_TRUE(IEManager.isFileIncluded("a/f.cpp"));
  EXPECT_TRUE(IEManager.isFileIncluded("a/dir/f.cpp"));
  EXPECT_TRUE(IEManager.isFileIncluded("b/b2/f.cpp"));
}

TEST(IncludeExcludeTest, MatchOnBothLists) {
  // If the file appears on both the include or exclude list then it is not
  // safe to transform.
  EXPECT_FALSE(IEManager.isFileIncluded("a/af.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("a/a2/f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("a/a2/dir/f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("b/b2/b2f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("c/c2/c3/f.cpp"));
}
