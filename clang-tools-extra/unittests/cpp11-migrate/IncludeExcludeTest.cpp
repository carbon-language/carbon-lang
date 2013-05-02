#include "Core/IncludeExcludeInfo.h"
#include "gtest/gtest.h"
#include "llvm/Support/Path.h"

TEST(IncludeExcludeTest, ParseString) {
  IncludeExcludeInfo IEManager;
  llvm::error_code Err = IEManager.readListFromString(
      /*include=*/ "a,b/b2,c/c2",
      /*exclude=*/ "a/af.cpp,a/a2,b/b2/b2f.cpp,c/c2");

  ASSERT_EQ(Err, llvm::error_code::success());

  // If the file does not appear on the include list then it is not safe to
  // transform. Files are not safe to transform by default.
  EXPECT_FALSE(IEManager.isFileIncluded("f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("b/dir/f.cpp"));

  // If the file appears on only the include list then it is safe to transform.
  EXPECT_TRUE(IEManager.isFileIncluded("a/f.cpp"));
  EXPECT_TRUE(IEManager.isFileIncluded("a/dir/f.cpp"));
  EXPECT_TRUE(IEManager.isFileIncluded("b/b2/f.cpp"));

  // If the file appears on both the include or exclude list then it is not
  // safe to transform.
  EXPECT_FALSE(IEManager.isFileIncluded("a/af.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("a/a2/f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("a/a2/dir/f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("b/b2/b2f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("c/c2/c3/f.cpp"));
}

// The IncludeExcludeTest suite requires data files. The location of these
// files must be provided in the 'DATADIR' environment variable.
class IncludeExcludeFileTest : public ::testing::Test {
public:
  virtual void SetUp() {
    DataDir = getenv("DATADIR");
    if (DataDir == 0) {
      FAIL()
          << "IncludeExcludeFileTest requires the DATADIR environment variable "
             "to be set.";
    }
  }

  const char *DataDir;
};

TEST_F(IncludeExcludeFileTest, UNIXFile) {
  llvm::SmallString<128> IncludeData(DataDir);
  llvm::SmallString<128> ExcludeData(IncludeData);
  llvm::sys::path::append(IncludeData, "IncludeData.in");
  llvm::sys::path::append(ExcludeData, "ExcludeData.in");

  IncludeExcludeInfo IEManager;
  llvm::error_code Err = IEManager.readListFromFile(IncludeData, ExcludeData);

  ASSERT_EQ(Err, llvm::error_code::success());

  EXPECT_FALSE(IEManager.isFileIncluded("f.cpp"));
  EXPECT_TRUE(IEManager.isFileIncluded("a/f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("a/af.cpp"));
}

TEST_F(IncludeExcludeFileTest, DOSFile) {
  llvm::SmallString<128> IncludeData(DataDir);
  llvm::SmallString<128> ExcludeData(IncludeData);
  llvm::sys::path::append(IncludeData, "IncludeDataCRLF.in");
  llvm::sys::path::append(ExcludeData, "ExcludeDataCRLF.in");

  IncludeExcludeInfo IEManager;
  llvm::error_code Err = IEManager.readListFromFile(IncludeData, ExcludeData);

  ASSERT_EQ(Err, llvm::error_code::success());

  EXPECT_FALSE(IEManager.isFileIncluded("f.cpp"));
  EXPECT_TRUE(IEManager.isFileIncluded("a/f.cpp"));
  EXPECT_FALSE(IEManager.isFileIncluded("a/af.cpp"));
}
