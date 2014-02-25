//===- unittests/libclang/LibclangTest.cpp --- libclang tests -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang-c/Index.h"
#include "gtest/gtest.h"

TEST(libclang, clang_parseTranslationUnit2_InvalidArgs) {
  EXPECT_EQ(CXError_InvalidArguments,
            clang_parseTranslationUnit2(0, 0, 0, 0, 0, 0, 0, 0));
}

TEST(libclang, clang_createTranslationUnit_InvalidArgs) {
  EXPECT_EQ(0, clang_createTranslationUnit(0, 0));
}

TEST(libclang, clang_createTranslationUnit2_InvalidArgs) {
  EXPECT_EQ(CXError_InvalidArguments,
            clang_createTranslationUnit2(0, 0, 0));

  CXTranslationUnit TU = reinterpret_cast<CXTranslationUnit>(1);
  EXPECT_EQ(CXError_InvalidArguments,
            clang_createTranslationUnit2(0, 0, &TU));
  EXPECT_EQ(0, TU);
}

namespace {
struct TestVFO {
  const char *Contents;
  CXVirtualFileOverlay VFO;

  TestVFO(const char *Contents) : Contents(Contents) {
    VFO = clang_VirtualFileOverlay_create(0);
  }

  void map(const char *VPath, const char *RPath) {
    CXErrorCode Err = clang_VirtualFileOverlay_addFileMapping(VFO, VPath, RPath);
    EXPECT_EQ(Err, CXError_Success);
  }

  void mapError(const char *VPath, const char *RPath, CXErrorCode ExpErr) {
    CXErrorCode Err = clang_VirtualFileOverlay_addFileMapping(VFO, VPath, RPath);
    EXPECT_EQ(Err, ExpErr);
  }

  ~TestVFO() {
    if (!Contents)
      return;
    CXString Buf;
    clang_VirtualFileOverlay_writeToBuffer(VFO, 0, &Buf);
    EXPECT_STREQ(Contents, clang_getCString(Buf));
    clang_disposeString(Buf);
    clang_VirtualFileOverlay_dispose(VFO);
  }
};
}

TEST(libclang, VirtualFileOverlay) {
  {
    const char *contents =
    "{\n"
    "  'version': 0,\n"
    "  'roots': [\n"
    "    {\n"
    "      'type': 'directory',\n"
    "      'name': \"/path/virtual\",\n"
    "      'contents': [\n"
    "        {\n"
    "          'type': 'file',\n"
    "          'name': \"foo.h\",\n"
    "          'external-contents': \"/real/foo.h\"\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n";
    TestVFO T(contents);
    T.map("/path/virtual/foo.h", "/real/foo.h");
  }
  {
    TestVFO T(NULL);
    T.mapError("/path/./virtual/../foo.h", "/real/foo.h",
               CXError_InvalidArguments);
  }
  {
    const char *contents =
    "{\n"
    "  'version': 0,\n"
    "  'roots': [\n"
    "    {\n"
    "      'type': 'directory',\n"
    "      'name': \"/another/dir\",\n"
    "      'contents': [\n"
    "        {\n"
    "          'type': 'file',\n"
    "          'name': \"foo2.h\",\n"
    "          'external-contents': \"/real/foo2.h\"\n"
    "        }\n"
    "      ]\n"
    "    },\n"
    "    {\n"
    "      'type': 'directory',\n"
    "      'name': \"/path/virtual/dir\",\n"
    "      'contents': [\n"
    "        {\n"
    "          'type': 'file',\n"
    "          'name': \"foo1.h\",\n"
    "          'external-contents': \"/real/foo1.h\"\n"
    "        },\n"
    "        {\n"
    "          'type': 'file',\n"
    "          'name': \"foo3.h\",\n"
    "          'external-contents': \"/real/foo3.h\"\n"
    "        },\n"
    "        {\n"
    "          'type': 'directory',\n"
    "          'name': \"in/subdir\",\n"
    "          'contents': [\n"
    "            {\n"
    "              'type': 'file',\n"
    "              'name': \"foo4.h\",\n"
    "              'external-contents': \"/real/foo4.h\"\n"
    "            }\n"
    "          ]\n"
    "        }\n"
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}\n";
    TestVFO T(contents);
    T.map("/path/virtual/dir/foo1.h", "/real/foo1.h");
    T.map("/another/dir/foo2.h", "/real/foo2.h");
    T.map("/path/virtual/dir/foo3.h", "/real/foo3.h");
    T.map("/path/virtual/dir/in/subdir/foo4.h", "/real/foo4.h");
  }
}
