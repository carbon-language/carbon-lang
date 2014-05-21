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
    if (Contents) {
      char *BufPtr;
      unsigned BufSize;
      clang_VirtualFileOverlay_writeToBuffer(VFO, 0, &BufPtr, &BufSize);
      std::string BufStr(BufPtr, BufSize);
      EXPECT_STREQ(Contents, BufStr.c_str());
      free(BufPtr);
    }
    clang_VirtualFileOverlay_dispose(VFO);
  }
};
}

TEST(libclang, VirtualFileOverlay_Basic) {
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

TEST(libclang, VirtualFileOverlay_Unicode) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/\\u266B\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"\\u2602.h\",\n"
      "          'external-contents': \"/real/\\u2602.h\"\n"
      "        }\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
  T.map("/path/♫/☂.h", "/real/☂.h");
}

TEST(libclang, VirtualFileOverlay_InvalidArgs) {
  TestVFO T(NULL);
  T.mapError("/path/./virtual/../foo.h", "/real/foo.h",
             CXError_InvalidArguments);
}

TEST(libclang, VirtualFileOverlay_RemapDirectories) {
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

TEST(libclang, VirtualFileOverlay_CaseInsensitive) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'case-sensitive': 'false',\n"
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
  clang_VirtualFileOverlay_setCaseSensitivity(T.VFO, false);
}

TEST(libclang, VirtualFileOverlay_SharedPrefix) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/foo\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"bar\",\n"
      "          'external-contents': \"/real/bar\"\n"
      "        },\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"bar.h\",\n"
      "          'external-contents': \"/real/bar.h\"\n"
      "        }\n"
      "      ]\n"
      "    },\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/foobar\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"baz.h\",\n"
      "          'external-contents': \"/real/baz.h\"\n"
      "        }\n"
      "      ]\n"
      "    },\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"foobarbaz.h\",\n"
      "          'external-contents': \"/real/foobarbaz.h\"\n"
      "        }\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
  T.map("/path/foo/bar.h", "/real/bar.h");
  T.map("/path/foo/bar", "/real/bar");
  T.map("/path/foobar/baz.h", "/real/baz.h");
  T.map("/path/foobarbaz.h", "/real/foobarbaz.h");
}

TEST(libclang, VirtualFileOverlay_AdjacentDirectory) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/dir1\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"foo.h\",\n"
      "          'external-contents': \"/real/foo.h\"\n"
      "        },\n"
      "        {\n"
      "          'type': 'directory',\n"
      "          'name': \"subdir\",\n"
      "          'contents': [\n"
      "            {\n"
      "              'type': 'file',\n"
      "              'name': \"bar.h\",\n"
      "              'external-contents': \"/real/bar.h\"\n"
      "            }\n"
      "          ]\n"
      "        }\n"
      "      ]\n"
      "    },\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/path/dir2\",\n"
      "      'contents': [\n"
      "        {\n"
      "          'type': 'file',\n"
      "          'name': \"baz.h\",\n"
      "          'external-contents': \"/real/baz.h\"\n"
      "        }\n"
      "      ]\n"
      "    }\n"
      "  ]\n"
      "}\n";
  TestVFO T(contents);
  T.map("/path/dir1/foo.h", "/real/foo.h");
  T.map("/path/dir1/subdir/bar.h", "/real/bar.h");
  T.map("/path/dir2/baz.h", "/real/baz.h");
}

TEST(libclang, VirtualFileOverlay_TopLevel) {
  const char *contents =
      "{\n"
      "  'version': 0,\n"
      "  'roots': [\n"
      "    {\n"
      "      'type': 'directory',\n"
      "      'name': \"/\",\n"
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
  T.map("/foo.h", "/real/foo.h");
}

TEST(libclang, ModuleMapDescriptor) {
  const char *Contents =
    "framework module TestFrame {\n"
    "  umbrella header \"TestFrame.h\"\n"
    "\n"
    "  export *\n"
    "  module * { export * }\n"
    "}\n";

  CXModuleMapDescriptor MMD = clang_ModuleMapDescriptor_create(0);

  clang_ModuleMapDescriptor_setFrameworkModuleName(MMD, "TestFrame");
  clang_ModuleMapDescriptor_setUmbrellaHeader(MMD, "TestFrame.h");

  char *BufPtr;
  unsigned BufSize;
  clang_ModuleMapDescriptor_writeToBuffer(MMD, 0, &BufPtr, &BufSize);
  std::string BufStr(BufPtr, BufSize);
  EXPECT_STREQ(Contents, BufStr.c_str());
  free(BufPtr);
  clang_ModuleMapDescriptor_dispose(MMD);
}
