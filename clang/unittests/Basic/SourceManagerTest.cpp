//===- unittests/Basic/SourceManagerTest.cpp ------ SourceManager tests ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceManager.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/Preprocessor.h"
#include "llvm/Config/config.h"

#include "gtest/gtest.h"

using namespace llvm;
using namespace clang;

namespace {

// The test fixture.
class SourceManagerTest : public ::testing::Test {
protected:
  SourceManagerTest()
    : FileMgr(FileMgrOpts),
      DiagID(new DiagnosticIDs()),
      Diags(DiagID, new IgnoringDiagConsumer()),
      SourceMgr(Diags, FileMgr) {
    TargetOpts.Triple = "x86_64-apple-darwin11.1.0";
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
  }

  FileSystemOptions FileMgrOpts;
  FileManager FileMgr;
  llvm::IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  TargetOptions TargetOpts;
  llvm::IntrusiveRefCntPtr<TargetInfo> Target;
};

class VoidModuleLoader : public ModuleLoader {
  virtual Module *loadModule(SourceLocation ImportLoc, ModuleIdPath Path,
                             Module::NameVisibilityKind Visibility,
                             bool IsInclusionDirective) {
    return 0;
  }
};

TEST_F(SourceManagerTest, isBeforeInTranslationUnit) {
  const char *source =
    "#define M(x) [x]\n"
    "M(foo)";
  MemoryBuffer *buf = MemoryBuffer::getMemBuffer(source);
  FileID mainFileID = SourceMgr.createMainFileIDForMemBuffer(buf);

  VoidModuleLoader ModLoader;
  HeaderSearch HeaderInfo(FileMgr, Diags, LangOpts);
  Preprocessor PP(Diags, LangOpts,
                  Target.getPtr(),
                  SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup =*/ 0,
                  /*OwnsHeaderSearch =*/false,
                  /*DelayInitialization =*/ false);
  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  while (1) {
    Token tok;
    PP.Lex(tok);
    if (tok.is(tok::eof))
      break;
    toks.push_back(tok);
  }

  // Make sure we got the tokens that we expected.
  ASSERT_EQ(3U, toks.size());
  ASSERT_EQ(tok::l_square, toks[0].getKind());
  ASSERT_EQ(tok::identifier, toks[1].getKind());
  ASSERT_EQ(tok::r_square, toks[2].getKind());
  
  SourceLocation lsqrLoc = toks[0].getLocation();
  SourceLocation idLoc = toks[1].getLocation();
  SourceLocation rsqrLoc = toks[2].getLocation();
  
  SourceLocation macroExpStartLoc = SourceMgr.translateLineCol(mainFileID, 2, 1);
  SourceLocation macroExpEndLoc = SourceMgr.translateLineCol(mainFileID, 2, 6);
  ASSERT_TRUE(macroExpStartLoc.isFileID());
  ASSERT_TRUE(macroExpEndLoc.isFileID());

  llvm::SmallString<32> str;
  ASSERT_EQ("M", PP.getSpelling(macroExpStartLoc, str));
  ASSERT_EQ(")", PP.getSpelling(macroExpEndLoc, str));

  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(lsqrLoc, idLoc));
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(idLoc, rsqrLoc));
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(macroExpStartLoc, idLoc));
  EXPECT_TRUE(SourceMgr.isBeforeInTranslationUnit(idLoc, macroExpEndLoc));
}

#if defined(LLVM_ON_UNIX)

TEST_F(SourceManagerTest, getMacroArgExpandedLocation) {
  const char *header =
    "#define FM(x,y) x\n";

  const char *main =
    "#include \"/test-header.h\"\n"
    "#define VAL 0\n"
    "FM(VAL,0)\n"
    "FM(0,VAL)\n"
    "FM(FM(0,VAL),0)\n"
    "#define CONCAT(X, Y) X##Y\n"
    "CONCAT(1,1)\n";

  MemoryBuffer *headerBuf = MemoryBuffer::getMemBuffer(header);
  MemoryBuffer *mainBuf = MemoryBuffer::getMemBuffer(main);
  FileID mainFileID = SourceMgr.createMainFileIDForMemBuffer(mainBuf);

  const FileEntry *headerFile = FileMgr.getVirtualFile("/test-header.h",
                                                 headerBuf->getBufferSize(), 0);
  SourceMgr.overrideFileContents(headerFile, headerBuf);

  VoidModuleLoader ModLoader;
  HeaderSearch HeaderInfo(FileMgr, Diags, LangOpts);
  Preprocessor PP(Diags, LangOpts,
                  Target.getPtr(),
                  SourceMgr, HeaderInfo, ModLoader,
                  /*IILookup =*/ 0,
                  /*OwnsHeaderSearch =*/false,
                  /*DelayInitialization =*/ false);
  PP.EnterMainSourceFile();

  std::vector<Token> toks;
  while (1) {
    Token tok;
    PP.Lex(tok);
    if (tok.is(tok::eof))
      break;
    toks.push_back(tok);
  }

  // Make sure we got the tokens that we expected.
  ASSERT_EQ(4U, toks.size());
  ASSERT_EQ(tok::numeric_constant, toks[0].getKind());
  ASSERT_EQ(tok::numeric_constant, toks[1].getKind());
  ASSERT_EQ(tok::numeric_constant, toks[2].getKind());
  ASSERT_EQ(tok::numeric_constant, toks[3].getKind());

  SourceLocation defLoc = SourceMgr.translateLineCol(mainFileID, 2, 13);
  SourceLocation loc1 = SourceMgr.translateLineCol(mainFileID, 3, 8);
  SourceLocation loc2 = SourceMgr.translateLineCol(mainFileID, 4, 4);
  SourceLocation loc3 = SourceMgr.translateLineCol(mainFileID, 5, 7);
  SourceLocation defLoc2 = SourceMgr.translateLineCol(mainFileID, 6, 22);
  defLoc = SourceMgr.getMacroArgExpandedLocation(defLoc);
  loc1 = SourceMgr.getMacroArgExpandedLocation(loc1);
  loc2 = SourceMgr.getMacroArgExpandedLocation(loc2);
  loc3 = SourceMgr.getMacroArgExpandedLocation(loc3);
  defLoc2 = SourceMgr.getMacroArgExpandedLocation(defLoc2);

  EXPECT_TRUE(defLoc.isFileID());
  EXPECT_TRUE(loc1.isFileID());
  EXPECT_TRUE(SourceMgr.isMacroArgExpansion(loc2));
  EXPECT_TRUE(SourceMgr.isMacroArgExpansion(loc3));
  EXPECT_EQ(loc2, toks[1].getLocation());
  EXPECT_EQ(loc3, toks[2].getLocation());
  EXPECT_TRUE(defLoc2.isFileID());
}

#endif

} // anonymous namespace
