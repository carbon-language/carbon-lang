//===- unittests/Lex/PPCallbacksTest.cpp - PPCallbacks tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===--------------------------------------------------------------===//

#include "clang/Lex/Preprocessor.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Parse/Parser.h"
#include "clang/Sema/Sema.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"
#include "gtest/gtest.h"

using namespace clang;

namespace {

// Stub to collect data from InclusionDirective callbacks.
class InclusionDirectiveCallbacks : public PPCallbacks {
public:
  void InclusionDirective(SourceLocation HashLoc, const Token &IncludeTok,
                          StringRef FileName, bool IsAngled,
                          CharSourceRange FilenameRange,
                          Optional<FileEntryRef> File, StringRef SearchPath,
                          StringRef RelativePath, const Module *Imported,
                          SrcMgr::CharacteristicKind FileType) override {
    this->HashLoc = HashLoc;
    this->IncludeTok = IncludeTok;
    this->FileName = FileName.str();
    this->IsAngled = IsAngled;
    this->FilenameRange = FilenameRange;
    this->File = File;
    this->SearchPath = SearchPath.str();
    this->RelativePath = RelativePath.str();
    this->Imported = Imported;
    this->FileType = FileType;
  }

  SourceLocation HashLoc;
  Token IncludeTok;
  SmallString<16> FileName;
  bool IsAngled;
  CharSourceRange FilenameRange;
  Optional<FileEntryRef> File;
  SmallString<16> SearchPath;
  SmallString<16> RelativePath;
  const Module* Imported;
  SrcMgr::CharacteristicKind FileType;
};

class CondDirectiveCallbacks : public PPCallbacks {
public:
  struct Result {
    SourceRange ConditionRange;
    ConditionValueKind ConditionValue;

    Result(SourceRange R, ConditionValueKind K)
        : ConditionRange(R), ConditionValue(K) {}
  };

  std::vector<Result> Results;

  void If(SourceLocation Loc, SourceRange ConditionRange,
          ConditionValueKind ConditionValue) override {
    Results.emplace_back(ConditionRange, ConditionValue);
  }

  void Elif(SourceLocation Loc, SourceRange ConditionRange,
            ConditionValueKind ConditionValue, SourceLocation IfLoc) override {
    Results.emplace_back(ConditionRange, ConditionValue);
  }
};

// Stub to collect data from PragmaOpenCLExtension callbacks.
class PragmaOpenCLExtensionCallbacks : public PPCallbacks {
public:
  typedef struct {
    SmallString<16> Name;
    unsigned State;
  } CallbackParameters;

  PragmaOpenCLExtensionCallbacks() : Name("Not called."), State(99) {}

  void PragmaOpenCLExtension(clang::SourceLocation NameLoc,
                             const clang::IdentifierInfo *Name,
                             clang::SourceLocation StateLoc,
                             unsigned State) override {
      this->NameLoc = NameLoc;
      this->Name = Name->getName();
      this->StateLoc = StateLoc;
      this->State = State;
  }

  SourceLocation NameLoc;
  SmallString<16> Name;
  SourceLocation StateLoc;
  unsigned State;
};

class PragmaMarkCallbacks : public PPCallbacks {
public:
  struct Mark {
    SourceLocation Location;
    std::string Trivia;
  };

  std::vector<Mark> Marks;

  void PragmaMark(SourceLocation Loc, StringRef Trivia) override {
    Marks.emplace_back(Mark{Loc, Trivia.str()});
  }
};

// PPCallbacks test fixture.
class PPCallbacksTest : public ::testing::Test {
protected:
  PPCallbacksTest()
      : InMemoryFileSystem(new llvm::vfs::InMemoryFileSystem),
        FileMgr(FileSystemOptions(), InMemoryFileSystem),
        DiagID(new DiagnosticIDs()), DiagOpts(new DiagnosticOptions()),
        Diags(DiagID, DiagOpts.get(), new IgnoringDiagConsumer()),
        SourceMgr(Diags, FileMgr), TargetOpts(new TargetOptions()) {
    TargetOpts->Triple = "x86_64-apple-darwin11.1.0";
    Target = TargetInfo::CreateTargetInfo(Diags, TargetOpts);
  }

  IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> InMemoryFileSystem;
  FileManager FileMgr;
  IntrusiveRefCntPtr<DiagnosticIDs> DiagID;
  IntrusiveRefCntPtr<DiagnosticOptions> DiagOpts;
  DiagnosticsEngine Diags;
  SourceManager SourceMgr;
  LangOptions LangOpts;
  std::shared_ptr<TargetOptions> TargetOpts;
  IntrusiveRefCntPtr<TargetInfo> Target;

  // Register a header path as a known file and add its location
  // to search path.
  void AddFakeHeader(HeaderSearch &HeaderInfo, const char *HeaderPath,
                     bool IsSystemHeader) {
    // Tell FileMgr about header.
    InMemoryFileSystem->addFile(HeaderPath, 0,
                                llvm::MemoryBuffer::getMemBuffer("\n"));

    // Add header's parent path to search path.
    StringRef SearchPath = llvm::sys::path::parent_path(HeaderPath);
    auto DE = FileMgr.getOptionalDirectoryRef(SearchPath);
    DirectoryLookup DL(*DE, SrcMgr::C_User, false);
    HeaderInfo.AddSearchPath(DL, IsSystemHeader);
  }

  // Get the raw source string of the range.
  StringRef GetSourceString(CharSourceRange Range) {
    const char* B = SourceMgr.getCharacterData(Range.getBegin());
    const char* E = SourceMgr.getCharacterData(Range.getEnd());

    return StringRef(B, E - B);
  }

  StringRef GetSourceStringToEnd(CharSourceRange Range) {
    const char *B = SourceMgr.getCharacterData(Range.getBegin());
    const char *E = SourceMgr.getCharacterData(Range.getEnd());

    return StringRef(
        B,
        E - B + Lexer::MeasureTokenLength(Range.getEnd(), SourceMgr, LangOpts));
  }

  // Run lexer over SourceText and collect FilenameRange from
  // the InclusionDirective callback.
  CharSourceRange InclusionDirectiveFilenameRange(const char *SourceText,
                                                  const char *HeaderPath,
                                                  bool SystemHeader) {
    std::unique_ptr<llvm::MemoryBuffer> Buf =
        llvm::MemoryBuffer::getMemBuffer(SourceText);
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));

    TrivialModuleLoader ModLoader;

    HeaderSearch HeaderInfo(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                            Diags, LangOpts, Target.get());
    AddFakeHeader(HeaderInfo, HeaderPath, SystemHeader);

    Preprocessor PP(std::make_shared<PreprocessorOptions>(), Diags, LangOpts,
                    SourceMgr, HeaderInfo, ModLoader,
                    /*IILookup =*/nullptr,
                    /*OwnsHeaderSearch =*/false);
    return InclusionDirectiveCallback(PP)->FilenameRange;
  }

  SrcMgr::CharacteristicKind InclusionDirectiveCharacteristicKind(
      const char *SourceText, const char *HeaderPath, bool SystemHeader) {
    std::unique_ptr<llvm::MemoryBuffer> Buf =
        llvm::MemoryBuffer::getMemBuffer(SourceText);
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));

    TrivialModuleLoader ModLoader;

    HeaderSearch HeaderInfo(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                            Diags, LangOpts, Target.get());
    AddFakeHeader(HeaderInfo, HeaderPath, SystemHeader);

    Preprocessor PP(std::make_shared<PreprocessorOptions>(), Diags, LangOpts,
                    SourceMgr, HeaderInfo, ModLoader,
                    /*IILookup =*/nullptr,
                    /*OwnsHeaderSearch =*/false);
    return InclusionDirectiveCallback(PP)->FileType;
  }

  InclusionDirectiveCallbacks *InclusionDirectiveCallback(Preprocessor &PP) {
    PP.Initialize(*Target);
    InclusionDirectiveCallbacks* Callbacks = new InclusionDirectiveCallbacks;
    PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(Callbacks));

    // Lex source text.
    PP.EnterMainSourceFile();

    while (true) {
      Token Tok;
      PP.Lex(Tok);
      if (Tok.is(tok::eof))
        break;
    }

    // Callbacks have been executed at this point -- return filename range.
    return Callbacks;
  }

  std::vector<CondDirectiveCallbacks::Result>
  DirectiveExprRange(StringRef SourceText) {
    TrivialModuleLoader ModLoader;
    std::unique_ptr<llvm::MemoryBuffer> Buf =
        llvm::MemoryBuffer::getMemBuffer(SourceText);
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(Buf)));
    HeaderSearch HeaderInfo(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                            Diags, LangOpts, Target.get());
    Preprocessor PP(std::make_shared<PreprocessorOptions>(), Diags, LangOpts,
                    SourceMgr, HeaderInfo, ModLoader,
                    /*IILookup =*/nullptr,
                    /*OwnsHeaderSearch =*/false);
    PP.Initialize(*Target);
    auto *Callbacks = new CondDirectiveCallbacks;
    PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(Callbacks));

    // Lex source text.
    PP.EnterMainSourceFile();

    while (true) {
      Token Tok;
      PP.Lex(Tok);
      if (Tok.is(tok::eof))
        break;
    }

    return Callbacks->Results;
  }

  std::vector<PragmaMarkCallbacks::Mark>
  PragmaMarkCall(const char *SourceText) {
    std::unique_ptr<llvm::MemoryBuffer> SourceBuf =
        llvm::MemoryBuffer::getMemBuffer(SourceText, "test.c");
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(SourceBuf)));

    HeaderSearch HeaderInfo(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                            Diags, LangOpts, Target.get());
    TrivialModuleLoader ModLoader;

    Preprocessor PP(std::make_shared<PreprocessorOptions>(), Diags, LangOpts,
                    SourceMgr, HeaderInfo, ModLoader, /*IILookup=*/nullptr,
                    /*OwnsHeaderSearch=*/false);
    PP.Initialize(*Target);

    auto *Callbacks = new PragmaMarkCallbacks;
    PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(Callbacks));

    // Lex source text.
    PP.EnterMainSourceFile();
    while (true) {
      Token Tok;
      PP.Lex(Tok);
      if (Tok.is(tok::eof))
        break;
    }

    return Callbacks->Marks;
  }

  PragmaOpenCLExtensionCallbacks::CallbackParameters
  PragmaOpenCLExtensionCall(const char *SourceText) {
    LangOptions OpenCLLangOpts;
    OpenCLLangOpts.OpenCL = 1;

    std::unique_ptr<llvm::MemoryBuffer> SourceBuf =
        llvm::MemoryBuffer::getMemBuffer(SourceText, "test.cl");
    SourceMgr.setMainFileID(SourceMgr.createFileID(std::move(SourceBuf)));

    TrivialModuleLoader ModLoader;
    HeaderSearch HeaderInfo(std::make_shared<HeaderSearchOptions>(), SourceMgr,
                            Diags, OpenCLLangOpts, Target.get());

    Preprocessor PP(std::make_shared<PreprocessorOptions>(), Diags,
                    OpenCLLangOpts, SourceMgr, HeaderInfo, ModLoader,
                    /*IILookup =*/nullptr,
                    /*OwnsHeaderSearch =*/false);
    PP.Initialize(*Target);

    // parser actually sets correct pragma handlers for preprocessor
    // according to LangOptions, so we init Parser to register opencl
    // pragma handlers
    ASTContext Context(OpenCLLangOpts, SourceMgr, PP.getIdentifierTable(),
                       PP.getSelectorTable(), PP.getBuiltinInfo(), PP.TUKind);
    Context.InitBuiltinTypes(*Target);

    ASTConsumer Consumer;
    Sema S(PP, Context, Consumer);
    Parser P(PP, S, false);
    PragmaOpenCLExtensionCallbacks* Callbacks = new PragmaOpenCLExtensionCallbacks;
    PP.addPPCallbacks(std::unique_ptr<PPCallbacks>(Callbacks));

    // Lex source text.
    PP.EnterMainSourceFile();
    while (true) {
      Token Tok;
      PP.Lex(Tok);
      if (Tok.is(tok::eof))
        break;
    }

    PragmaOpenCLExtensionCallbacks::CallbackParameters RetVal = {
      Callbacks->Name,
      Callbacks->State
    };
    return RetVal;
  }
};

TEST_F(PPCallbacksTest, UserFileCharacteristics) {
  const char *Source = "#include \"quoted.h\"\n";

  SrcMgr::CharacteristicKind Kind =
      InclusionDirectiveCharacteristicKind(Source, "/quoted.h", false);

  ASSERT_EQ(SrcMgr::CharacteristicKind::C_User, Kind);
}

TEST_F(PPCallbacksTest, QuotedFilename) {
  const char* Source =
    "#include \"quoted.h\"\n";

  CharSourceRange Range =
    InclusionDirectiveFilenameRange(Source, "/quoted.h", false);

  ASSERT_EQ("\"quoted.h\"", GetSourceString(Range));
}

TEST_F(PPCallbacksTest, AngledFilename) {
  const char* Source =
    "#include <angled.h>\n";

  CharSourceRange Range =
    InclusionDirectiveFilenameRange(Source, "/angled.h", true);

  ASSERT_EQ("<angled.h>", GetSourceString(Range));
}

TEST_F(PPCallbacksTest, QuotedInMacro) {
  const char* Source =
    "#define MACRO_QUOTED \"quoted.h\"\n"
    "#include MACRO_QUOTED\n";

  CharSourceRange Range =
    InclusionDirectiveFilenameRange(Source, "/quoted.h", false);

  ASSERT_EQ("\"quoted.h\"", GetSourceString(Range));
}

TEST_F(PPCallbacksTest, AngledInMacro) {
  const char* Source =
    "#define MACRO_ANGLED <angled.h>\n"
    "#include MACRO_ANGLED\n";

  CharSourceRange Range =
    InclusionDirectiveFilenameRange(Source, "/angled.h", true);

  ASSERT_EQ("<angled.h>", GetSourceString(Range));
}

TEST_F(PPCallbacksTest, StringizedMacroArgument) {
  const char* Source =
    "#define MACRO_STRINGIZED(x) #x\n"
    "#include MACRO_STRINGIZED(quoted.h)\n";

  CharSourceRange Range =
    InclusionDirectiveFilenameRange(Source, "/quoted.h", false);

  ASSERT_EQ("\"quoted.h\"", GetSourceString(Range));
}

TEST_F(PPCallbacksTest, ConcatenatedMacroArgument) {
  const char* Source =
    "#define MACRO_ANGLED <angled.h>\n"
    "#define MACRO_CONCAT(x, y) x ## _ ## y\n"
    "#include MACRO_CONCAT(MACRO, ANGLED)\n";

  CharSourceRange Range =
    InclusionDirectiveFilenameRange(Source, "/angled.h", false);

  ASSERT_EQ("<angled.h>", GetSourceString(Range));
}

TEST_F(PPCallbacksTest, TrigraphFilename) {
  const char* Source =
    "#include \"tri\?\?-graph.h\"\n";

  CharSourceRange Range =
    InclusionDirectiveFilenameRange(Source, "/tri~graph.h", false);

  ASSERT_EQ("\"tri\?\?-graph.h\"", GetSourceString(Range));
}

TEST_F(PPCallbacksTest, TrigraphInMacro) {
  const char* Source =
    "#define MACRO_TRIGRAPH \"tri\?\?-graph.h\"\n"
    "#include MACRO_TRIGRAPH\n";

  CharSourceRange Range =
    InclusionDirectiveFilenameRange(Source, "/tri~graph.h", false);

  ASSERT_EQ("\"tri\?\?-graph.h\"", GetSourceString(Range));
}

TEST_F(PPCallbacksTest, OpenCLExtensionPragmaEnabled) {
  const char* Source =
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";

  PragmaOpenCLExtensionCallbacks::CallbackParameters Parameters =
    PragmaOpenCLExtensionCall(Source);

  ASSERT_EQ("cl_khr_fp64", Parameters.Name);
  unsigned ExpectedState = 1;
  ASSERT_EQ(ExpectedState, Parameters.State);
}

TEST_F(PPCallbacksTest, OpenCLExtensionPragmaDisabled) {
  const char* Source =
    "#pragma OPENCL EXTENSION cl_khr_fp16 : disable\n";

  PragmaOpenCLExtensionCallbacks::CallbackParameters Parameters =
    PragmaOpenCLExtensionCall(Source);

  ASSERT_EQ("cl_khr_fp16", Parameters.Name);
  unsigned ExpectedState = 0;
  ASSERT_EQ(ExpectedState, Parameters.State);
}

TEST_F(PPCallbacksTest, CollectMarks) {
  const char *Source =
    "#pragma mark\n"
    "#pragma mark\r\n"
    "#pragma mark - trivia\n"
    "#pragma mark - trivia\r\n";

  auto Marks = PragmaMarkCall(Source);

  ASSERT_EQ(4u, Marks.size());
  ASSERT_TRUE(Marks[0].Trivia.empty());
  ASSERT_TRUE(Marks[1].Trivia.empty());
  ASSERT_FALSE(Marks[2].Trivia.empty());
  ASSERT_FALSE(Marks[3].Trivia.empty());
  ASSERT_EQ(" - trivia", Marks[2].Trivia);
  ASSERT_EQ(" - trivia", Marks[3].Trivia);
}

TEST_F(PPCallbacksTest, DirectiveExprRanges) {
  const auto &Results1 = DirectiveExprRange("#if FLUZZY_FLOOF\n#endif\n");
  EXPECT_EQ(Results1.size(), 1U);
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results1[0].ConditionRange, false)),
      "FLUZZY_FLOOF");

  const auto &Results2 = DirectiveExprRange("#if 1 + 4 < 7\n#endif\n");
  EXPECT_EQ(Results2.size(), 1U);
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results2[0].ConditionRange, false)),
      "1 + 4 < 7");

  const auto &Results3 = DirectiveExprRange("#if 1 + \\\n  2\n#endif\n");
  EXPECT_EQ(Results3.size(), 1U);
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results3[0].ConditionRange, false)),
      "1 + \\\n  2");

  const auto &Results4 = DirectiveExprRange("#if 0\n#elif FLOOFY\n#endif\n");
  EXPECT_EQ(Results4.size(), 2U);
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results4[0].ConditionRange, false)),
      "0");
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results4[1].ConditionRange, false)),
      "FLOOFY");

  const auto &Results5 = DirectiveExprRange("#if 1\n#elif FLOOFY\n#endif\n");
  EXPECT_EQ(Results5.size(), 2U);
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results5[0].ConditionRange, false)),
      "1");
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results5[1].ConditionRange, false)),
      "FLOOFY");

  const auto &Results6 =
      DirectiveExprRange("#if defined(FLUZZY_FLOOF)\n#endif\n");
  EXPECT_EQ(Results6.size(), 1U);
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results6[0].ConditionRange, false)),
      "defined(FLUZZY_FLOOF)");

  const auto &Results7 =
      DirectiveExprRange("#if 1\n#elif defined(FLOOFY)\n#endif\n");
  EXPECT_EQ(Results7.size(), 2U);
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results7[0].ConditionRange, false)),
      "1");
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results7[1].ConditionRange, false)),
      "defined(FLOOFY)");

  const auto &Results8 =
      DirectiveExprRange("#define FLOOFY 0\n#if __FILE__ > FLOOFY\n#endif\n");
  EXPECT_EQ(Results8.size(), 1U);
  EXPECT_EQ(
      GetSourceStringToEnd(CharSourceRange(Results8[0].ConditionRange, false)),
      "__FILE__ > FLOOFY");
  EXPECT_EQ(
      Lexer::getSourceText(CharSourceRange(Results8[0].ConditionRange, false),
                           SourceMgr, LangOpts),
      "__FILE__ > FLOOFY");
}

} // namespace
