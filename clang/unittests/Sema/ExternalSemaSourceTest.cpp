//=== unittests/Sema/ExternalSemaSourceTest.cpp - ExternalSemaSource tests ===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Parse/ParseAST.h"
#include "clang/Sema/ExternalSemaSource.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaDiagnostic.h"
#include "clang/Sema/TypoCorrection.h"
#include "clang/Tooling/Tooling.h"
#include "gtest/gtest.h"

using namespace clang;
using namespace clang::tooling;

namespace {

// \brief Counts the number of times MaybeDiagnoseMissingCompleteType
// is called. Returns the result it was provided on creation.
class CompleteTypeDiagnoser : public clang::ExternalSemaSource {
public:
  CompleteTypeDiagnoser(bool MockResult) : CallCount(0), Result(MockResult) {}

  bool MaybeDiagnoseMissingCompleteType(SourceLocation L, QualType T) override {
    ++CallCount;
    return Result;
  }

  int CallCount;
  bool Result;
};

// \brief Counts the number of err_using_directive_member_suggest diagnostics
// correcting from one namespace to another while still passing all diagnostics
// along a chain of consumers.
class NamespaceDiagnosticWatcher : public clang::DiagnosticConsumer {
  DiagnosticConsumer *Chained;
  std::string FromNS;
  std::string ToNS;

public:
  NamespaceDiagnosticWatcher(StringRef From, StringRef To)
      : Chained(nullptr), FromNS(From), ToNS("'"), SeenCount(0) {
    ToNS.append(To);
    ToNS.append("'");
  }

  void HandleDiagnostic(DiagnosticsEngine::Level DiagLevel,
                        const Diagnostic &Info) override {
    if (Chained)
      Chained->HandleDiagnostic(DiagLevel, Info);
    if (Info.getID() - 1 == diag::err_using_directive_member_suggest) {
      const IdentifierInfo *Ident = Info.getArgIdentifier(0);
      const std::string &CorrectedQuotedStr = Info.getArgStdStr(1);
      if (Ident->getName() == FromNS && CorrectedQuotedStr == ToNS)
        ++SeenCount;
    }
  }

  void clear() override {
    DiagnosticConsumer::clear();
    if (Chained)
      Chained->clear();
  }

  bool IncludeInDiagnosticCounts() const override {
    if (Chained)
      return Chained->IncludeInDiagnosticCounts();
    return false;
  }

  NamespaceDiagnosticWatcher *Chain(DiagnosticConsumer *ToChain) {
    Chained = ToChain;
    return this;
  }

  int SeenCount;
};

// \brief Always corrects a typo matching CorrectFrom with a new namespace
// with the name CorrectTo.
class NamespaceTypoProvider : public clang::ExternalSemaSource {
  std::string CorrectFrom;
  std::string CorrectTo;
  Sema *CurrentSema;

public:
  NamespaceTypoProvider(StringRef From, StringRef To)
      : CorrectFrom(From), CorrectTo(To), CurrentSema(nullptr), CallCount(0) {}

  void InitializeSema(Sema &S) override { CurrentSema = &S; }

  void ForgetSema() override { CurrentSema = nullptr; }

  TypoCorrection CorrectTypo(const DeclarationNameInfo &Typo, int LookupKind,
                             Scope *S, CXXScopeSpec *SS,
                             CorrectionCandidateCallback &CCC,
                             DeclContext *MemberContext, bool EnteringContext,
                             const ObjCObjectPointerType *OPT) override {
    ++CallCount;
    if (CurrentSema && Typo.getName().getAsString() == CorrectFrom) {
      DeclContext *DestContext = nullptr;
      ASTContext &Context = CurrentSema->getASTContext();
      if (SS)
        DestContext = CurrentSema->computeDeclContext(*SS, EnteringContext);
      if (!DestContext)
        DestContext = Context.getTranslationUnitDecl();
      IdentifierInfo *ToIdent =
          CurrentSema->getPreprocessor().getIdentifierInfo(CorrectTo);
      NamespaceDecl *NewNamespace =
          NamespaceDecl::Create(Context, DestContext, false, Typo.getBeginLoc(),
                                Typo.getLoc(), ToIdent, nullptr);
      DestContext->addDecl(NewNamespace);
      TypoCorrection Correction(ToIdent);
      Correction.addCorrectionDecl(NewNamespace);
      return Correction;
    }
    return TypoCorrection();
  }

  int CallCount;
};

// \brief Chains together a vector of NamespaceDiagnosticWatchers and
// adds a vector of ExternalSemaSources to the CompilerInstance before
// performing semantic analysis.
class ExternalSemaSourceInstaller : public clang::ASTFrontendAction {
  std::vector<NamespaceDiagnosticWatcher *> Watchers;
  std::vector<clang::ExternalSemaSource *> Sources;
  std::unique_ptr<DiagnosticConsumer> OwnedClient;

protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef /* dummy */) override {
    return llvm::make_unique<clang::ASTConsumer>();
  }

  void ExecuteAction() override {
    CompilerInstance &CI = getCompilerInstance();
    ASSERT_FALSE(CI.hasSema());
    CI.createSema(getTranslationUnitKind(), nullptr);
    ASSERT_TRUE(CI.hasDiagnostics());
    DiagnosticsEngine &Diagnostics = CI.getDiagnostics();
    DiagnosticConsumer *Client = Diagnostics.getClient();
    if (Diagnostics.ownsClient())
      OwnedClient = Diagnostics.takeClient();
    for (size_t I = 0, E = Watchers.size(); I < E; ++I)
      Client = Watchers[I]->Chain(Client);
    Diagnostics.setClient(Client, false);
    for (size_t I = 0, E = Sources.size(); I < E; ++I) {
      Sources[I]->InitializeSema(CI.getSema());
      CI.getSema().addExternalSource(Sources[I]);
    }
    ParseAST(CI.getSema(), CI.getFrontendOpts().ShowStats,
             CI.getFrontendOpts().SkipFunctionBodies);
  }

public:
  void PushSource(clang::ExternalSemaSource *Source) {
    Sources.push_back(Source);
  }

  void PushWatcher(NamespaceDiagnosticWatcher *Watcher) {
    Watchers.push_back(Watcher);
  }
};

// Make sure that the NamespaceDiagnosticWatcher is not miscounting.
TEST(ExternalSemaSource, SanityCheck) {
  std::unique_ptr<ExternalSemaSourceInstaller> Installer(
      new ExternalSemaSourceInstaller);
  NamespaceDiagnosticWatcher Watcher("AAB", "BBB");
  Installer->PushWatcher(&Watcher);
  std::vector<std::string> Args(1, "-std=c++11");
  ASSERT_TRUE(clang::tooling::runToolOnCodeWithArgs(
      Installer.release(), "namespace AAA { } using namespace AAB;", Args));
  ASSERT_EQ(0, Watcher.SeenCount);
}

// Check that when we add a NamespaceTypeProvider, we use that suggestion
// instead of the usual suggestion we would use above.
TEST(ExternalSemaSource, ExternalTypoCorrectionPrioritized) {
  std::unique_ptr<ExternalSemaSourceInstaller> Installer(
      new ExternalSemaSourceInstaller);
  NamespaceTypoProvider Provider("AAB", "BBB");
  NamespaceDiagnosticWatcher Watcher("AAB", "BBB");
  Installer->PushSource(&Provider);
  Installer->PushWatcher(&Watcher);
  std::vector<std::string> Args(1, "-std=c++11");
  ASSERT_TRUE(clang::tooling::runToolOnCodeWithArgs(
      Installer.release(), "namespace AAA { } using namespace AAB;", Args));
  ASSERT_LE(0, Provider.CallCount);
  ASSERT_EQ(1, Watcher.SeenCount);
}

// Check that we use the first successful TypoCorrection returned from an
// ExternalSemaSource.
TEST(ExternalSemaSource, ExternalTypoCorrectionOrdering) {
  std::unique_ptr<ExternalSemaSourceInstaller> Installer(
      new ExternalSemaSourceInstaller);
  NamespaceTypoProvider First("XXX", "BBB");
  NamespaceTypoProvider Second("AAB", "CCC");
  NamespaceTypoProvider Third("AAB", "DDD");
  NamespaceDiagnosticWatcher Watcher("AAB", "CCC");
  Installer->PushSource(&First);
  Installer->PushSource(&Second);
  Installer->PushSource(&Third);
  Installer->PushWatcher(&Watcher);
  std::vector<std::string> Args(1, "-std=c++11");
  ASSERT_TRUE(clang::tooling::runToolOnCodeWithArgs(
      Installer.release(), "namespace AAA { } using namespace AAB;", Args));
  ASSERT_LE(1, First.CallCount);
  ASSERT_LE(1, Second.CallCount);
  ASSERT_EQ(0, Third.CallCount);
  ASSERT_EQ(1, Watcher.SeenCount);
}

// We should only try MaybeDiagnoseMissingCompleteType if we can't otherwise
// solve the problem.
TEST(ExternalSemaSource, TryOtherTacticsBeforeDiagnosing) {
  std::unique_ptr<ExternalSemaSourceInstaller> Installer(
      new ExternalSemaSourceInstaller);
  CompleteTypeDiagnoser Diagnoser(false);
  Installer->PushSource(&Diagnoser);
  std::vector<std::string> Args(1, "-std=c++11");
  // This code hits the class template specialization/class member of a class
  // template specialization checks in Sema::RequireCompleteTypeImpl.
  ASSERT_TRUE(clang::tooling::runToolOnCodeWithArgs(
      Installer.release(),
      "template <typename T> struct S { class C { }; }; S<char>::C SCInst;",
      Args));
  ASSERT_EQ(0, Diagnoser.CallCount);
}

// The first ExternalSemaSource where MaybeDiagnoseMissingCompleteType returns
// true should be the last one called.
TEST(ExternalSemaSource, FirstDiagnoserTaken) {
  std::unique_ptr<ExternalSemaSourceInstaller> Installer(
      new ExternalSemaSourceInstaller);
  CompleteTypeDiagnoser First(false);
  CompleteTypeDiagnoser Second(true);
  CompleteTypeDiagnoser Third(true);
  Installer->PushSource(&First);
  Installer->PushSource(&Second);
  Installer->PushSource(&Third);
  std::vector<std::string> Args(1, "-std=c++11");
  ASSERT_FALSE(clang::tooling::runToolOnCodeWithArgs(
      Installer.release(), "class Incomplete; Incomplete IncompleteInstance;",
      Args));
  ASSERT_EQ(1, First.CallCount);
  ASSERT_EQ(1, Second.CallCount);
  ASSERT_EQ(0, Third.CallCount);
}

} // anonymous namespace
