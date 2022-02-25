#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/CodeGen/ObjectFilePCHContainerOperations.h"
#include "clang/Frontend/ASTConsumers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Rewrite/Core/Rewriter.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>
#include <string>

using namespace clang;
using namespace clang::driver;
using namespace clang::tooling;

static llvm::cl::OptionCategory InstrCategory("LLDB Instrumentation Generator");

/// Get the macro name for recording method calls.
///
/// LLDB_RECORD_METHOD
/// LLDB_RECORD_METHOD_CONST
/// LLDB_RECORD_METHOD_NO_ARGS
/// LLDB_RECORD_METHOD_CONST_NO_ARGS
/// LLDB_RECORD_STATIC_METHOD
/// LLDB_RECORD_STATIC_METHOD_NO_ARGS
static std::string GetRecordMethodMacroName(bool Static, bool Const,
                                            bool NoArgs) {
  std::string Macro;
  llvm::raw_string_ostream OS(Macro);

  OS << "LLDB_RECORD";
  if (Static)
    OS << "_STATIC";
  OS << "_METHOD";
  if (Const)
    OS << "_CONST";
  if (NoArgs)
    OS << "_NO_ARGS";

  return OS.str();
}

/// Get the macro name for register methods.
///
/// LLDB_REGISTER_CONSTRUCTOR
/// LLDB_REGISTER_METHOD
/// LLDB_REGISTER_METHOD_CONST
/// LLDB_REGISTER_STATIC_METHOD
static std::string GetRegisterMethodMacroName(bool Static, bool Const) {
  std::string Macro;
  llvm::raw_string_ostream OS(Macro);

  OS << "LLDB_REGISTER";
  if (Static)
    OS << "_STATIC";
  OS << "_METHOD";
  if (Const)
    OS << "_CONST";

  return OS.str();
}

static std::string GetRecordMethodMacro(StringRef Result, StringRef Class,
                                        StringRef Method, StringRef Signature,
                                        StringRef Values, bool Static,
                                        bool Const) {
  std::string Macro;
  llvm::raw_string_ostream OS(Macro);

  OS << GetRecordMethodMacroName(Static, Const, Values.empty());
  OS << "(" << Result << ", " << Class << ", " << Method;

  if (!Values.empty()) {
    OS << ", (" << Signature << "), " << Values << ");\n\n";
  } else {
    OS << ");\n\n";
  }

  return OS.str();
}

static std::string GetRecordConstructorMacro(StringRef Class,
                                             StringRef Signature,
                                             StringRef Values) {
  std::string Macro;
  llvm::raw_string_ostream OS(Macro);
  if (!Values.empty()) {
    OS << "LLDB_RECORD_CONSTRUCTOR(" << Class << ", (" << Signature << "), "
       << Values << ");\n\n";
  } else {
    OS << "LLDB_RECORD_CONSTRUCTOR_NO_ARGS(" << Class << ");\n\n";
  }
  return OS.str();
}

static std::string GetRecordDummyMacro(StringRef Result, StringRef Class,
                                       StringRef Method, StringRef Signature,
                                       StringRef Values) {
  assert(!Values.empty());
  std::string Macro;
  llvm::raw_string_ostream OS(Macro);

  OS << "LLDB_RECORD_DUMMY(" << Result << ", " << Class << ", " << Method;
  OS << ", (" << Signature << "), " << Values << ");\n\n";

  return OS.str();
}

static std::string GetRegisterConstructorMacro(StringRef Class,
                                               StringRef Signature) {
  std::string Macro;
  llvm::raw_string_ostream OS(Macro);
  OS << "LLDB_REGISTER_CONSTRUCTOR(" << Class << ", (" << Signature << "));\n";
  return OS.str();
}

static std::string GetRegisterMethodMacro(StringRef Result, StringRef Class,
                                          StringRef Method, StringRef Signature,
                                          bool Static, bool Const) {
  std::string Macro;
  llvm::raw_string_ostream OS(Macro);
  OS << GetRegisterMethodMacroName(Static, Const);
  OS << "(" << Result << ", " << Class << ", " << Method << ", (" << Signature
     << "));\n";
  return OS.str();
}

class SBReturnVisitor : public RecursiveASTVisitor<SBReturnVisitor> {
public:
  SBReturnVisitor(Rewriter &R) : MyRewriter(R) {}

  bool VisitReturnStmt(ReturnStmt *Stmt) {
    Expr *E = Stmt->getRetValue();

    if (E->getBeginLoc().isMacroID())
      return false;

    SourceRange R(E->getBeginLoc(), E->getEndLoc());

    StringRef WrittenExpr = Lexer::getSourceText(
        CharSourceRange::getTokenRange(R), MyRewriter.getSourceMgr(),
        MyRewriter.getLangOpts());

    std::string ReplacementText =
        "LLDB_RECORD_RESULT(" + WrittenExpr.str() + ")";
    MyRewriter.ReplaceText(R, ReplacementText);

    return true;
  }

private:
  Rewriter &MyRewriter;
};

class SBVisitor : public RecursiveASTVisitor<SBVisitor> {
public:
  SBVisitor(Rewriter &R, ASTContext &Context)
      : MyRewriter(R), Context(Context) {}

  bool VisitCXXMethodDecl(CXXMethodDecl *Decl) {
    // Not all decls should be registered. Please refer to that method's
    // comment for details.
    if (ShouldSkip(Decl))
      return false;

    // Skip CXXMethodDecls that already starts with a macro. This should make
    // it easier to rerun the tool to find missing macros.
    Stmt *Body = Decl->getBody();
    for (auto &C : Body->children()) {
      if (C->getBeginLoc().isMacroID())
        return false;
      break;
    }

    // Print 'bool' instead of '_Bool'.
    PrintingPolicy Policy(Context.getLangOpts());
    Policy.Bool = true;

    // Unsupported signatures get a dummy macro.
    bool ShouldInsertDummy = false;

    // Collect the functions parameter types and names.
    std::vector<std::string> ParamTypes;
    std::vector<std::string> ParamNames;
    for (auto *P : Decl->parameters()) {
      QualType T = P->getType();
      ParamTypes.push_back(T.getAsString(Policy));
      ParamNames.push_back(P->getNameAsString());

      // Currently we don't support functions that have function pointers as an
      // argument, in which case we insert a dummy macro.
      ShouldInsertDummy |= T->isFunctionPointerType();
    }

    // Convert the two lists to string for the macros.
    std::string ParamTypesStr = llvm::join(ParamTypes, ", ");
    std::string ParamNamesStr = llvm::join(ParamNames, ", ");

    CXXRecordDecl *Record = Decl->getParent();
    QualType ReturnType = Decl->getReturnType();

    // Construct the macros.
    std::string Macro;
    if (ShouldInsertDummy) {
      // Don't insert a register call for dummy macros.
      Macro = GetRecordDummyMacro(
          ReturnType.getAsString(Policy), Record->getNameAsString(),
          Decl->getNameAsString(), ParamTypesStr, ParamNamesStr);

    } else if (isa<CXXConstructorDecl>(Decl)) {
      llvm::outs() << GetRegisterConstructorMacro(Record->getNameAsString(),
                                                  ParamTypesStr);

      Macro = GetRecordConstructorMacro(Record->getNameAsString(),
                                        ParamTypesStr, ParamNamesStr);
    } else {
      llvm::outs() << GetRegisterMethodMacro(
          ReturnType.getAsString(Policy), Record->getNameAsString(),
          Decl->getNameAsString(), ParamTypesStr, Decl->isStatic(),
          Decl->isConst());

      Macro = GetRecordMethodMacro(
          ReturnType.getAsString(Policy), Record->getNameAsString(),
          Decl->getNameAsString(), ParamTypesStr, ParamNamesStr,
          Decl->isStatic(), Decl->isConst());
    }

    // Insert the macro at the beginning of the function. We don't attempt to
    // fix the formatting and instead rely on clang-format to fix it after the
    // tool has run. This is also the reason that the macros end with two
    // newlines, counting on clang-format to normalize this in case the macro
    // got inserted before an existing newline.
    SourceLocation InsertLoc = Lexer::getLocForEndOfToken(
        Body->getBeginLoc(), 0, MyRewriter.getSourceMgr(),
        MyRewriter.getLangOpts());
    MyRewriter.InsertTextAfter(InsertLoc, Macro);

    // If the function returns a class or struct, we need to wrap its return
    // statement(s).
    bool ShouldRecordResult = ReturnType->isStructureOrClassType() ||
                              ReturnType->getPointeeCXXRecordDecl();
    if (!ShouldInsertDummy && ShouldRecordResult) {
      SBReturnVisitor Visitor(MyRewriter);
      Visitor.TraverseDecl(Decl);
    }

    return true;
  }

private:
  /// Determine whether we need to consider the given CXXMethodDecl.
  ///
  /// Currently we skip the following cases:
  ///  1. Decls outside the main source file,
  ///  2. Decls that are only present in the source file,
  ///  3. Decls that are not definitions,
  ///  4. Non-public methods,
  ///  5. Variadic methods.
  ///  6. Destructors.
  bool ShouldSkip(CXXMethodDecl *Decl) {
    // Skip anything outside the main file.
    if (!MyRewriter.getSourceMgr().isInMainFile(Decl->getBeginLoc()))
      return true;

    // Skip if the canonical decl in the current decl. It means that the method
    // is declared in the implementation and is therefore not exposed as part
    // of the API.
    if (Decl == Decl->getCanonicalDecl())
      return true;

    // Skip decls that have no body, i.e. are just declarations.
    Stmt *Body = Decl->getBody();
    if (!Body)
      return true;

    // Skip non-public methods.
    AccessSpecifier AS = Decl->getAccess();
    if (AS != AccessSpecifier::AS_public)
      return true;

    // Skip variadic methods.
    if (Decl->isVariadic())
      return true;

    // Skip destructors.
    if (isa<CXXDestructorDecl>(Decl))
      return true;

    return false;
  }

  Rewriter &MyRewriter;
  ASTContext &Context;
};

class SBConsumer : public ASTConsumer {
public:
  SBConsumer(Rewriter &R, ASTContext &Context) : Visitor(R, Context) {}

  // Override the method that gets called for each parsed top-level
  // declaration.
  bool HandleTopLevelDecl(DeclGroupRef DR) override {
    for (DeclGroupRef::iterator b = DR.begin(), e = DR.end(); b != e; ++b) {
      Visitor.TraverseDecl(*b);
    }
    return true;
  }

private:
  SBVisitor Visitor;
};

class SBAction : public ASTFrontendAction {
public:
  SBAction() = default;

  bool BeginSourceFileAction(CompilerInstance &CI) override {
    llvm::outs() << "{\n";
    return true;
  }

  void EndSourceFileAction() override {
    llvm::outs() << "}\n";
    MyRewriter.overwriteChangedFiles();
  }

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                 StringRef File) override {
    MyRewriter.setSourceMgr(CI.getSourceManager(), CI.getLangOpts());
    return std::make_unique<SBConsumer>(MyRewriter, CI.getASTContext());
  }

private:
  Rewriter MyRewriter;
};

int main(int argc, const char **argv) {
  auto ExpectedParser = CommonOptionsParser::create(
      argc, argv, InstrCategory, llvm::cl::OneOrMore,
      "Utility for generating the macros for LLDB's "
      "instrumentation framework.");
  if (!ExpectedParser) {
    llvm::errs() << ExpectedParser.takeError();
    return 1;
  }
  CommonOptionsParser &OP = ExpectedParser.get();

  auto PCHOpts = std::make_shared<PCHContainerOperations>();
  PCHOpts->registerWriter(std::make_unique<ObjectFilePCHContainerWriter>());
  PCHOpts->registerReader(std::make_unique<ObjectFilePCHContainerReader>());

  ClangTool T(OP.getCompilations(), OP.getSourcePathList(), PCHOpts);
  return T.run(newFrontendActionFactory<SBAction>().get());
}
