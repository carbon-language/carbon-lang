//=- LocalizationChecker.cpp -------------------------------------*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a set of checks for localizability including:
//  1) A checker that warns about uses of non-localized NSStrings passed to
//     UI methods expecting localized strings
//  2) A syntactic checker that warns against the bad practice of
//     not including a comment in NSLocalizedString macros.
//
//===----------------------------------------------------------------------===//

#include "ClangSACheckers.h"
#include "SelectorExtras.h"
#include "clang/AST/Attr.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclObjC.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/Lex/Lexer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/StmtVisitor.h"
#include "llvm/Support/Unicode.h"
#include "llvm/ADT/StringSet.h"

using namespace clang;
using namespace ento;

namespace {
struct LocalizedState {
private:
  enum Kind { NonLocalized, Localized } K;
  LocalizedState(Kind InK) : K(InK) {}

public:
  bool isLocalized() const { return K == Localized; }
  bool isNonLocalized() const { return K == NonLocalized; }

  static LocalizedState getLocalized() { return LocalizedState(Localized); }
  static LocalizedState getNonLocalized() {
    return LocalizedState(NonLocalized);
  }

  // Overload the == operator
  bool operator==(const LocalizedState &X) const { return K == X.K; }

  // LLVMs equivalent of a hash function
  void Profile(llvm::FoldingSetNodeID &ID) const { ID.AddInteger(K); }
};

class NonLocalizedStringChecker
    : public Checker<check::PostCall, check::PreObjCMessage,
                     check::PostObjCMessage,
                     check::PostStmt<ObjCStringLiteral>> {

  mutable std::unique_ptr<BugType> BT;

  // Methods that require a localized string
  mutable llvm::StringMap<llvm::StringMap<uint8_t>> UIMethods;
  // Methods that return a localized string
  mutable llvm::SmallSet<std::pair<StringRef, StringRef>, 12> LSM;
  // C Functions that return a localized string
  mutable llvm::StringSet<> LSF;

  void initUIMethods(ASTContext &Ctx) const;
  void initLocStringsMethods(ASTContext &Ctx) const;

  bool hasNonLocalizedState(SVal S, CheckerContext &C) const;
  bool hasLocalizedState(SVal S, CheckerContext &C) const;
  void setNonLocalizedState(SVal S, CheckerContext &C) const;
  void setLocalizedState(SVal S, CheckerContext &C) const;

  bool isAnnotatedAsLocalized(const Decl *D) const;
  void reportLocalizationError(SVal S, const ObjCMethodCall &M,
                               CheckerContext &C, int argumentNumber = 0) const;

public:
  NonLocalizedStringChecker();

  // When this parameter is set to true, the checker assumes all
  // methods that return NSStrings are unlocalized. Thus, more false
  // positives will be reported.
  DefaultBool IsAggressive;

  void checkPreObjCMessage(const ObjCMethodCall &msg, CheckerContext &C) const;
  void checkPostObjCMessage(const ObjCMethodCall &msg, CheckerContext &C) const;
  void checkPostStmt(const ObjCStringLiteral *SL, CheckerContext &C) const;
  void checkPostCall(const CallEvent &Call, CheckerContext &C) const;
};

} // end anonymous namespace

REGISTER_MAP_WITH_PROGRAMSTATE(LocalizedMemMap, const MemRegion *,
                               LocalizedState)

NonLocalizedStringChecker::NonLocalizedStringChecker() {
  BT.reset(new BugType(this, "Unlocalized string", "Localizability Error"));
}

/// Initializes a list of methods that require a localized string
/// Format: {"ClassName", {{"selectorName:", LocStringArg#}, ...}, ...}
void NonLocalizedStringChecker::initUIMethods(ASTContext &Ctx) const {
  if (!UIMethods.empty())
    return;

  // TODO: This should eventually be a comprehensive list of UIKit methods

  // UILabel Methods
  llvm::StringMap<uint8_t> &UILabelM =
      UIMethods.insert({"UILabel", llvm::StringMap<uint8_t>()}).first->second;
  UILabelM.insert({"setText:", 0});

  // UIButton Methods
  llvm::StringMap<uint8_t> &UIButtonM =
      UIMethods.insert({"UIButton", llvm::StringMap<uint8_t>()}).first->second;
  UIButtonM.insert({"setText:", 0});

  // UIAlertAction Methods
  llvm::StringMap<uint8_t> &UIAlertActionM =
      UIMethods.insert({"UIAlertAction", llvm::StringMap<uint8_t>()})
          .first->second;
  UIAlertActionM.insert({"actionWithTitle:style:handler:", 0});

  // UIAlertController Methods
  llvm::StringMap<uint8_t> &UIAlertControllerM =
      UIMethods.insert({"UIAlertController", llvm::StringMap<uint8_t>()})
          .first->second;
  UIAlertControllerM.insert(
      {"alertControllerWithTitle:message:preferredStyle:", 1});

  // NSButton Methods
  llvm::StringMap<uint8_t> &NSButtonM =
      UIMethods.insert({"NSButton", llvm::StringMap<uint8_t>()}).first->second;
  NSButtonM.insert({"setTitle:", 0});

  // NSButtonCell Methods
  llvm::StringMap<uint8_t> &NSButtonCellM =
      UIMethods.insert({"NSButtonCell", llvm::StringMap<uint8_t>()})
          .first->second;
  NSButtonCellM.insert({"setTitle:", 0});

  // NSMenuItem Methods
  llvm::StringMap<uint8_t> &NSMenuItemM =
      UIMethods.insert({"NSMenuItem", llvm::StringMap<uint8_t>()})
          .first->second;
  NSMenuItemM.insert({"setTitle:", 0});

  // NSAttributedString Methods
  llvm::StringMap<uint8_t> &NSAttributedStringM =
      UIMethods.insert({"NSAttributedString", llvm::StringMap<uint8_t>()})
          .first->second;
  NSAttributedStringM.insert({"initWithString:", 0});
  NSAttributedStringM.insert({"initWithString:attributes:", 0});
}

/// Initializes a list of methods and C functions that return a localized string
void NonLocalizedStringChecker::initLocStringsMethods(ASTContext &Ctx) const {
  if (!LSM.empty())
    return;

  LSM.insert({"NSBundle", "localizedStringForKey:value:table:"});
  LSM.insert({"NSDateFormatter", "stringFromDate:"});
  LSM.insert(
      {"NSDateFormatter", "localizedStringFromDate:dateStyle:timeStyle:"});
  LSM.insert({"NSNumberFormatter", "stringFromNumber:"});
  LSM.insert({"UITextField", "text"});
  LSM.insert({"UITextView", "text"});
  LSM.insert({"UILabel", "text"});

  LSF.insert("CFDateFormatterCreateStringWithDate");
  LSF.insert("CFDateFormatterCreateStringWithAbsoluteTime");
  LSF.insert("CFNumberFormatterCreateStringWithNumber");
}

/// Checks to see if the method / function declaration includes
/// __attribute__((annotate("returns_localized_nsstring")))
bool NonLocalizedStringChecker::isAnnotatedAsLocalized(const Decl *D) const {
  return std::any_of(
      D->specific_attr_begin<AnnotateAttr>(),
      D->specific_attr_end<AnnotateAttr>(), [](const AnnotateAttr *Ann) {
        return Ann->getAnnotation() == "returns_localized_nsstring";
      });
}

/// Returns true if the given SVal is marked as Localized in the program state
bool NonLocalizedStringChecker::hasLocalizedState(SVal S,
                                                  CheckerContext &C) const {
  const MemRegion *mt = S.getAsRegion();
  if (mt) {
    const LocalizedState *LS = C.getState()->get<LocalizedMemMap>(mt);
    if (LS && LS->isLocalized())
      return true;
  }
  return false;
}

/// Returns true if the given SVal is marked as NonLocalized in the program
/// state
bool NonLocalizedStringChecker::hasNonLocalizedState(SVal S,
                                                     CheckerContext &C) const {
  const MemRegion *mt = S.getAsRegion();
  if (mt) {
    const LocalizedState *LS = C.getState()->get<LocalizedMemMap>(mt);
    if (LS && LS->isNonLocalized())
      return true;
  }
  return false;
}

/// Marks the given SVal as Localized in the program state
void NonLocalizedStringChecker::setLocalizedState(const SVal S,
                                                  CheckerContext &C) const {
  const MemRegion *mt = S.getAsRegion();
  if (mt) {
    ProgramStateRef State =
        C.getState()->set<LocalizedMemMap>(mt, LocalizedState::getLocalized());
    C.addTransition(State);
  }
}

/// Marks the given SVal as NonLocalized in the program state
void NonLocalizedStringChecker::setNonLocalizedState(const SVal S,
                                                     CheckerContext &C) const {
  const MemRegion *mt = S.getAsRegion();
  if (mt) {
    ProgramStateRef State = C.getState()->set<LocalizedMemMap>(
        mt, LocalizedState::getNonLocalized());
    C.addTransition(State);
  }
}

/// Reports a localization error for the passed in method call and SVal
void NonLocalizedStringChecker::reportLocalizationError(
    SVal S, const ObjCMethodCall &M, CheckerContext &C,
    int argumentNumber) const {

  ExplodedNode *ErrNode = C.getPredecessor();
  static CheckerProgramPointTag Tag("NonLocalizedStringChecker",
                                    "UnlocalizedString");
  ErrNode = C.addTransition(C.getState(), C.getPredecessor(), &Tag);

  if (!ErrNode)
    return;

  // Generate the bug report.
  std::unique_ptr<BugReport> R(
      new BugReport(*BT, "String should be localized", ErrNode));
  if (argumentNumber) {
    R->addRange(M.getArgExpr(argumentNumber - 1)->getSourceRange());
  } else {
    R->addRange(M.getSourceRange());
  }
  R->markInteresting(S);
  C.emitReport(std::move(R));
}

/// Check if the string being passed in has NonLocalized state
void NonLocalizedStringChecker::checkPreObjCMessage(const ObjCMethodCall &msg,
                                                    CheckerContext &C) const {
  initUIMethods(C.getASTContext());

  const ObjCInterfaceDecl *OD = msg.getReceiverInterface();
  if (!OD)
    return;
  const IdentifierInfo *odInfo = OD->getIdentifier();

  Selector S = msg.getSelector();

  std::string SelectorString = S.getAsString();
  StringRef SelectorName = SelectorString;
  assert(!SelectorName.empty());

  auto method = UIMethods.find(odInfo->getName());
  if (odInfo->isStr("NSString")) {
    // Handle the case where the receiver is an NSString
    // These special NSString methods draw to the screen

    if (!(SelectorName.startswith("drawAtPoint") ||
          SelectorName.startswith("drawInRect") ||
          SelectorName.startswith("drawWithRect")))
      return;

    SVal svTitle = msg.getReceiverSVal();

    bool isNonLocalized = hasNonLocalizedState(svTitle, C);

    if (isNonLocalized) {
      reportLocalizationError(svTitle, msg, C);
    }
  } else if (method != UIMethods.end()) {

    auto argumentIterator = method->getValue().find(SelectorName);

    if (argumentIterator == method->getValue().end())
      return;

    int argumentNumber = argumentIterator->getValue();

    SVal svTitle = msg.getArgSVal(argumentNumber);

    if (const ObjCStringRegion *SR =
            dyn_cast_or_null<ObjCStringRegion>(svTitle.getAsRegion())) {
      StringRef stringValue =
          SR->getObjCStringLiteral()->getString()->getString();
      if ((stringValue.trim().size() == 0 && stringValue.size() > 0) ||
          stringValue.empty())
        return;
      if (!IsAggressive && llvm::sys::unicode::columnWidthUTF8(stringValue) < 2)
        return;
    }

    bool isNonLocalized = hasNonLocalizedState(svTitle, C);

    if (isNonLocalized) {
      reportLocalizationError(svTitle, msg, C, argumentNumber + 1);
    }
  }
}

static inline bool isNSStringType(QualType T, ASTContext &Ctx) {

  const ObjCObjectPointerType *PT = T->getAs<ObjCObjectPointerType>();
  if (!PT)
    return false;

  ObjCInterfaceDecl *Cls = PT->getObjectType()->getInterface();
  if (!Cls)
    return false;

  IdentifierInfo *ClsName = Cls->getIdentifier();

  // FIXME: Should we walk the chain of classes?
  return ClsName == &Ctx.Idents.get("NSString") ||
         ClsName == &Ctx.Idents.get("NSMutableString");
}

/// Marks a string being returned by any call as localized
/// if it is in LocStringFunctions (LSF) or the function is annotated.
/// Otherwise, we mark it as NonLocalized (Aggressive) or
/// NonLocalized only if it is not backed by a SymRegion (Non-Aggressive),
/// basically leaving only string literals as NonLocalized.
void NonLocalizedStringChecker::checkPostCall(const CallEvent &Call,
                                              CheckerContext &C) const {
  initLocStringsMethods(C.getASTContext());

  if (!Call.getOriginExpr())
    return;

  // Anything that takes in a localized NSString as an argument
  // and returns an NSString will be assumed to be returning a
  // localized NSString. (Counter: Incorrectly combining two LocalizedStrings)
  const QualType RT = Call.getResultType();
  if (isNSStringType(RT, C.getASTContext())) {
    for (unsigned i = 0; i < Call.getNumArgs(); ++i) {
      SVal argValue = Call.getArgSVal(i);
      if (hasLocalizedState(argValue, C)) {
        SVal sv = Call.getReturnValue();
        setLocalizedState(sv, C);
        return;
      }
    }
  }

  const Decl *D = Call.getDecl();
  if (!D)
    return;

  StringRef IdentifierName = C.getCalleeName(D->getAsFunction());

  SVal sv = Call.getReturnValue();
  if (isAnnotatedAsLocalized(D) || LSF.find(IdentifierName) != LSF.end()) {
    setLocalizedState(sv, C);
  } else if (isNSStringType(RT, C.getASTContext()) &&
             !hasLocalizedState(sv, C)) {
    if (IsAggressive) {
      setNonLocalizedState(sv, C);
    } else {
      const SymbolicRegion *SymReg =
          dyn_cast_or_null<SymbolicRegion>(sv.getAsRegion());
      if (!SymReg)
        setNonLocalizedState(sv, C);
    }
  }
}

/// Marks a string being returned by an ObjC method as localized
/// if it is in LocStringMethods or the method is annotated
void NonLocalizedStringChecker::checkPostObjCMessage(const ObjCMethodCall &msg,
                                                     CheckerContext &C) const {
  initLocStringsMethods(C.getASTContext());

  if (!msg.isInstanceMessage())
    return;

  const ObjCInterfaceDecl *OD = msg.getReceiverInterface();
  if (!OD)
    return;
  const IdentifierInfo *odInfo = OD->getIdentifier();

  StringRef IdentifierName = odInfo->getName();

  Selector S = msg.getSelector();
  std::string SelectorName = S.getAsString();

  std::pair<StringRef, StringRef> MethodDescription = {IdentifierName,
                                                       SelectorName};

  if (LSM.count(MethodDescription) || isAnnotatedAsLocalized(msg.getDecl())) {
    SVal sv = msg.getReturnValue();
    setLocalizedState(sv, C);
  }
}

/// Marks all empty string literals as localized
void NonLocalizedStringChecker::checkPostStmt(const ObjCStringLiteral *SL,
                                              CheckerContext &C) const {
  SVal sv = C.getSVal(SL);
  setNonLocalizedState(sv, C);
}

namespace {
class EmptyLocalizationContextChecker
    : public Checker<check::ASTDecl<ObjCImplementationDecl>> {

  // A helper class, which walks the AST
  class MethodCrawler : public ConstStmtVisitor<MethodCrawler> {
    const ObjCMethodDecl *MD;
    BugReporter &BR;
    AnalysisManager &Mgr;
    const CheckerBase *Checker;
    LocationOrAnalysisDeclContext DCtx;

  public:
    MethodCrawler(const ObjCMethodDecl *InMD, BugReporter &InBR,
                  const CheckerBase *Checker, AnalysisManager &InMgr,
                  AnalysisDeclContext *InDCtx)
        : MD(InMD), BR(InBR), Mgr(InMgr), Checker(Checker), DCtx(InDCtx) {}

    void VisitStmt(const Stmt *S) { VisitChildren(S); }

    void VisitObjCMessageExpr(const ObjCMessageExpr *ME);

    void reportEmptyContextError(const ObjCMessageExpr *M) const;

    void VisitChildren(const Stmt *S) {
      for (const Stmt *Child : S->children()) {
        if (Child)
          this->Visit(Child);
      }
    }
  };

public:
  void checkASTDecl(const ObjCImplementationDecl *D, AnalysisManager &Mgr,
                    BugReporter &BR) const;
};
} // end anonymous namespace

void EmptyLocalizationContextChecker::checkASTDecl(
    const ObjCImplementationDecl *D, AnalysisManager &Mgr,
    BugReporter &BR) const {

  for (const ObjCMethodDecl *M : D->methods()) {
    AnalysisDeclContext *DCtx = Mgr.getAnalysisDeclContext(M);

    const Stmt *Body = M->getBody();
    assert(Body);

    MethodCrawler MC(M->getCanonicalDecl(), BR, this, Mgr, DCtx);
    MC.VisitStmt(Body);
  }
}

/// This check attempts to match these macros, assuming they are defined as
/// follows:
///
/// #define NSLocalizedString(key, comment) \
/// [[NSBundle mainBundle] localizedStringForKey:(key) value:@"" table:nil]
/// #define NSLocalizedStringFromTable(key, tbl, comment) \
/// [[NSBundle mainBundle] localizedStringForKey:(key) value:@"" table:(tbl)]
/// #define NSLocalizedStringFromTableInBundle(key, tbl, bundle, comment) \
/// [bundle localizedStringForKey:(key) value:@"" table:(tbl)]
/// #define NSLocalizedStringWithDefaultValue(key, tbl, bundle, val, comment)
///
/// We cannot use the path sensitive check because the macro argument we are
/// checking for (comment) is not used and thus not present in the AST,
/// so we use Lexer on the original macro call and retrieve the value of
/// the comment. If it's empty or nil, we raise a warning.
void EmptyLocalizationContextChecker::MethodCrawler::VisitObjCMessageExpr(
    const ObjCMessageExpr *ME) {

  const ObjCInterfaceDecl *OD = ME->getReceiverInterface();
  if (!OD)
    return;

  const IdentifierInfo *odInfo = OD->getIdentifier();

  if (!(odInfo->isStr("NSBundle") ||
        ME->getSelector().getAsString() ==
            "localizedStringForKey:value:table:")) {
    return;
  }

  SourceRange R = ME->getSourceRange();
  if (!R.getBegin().isMacroID())
    return;

  // getImmediateMacroCallerLoc gets the location of the immediate macro
  // caller, one level up the stack toward the initial macro typed into the
  // source, so SL should point to the NSLocalizedString macro.
  SourceLocation SL =
      Mgr.getSourceManager().getImmediateMacroCallerLoc(R.getBegin());
  std::pair<FileID, unsigned> SLInfo =
      Mgr.getSourceManager().getDecomposedLoc(SL);

  SrcMgr::SLocEntry SE = Mgr.getSourceManager().getSLocEntry(SLInfo.first);

  // If NSLocalizedString macro is wrapped in another macro, we need to
  // unwrap the expansion until we get to the NSLocalizedStringMacro.
  while (SE.isExpansion()) {
    SL = SE.getExpansion().getSpellingLoc();
    SLInfo = Mgr.getSourceManager().getDecomposedLoc(SL);
    SE = Mgr.getSourceManager().getSLocEntry(SLInfo.first);
  }

  llvm::MemoryBuffer *BF = SE.getFile().getContentCache()->getRawBuffer();
  Lexer TheLexer(SL, LangOptions(), BF->getBufferStart(),
                 BF->getBufferStart() + SLInfo.second, BF->getBufferEnd());

  Token I;
  Token Result;    // This will hold the token just before the last ')'
  int p_count = 0; // This is for parenthesis matching
  while (!TheLexer.LexFromRawLexer(I)) {
    if (I.getKind() == tok::l_paren)
      ++p_count;
    if (I.getKind() == tok::r_paren) {
      if (p_count == 1)
        break;
      --p_count;
    }
    Result = I;
  }

  if (isAnyIdentifier(Result.getKind())) {
    if (Result.getRawIdentifier().equals("nil")) {
      reportEmptyContextError(ME);
      return;
    }
  }

  if (!isStringLiteral(Result.getKind()))
    return;

  StringRef Comment =
      StringRef(Result.getLiteralData(), Result.getLength()).trim("\"");

  if ((Comment.trim().size() == 0 && Comment.size() > 0) || // Is Whitespace
      Comment.empty()) {
    reportEmptyContextError(ME);
  }
}

void EmptyLocalizationContextChecker::MethodCrawler::reportEmptyContextError(
    const ObjCMessageExpr *ME) const {
  // Generate the bug report.
  BR.EmitBasicReport(MD, Checker, "Context Missing", "Localizability Error",
                     "Localized string macro should include a non-empty "
                     "comment for translators",
                     PathDiagnosticLocation(ME, BR.getSourceManager(), DCtx));
}

//===----------------------------------------------------------------------===//
// Checker registration.
//===----------------------------------------------------------------------===//

void ento::registerNonLocalizedStringChecker(CheckerManager &mgr) {
  NonLocalizedStringChecker *checker =
      mgr.registerChecker<NonLocalizedStringChecker>();
  checker->IsAggressive =
      mgr.getAnalyzerOptions().getBooleanOption("AggressiveReport", false);
}

void ento::registerEmptyLocalizationContextChecker(CheckerManager &mgr) {
  mgr.registerChecker<EmptyLocalizationContextChecker>();
}
