//== CheckerContext.h - Context info for path-sensitive checkers--*- C++ -*--=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines CheckerContext that provides contextual info for
// path-sensitive checkers.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SA_CORE_PATHSENSITIVE_CHECKERCONTEXT
#define LLVM_CLANG_SA_CORE_PATHSENSITIVE_CHECKERCONTEXT

#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "llvm/ADT/ImmutableMap.h"

/// Declare an immutable map suitable for placement into the ProgramState.
#define REGISTER_MAP_WITH_PROGRAMSTATE(Map, Key, Value) \
  typedef llvm::ImmutableMap<Key, Value> Map; \
  namespace clang { \
  namespace ento { \
    template <> \
    struct ProgramStateTrait<Map> \
      : public ProgramStatePartialTrait<Map> { \
      static void *GDMIndex() { static int Index; return &Index; } \
    }; \
  } \
  }


namespace clang {
namespace ento {

class CheckerContext {
  ExprEngine &Eng;
  /// The current exploded(symbolic execution) graph node.
  ExplodedNode *Pred;
  /// The flag is true if the (state of the execution) has been modified
  /// by the checker using this context. For example, a new transition has been
  /// added or a bug report issued.
  bool Changed;
  /// The tagged location, which is used to generate all new nodes.
  const ProgramPoint Location;
  NodeBuilder &NB;

public:
  /// If we are post visiting a call, this flag will be set if the
  /// call was inlined.  In all other cases it will be false.
  const bool wasInlined;
  
  CheckerContext(NodeBuilder &builder,
                 ExprEngine &eng,
                 ExplodedNode *pred,
                 const ProgramPoint &loc,
                 bool wasInlined = false)
    : Eng(eng),
      Pred(pred),
      Changed(false),
      Location(loc),
      NB(builder),
      wasInlined(wasInlined) {
    assert(Pred->getState() &&
           "We should not call the checkers on an empty state.");
  }

  AnalysisManager &getAnalysisManager() {
    return Eng.getAnalysisManager();
  }

  ConstraintManager &getConstraintManager() {
    return Eng.getConstraintManager();
  }

  StoreManager &getStoreManager() {
    return Eng.getStoreManager();
  }

  const AnalyzerOptions::ConfigTable &getConfig() const {
    return Eng.getAnalysisManager().options.Config;
  }
  
  /// \brief Returns the previous node in the exploded graph, which includes
  /// the state of the program before the checker ran. Note, checkers should
  /// not retain the node in their state since the nodes might get invalidated.
  ExplodedNode *getPredecessor() { return Pred; }
  ProgramStateRef getState() const { return Pred->getState(); }

  /// \brief Check if the checker changed the state of the execution; ex: added
  /// a new transition or a bug report.
  bool isDifferent() { return Changed; }

  /// \brief Returns the number of times the current block has been visited
  /// along the analyzed path.
  unsigned blockCount() const {
    return NB.getContext().blockCount();
  }

  ASTContext &getASTContext() {
    return Eng.getContext();
  }

  const LangOptions &getLangOpts() const {
    return Eng.getContext().getLangOpts();
  }

  const LocationContext *getLocationContext() const {
    return Pred->getLocationContext();
  }

  const StackFrameContext *getStackFrame() const {
    return Pred->getStackFrame();
  }

  /// Returns true if the predecessor is within an inlined function/method.
  bool isWithinInlined() {
    return (getStackFrame()->getParent() != 0);
  }

  BugReporter &getBugReporter() {
    return Eng.getBugReporter();
  }
  
  SourceManager &getSourceManager() {
    return getBugReporter().getSourceManager();
  }

  SValBuilder &getSValBuilder() {
    return Eng.getSValBuilder();
  }

  SymbolManager &getSymbolManager() {
    return getSValBuilder().getSymbolManager();
  }

  bool isObjCGCEnabled() const {
    return Eng.isObjCGCEnabled();
  }

  ProgramStateManager &getStateManager() {
    return Eng.getStateManager();
  }

  AnalysisDeclContext *getCurrentAnalysisDeclContext() const {
    return Pred->getLocationContext()->getAnalysisDeclContext();
  }

  /// \brief If the given node corresponds to a PostStore program point, retrieve
  /// the location region as it was uttered in the code.
  ///
  /// This utility can be useful for generating extensive diagnostics, for
  /// example, for finding variables that the given symbol was assigned to.
  static const MemRegion *getLocationRegionIfPostStore(const ExplodedNode *N) {
    ProgramPoint L = N->getLocation();
    if (const PostStore *PSL = dyn_cast<PostStore>(&L))
      return reinterpret_cast<const MemRegion*>(PSL->getLocationValue());
    return 0;
  }

  /// \brief Get the value of arbitrary expressions at this point in the path.
  SVal getSVal(const Stmt *S) const {
    return getState()->getSVal(S, getLocationContext());
  }

  /// \brief Generates a new transition in the program state graph
  /// (ExplodedGraph). Uses the default CheckerContext predecessor node.
  ///
  /// @param State The state of the generated node. If not specified, the state
  ///        will not be changed, but the new node will have the checker's tag.
  /// @param Tag The tag is used to uniquely identify the creation site. If no
  ///        tag is specified, a default tag, unique to the given checker,
  ///        will be used. Tags are used to prevent states generated at
  ///        different sites from caching out.
  ExplodedNode *addTransition(ProgramStateRef State = 0,
                              const ProgramPointTag *Tag = 0) {
    return addTransitionImpl(State ? State : getState(), false, 0, Tag);
  }

  /// \brief Generates a new transition with the given predecessor.
  /// Allows checkers to generate a chain of nodes.
  ///
  /// @param State The state of the generated node.
  /// @param Pred The transition will be generated from the specified Pred node
  ///             to the newly generated node.
  /// @param Tag The tag to uniquely identify the creation site.
  ExplodedNode *addTransition(ProgramStateRef State,
                              ExplodedNode *Pred,
                              const ProgramPointTag *Tag = 0) {
    return addTransitionImpl(State, false, Pred, Tag);
  }

  /// \brief Generate a sink node. Generating a sink stops exploration of the
  /// given path.
  ExplodedNode *generateSink(ProgramStateRef State = 0,
                             ExplodedNode *Pred = 0,
                             const ProgramPointTag *Tag = 0) {
    return addTransitionImpl(State ? State : getState(), true, Pred, Tag);
  }

  /// \brief Emit the diagnostics report.
  void EmitReport(BugReport *R) {
    Changed = true;
    Eng.getBugReporter().EmitReport(R);
  }

  /// \brief Get the declaration of the called function (path-sensitive).
  const FunctionDecl *getCalleeDecl(const CallExpr *CE) const;

  /// \brief Get the name of the called function (path-sensitive).
  StringRef getCalleeName(const FunctionDecl *FunDecl) const;

  /// \brief Get the identifier of the called function (path-sensitive).
  const IdentifierInfo *getCalleeIdentifier(const CallExpr *CE) const {
    const FunctionDecl *FunDecl = getCalleeDecl(CE);
    if (FunDecl)
      return FunDecl->getIdentifier();
    else
      return 0;
  }

  /// \brief Get the name of the called function (path-sensitive).
  StringRef getCalleeName(const CallExpr *CE) const {
    const FunctionDecl *FunDecl = getCalleeDecl(CE);
    return getCalleeName(FunDecl);
  }

  /// Given a function declaration and a name checks if this is a C lib
  /// function with the given name.
  bool isCLibraryFunction(const FunctionDecl *FD, StringRef Name);
  static bool isCLibraryFunction(const FunctionDecl *FD, StringRef Name,
                                 ASTContext &Context);

  /// \brief Depending on wither the location corresponds to a macro, return 
  /// either the macro name or the token spelling.
  ///
  /// This could be useful when checkers' logic depends on whether a function
  /// is called with a given macro argument. For example:
  ///   s = socket(AF_INET,..)
  /// If AF_INET is a macro, the result should be treated as a source of taint.
  ///
  /// \sa clang::Lexer::getSpelling(), clang::Lexer::getImmediateMacroName().
  StringRef getMacroNameOrSpelling(SourceLocation &Loc);

private:
  ExplodedNode *addTransitionImpl(ProgramStateRef State,
                                 bool MarkAsSink,
                                 ExplodedNode *P = 0,
                                 const ProgramPointTag *Tag = 0) {
    if (!State || (State == Pred->getState() && !Tag && !MarkAsSink))
      return Pred;

    Changed = true;
    const ProgramPoint &LocalLoc = (Tag ? Location.withTag(Tag) : Location);
    if (!P)
      P = Pred;

    ExplodedNode *node;
    if (MarkAsSink)
      node = NB.generateSink(LocalLoc, State, P);
    else
      node = NB.generateNode(LocalLoc, State, P);
    return node;
  }
};

/// \brief A helper class which wraps a boolean value set to false by default.
struct DefaultBool {
  bool Val;
  DefaultBool() : Val(false) {}
  operator bool() const { return Val; }
  DefaultBool &operator=(bool b) { Val = b; return *this; }
};

} // end GR namespace

} // end clang namespace

#endif
