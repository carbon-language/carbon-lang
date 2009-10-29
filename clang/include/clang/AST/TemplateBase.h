//===-- TemplateBase.h - Core classes for C++ templates ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides definitions which are common for all kinds of
//  template representation.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TEMPLATEBASE_H
#define LLVM_CLANG_AST_TEMPLATEBASE_H

#include "llvm/ADT/APSInt.h"
#include "llvm/Support/ErrorHandling.h"
#include "clang/AST/Type.h"

namespace llvm {
  class FoldingSetNodeID;
}

namespace clang {

class Decl;
class Expr;
class DeclaratorInfo;

/// \brief Represents a template argument within a class template
/// specialization.
class TemplateArgument {
  union {
    uintptr_t TypeOrValue;
    struct {
      char Value[sizeof(llvm::APSInt)];
      void *Type;
    } Integer;
    struct {
      TemplateArgument *Args;
      unsigned NumArgs;
      bool CopyArgs;
    } Args;
  };

public:
  /// \brief The type of template argument we're storing.
  enum ArgKind {
    Null = 0,
    /// The template argument is a type. Its value is stored in the
    /// TypeOrValue field.
    Type = 1,
    /// The template argument is a declaration
    Declaration = 2,
    /// The template argument is an integral value stored in an llvm::APSInt.
    Integral = 3,
    /// The template argument is a value- or type-dependent expression
    /// stored in an Expr*.
    Expression = 4,

    /// The template argument is actually a parameter pack. Arguments are stored
    /// in the Args struct.
    Pack = 5
  } Kind;

  /// \brief Construct an empty, invalid template argument.
  TemplateArgument() : TypeOrValue(0), Kind(Null) { }

  /// \brief Construct a template type argument.
  TemplateArgument(QualType T) : Kind(Type) {
    TypeOrValue = reinterpret_cast<uintptr_t>(T.getAsOpaquePtr());
  }

  /// \brief Construct a template argument that refers to a
  /// declaration, which is either an external declaration or a
  /// template declaration.
  TemplateArgument(Decl *D) : Kind(Declaration) {
    // FIXME: Need to be sure we have the "canonical" declaration!
    TypeOrValue = reinterpret_cast<uintptr_t>(D);
  }

  /// \brief Construct an integral constant template argument.
  TemplateArgument(const llvm::APSInt &Value, QualType Type)
  : Kind(Integral) {
    new (Integer.Value) llvm::APSInt(Value);
    Integer.Type = Type.getAsOpaquePtr();
  }

  /// \brief Construct a template argument that is an expression.
  ///
  /// This form of template argument only occurs in template argument
  /// lists used for dependent types and for expression; it will not
  /// occur in a non-dependent, canonical template argument list.
  TemplateArgument(Expr *E) : Kind(Expression) {
    TypeOrValue = reinterpret_cast<uintptr_t>(E);
  }

  /// \brief Copy constructor for a template argument.
  TemplateArgument(const TemplateArgument &Other) : Kind(Other.Kind) {
    if (Kind == Integral) {
      new (Integer.Value) llvm::APSInt(*Other.getAsIntegral());
      Integer.Type = Other.Integer.Type;
    } else if (Kind == Pack) {
      Args.NumArgs = Other.Args.NumArgs;
      Args.Args = new TemplateArgument[Args.NumArgs];
      for (unsigned I = 0; I != Args.NumArgs; ++I)
        Args.Args[I] = Other.Args.Args[I];
    }
    else
      TypeOrValue = Other.TypeOrValue;
  }

  TemplateArgument& operator=(const TemplateArgument& Other) {
    // FIXME: Does not provide the strong guarantee for exception
    // safety.
    using llvm::APSInt;

    // FIXME: Handle Packs
    assert(Kind != Pack && "FIXME: Handle packs");
    assert(Other.Kind != Pack && "FIXME: Handle packs");

    if (Kind == Other.Kind && Kind == Integral) {
      // Copy integral values.
      *this->getAsIntegral() = *Other.getAsIntegral();
      Integer.Type = Other.Integer.Type;
    } else {
      // Destroy the current integral value, if that's what we're holding.
      if (Kind == Integral)
        getAsIntegral()->~APSInt();

      Kind = Other.Kind;

      if (Other.Kind == Integral) {
        new (Integer.Value) llvm::APSInt(*Other.getAsIntegral());
        Integer.Type = Other.Integer.Type;
      } else
        TypeOrValue = Other.TypeOrValue;
    }

    return *this;
  }

  ~TemplateArgument() {
    using llvm::APSInt;

    if (Kind == Integral)
      getAsIntegral()->~APSInt();
    else if (Kind == Pack && Args.CopyArgs)
      delete[] Args.Args;
  }

  /// \brief Return the kind of stored template argument.
  ArgKind getKind() const { return Kind; }

  /// \brief Determine whether this template argument has no value.
  bool isNull() const { return Kind == Null; }

  /// \brief Retrieve the template argument as a type.
  QualType getAsType() const {
    if (Kind != Type)
      return QualType();

    return QualType::getFromOpaquePtr(reinterpret_cast<void*>(TypeOrValue));
  }

  /// \brief Retrieve the template argument as a declaration.
  Decl *getAsDecl() const {
    if (Kind != Declaration)
      return 0;
    return reinterpret_cast<Decl *>(TypeOrValue);
  }

  /// \brief Retrieve the template argument as an integral value.
  llvm::APSInt *getAsIntegral() {
    if (Kind != Integral)
      return 0;
    return reinterpret_cast<llvm::APSInt*>(&Integer.Value[0]);
  }

  const llvm::APSInt *getAsIntegral() const {
    return const_cast<TemplateArgument*>(this)->getAsIntegral();
  }

  /// \brief Retrieve the type of the integral value.
  QualType getIntegralType() const {
    if (Kind != Integral)
      return QualType();

    return QualType::getFromOpaquePtr(Integer.Type);
  }

  void setIntegralType(QualType T) {
    assert(Kind == Integral &&
           "Cannot set the integral type of a non-integral template argument");
    Integer.Type = T.getAsOpaquePtr();
  };

  /// \brief Retrieve the template argument as an expression.
  Expr *getAsExpr() const {
    if (Kind != Expression)
      return 0;

    return reinterpret_cast<Expr *>(TypeOrValue);
  }

  /// \brief Iterator that traverses the elements of a template argument pack.
  typedef const TemplateArgument * pack_iterator;

  /// \brief Iterator referencing the first argument of a template argument
  /// pack.
  pack_iterator pack_begin() const {
    assert(Kind == Pack);
    return Args.Args;
  }

  /// \brief Iterator referencing one past the last argument of a template
  /// argument pack.
  pack_iterator pack_end() const {
    assert(Kind == Pack);
    return Args.Args + Args.NumArgs;
  }

  /// \brief The number of template arguments in the given template argument
  /// pack.
  unsigned pack_size() const {
    assert(Kind == Pack);
    return Args.NumArgs;
  }

  /// \brief Construct a template argument pack.
  void setArgumentPack(TemplateArgument *Args, unsigned NumArgs, bool CopyArgs);

  /// \brief Used to insert TemplateArguments into FoldingSets.
  void Profile(llvm::FoldingSetNodeID &ID, ASTContext &Context) const;
};

/// Location information for a TemplateArgument.
struct TemplateArgumentLocInfo {
private:
  void *Union;

#ifndef NDEBUG
  enum Kind {
    K_None,
    K_DeclaratorInfo,
    K_Expression
  } Kind;
#endif

public:
  TemplateArgumentLocInfo()
    : Union(), Kind(K_None) {}
  TemplateArgumentLocInfo(DeclaratorInfo *DInfo)
    : Union(DInfo), Kind(K_DeclaratorInfo) {}
  TemplateArgumentLocInfo(Expr *E)
    : Union(E), Kind(K_Expression) {}

  /// \brief Returns whether this 
  bool empty() const {
    return Union == NULL;
  }

  DeclaratorInfo *getAsDeclaratorInfo() const {
    assert(Kind == K_DeclaratorInfo);
    return reinterpret_cast<DeclaratorInfo*>(Union);
  }

  Expr *getAsExpr() const {
    assert(Kind == K_Expression);
    return reinterpret_cast<Expr*>(Union);
  }

#ifndef NDEBUG
  void validateForArgument(const TemplateArgument &Arg) {
    switch (Arg.getKind()) {
    case TemplateArgument::Type:
      assert(Kind == K_DeclaratorInfo);
      break;
    case TemplateArgument::Expression:
      assert(Kind == K_Expression);
      break;
    case TemplateArgument::Declaration:
    case TemplateArgument::Integral:
    case TemplateArgument::Pack:
      assert(Kind == K_None);
      break;
    case TemplateArgument::Null:
      llvm::llvm_unreachable("source info for null template argument?");
    }
  }
#endif
};

/// Location wrapper for a TemplateArgument.  TemplateArgument is to
/// TemplateArgumentLoc as Type is to TypeLoc.
class TemplateArgumentLoc {
  TemplateArgument Argument;
  TemplateArgumentLocInfo LocInfo;

  friend class TemplateSpecializationTypeLoc;
  TemplateArgumentLoc(const TemplateArgument &Argument,
                      TemplateArgumentLocInfo Opaque)
    : Argument(Argument), LocInfo(Opaque) {
  }

public:
  TemplateArgumentLoc() {}

  TemplateArgumentLoc(const TemplateArgument &Argument, DeclaratorInfo *DInfo)
    : Argument(Argument), LocInfo(DInfo) {
    assert(Argument.getKind() == TemplateArgument::Type);
  }

  TemplateArgumentLoc(const TemplateArgument &Argument, Expr *E)
    : Argument(Argument), LocInfo(E) {
    assert(Argument.getKind() == TemplateArgument::Expression);
  }

  /// This is a temporary measure.
  TemplateArgumentLoc(const TemplateArgument &Argument)
    : Argument(Argument), LocInfo() {
    assert(Argument.getKind() != TemplateArgument::Expression &&
           Argument.getKind() != TemplateArgument::Type);
  }

  /// \brief - Fetches the start location of the argument, if possible.
  SourceLocation getLocation() const;

  const TemplateArgument &getArgument() const {
    return Argument;
  }

  TemplateArgumentLocInfo getLocInfo() const {
    return LocInfo;
  }

  DeclaratorInfo *getSourceDeclaratorInfo() const {
    assert(Argument.getKind() == TemplateArgument::Type);
    if (LocInfo.empty()) return 0;
    return LocInfo.getAsDeclaratorInfo();
  }

  Expr *getSourceExpression() const {
    assert(Argument.getKind() == TemplateArgument::Expression);
    if (LocInfo.empty()) return 0;
    return LocInfo.getAsExpr();
  }
};

}

#endif
