//===-- DeclTemplate.h - Classes for representing C++ templates -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the C++ template declaration subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_DECLTEMPLATE_H
#define LLVM_CLANG_AST_DECLTEMPLATE_H

#include "clang/AST/DeclCXX.h"

namespace clang {

class TemplateParameterList;
class TemplateDecl;
class FunctionTemplateDecl;
class ClassTemplateDecl;
class TemplateTypeParmDecl;
class NonTypeTemplateParmDecl;
class TemplateTemplateParmDecl;

/// TemplateParameterList - Stores a list of template parameters for a
/// TemplateDecl and its derived classes.
class TemplateParameterList {
  /// The location of the 'template' keyword.
  SourceLocation TemplateLoc;

  /// The locations of the '<' and '>' angle brackets.
  SourceLocation LAngleLoc, RAngleLoc;

  /// The number of template parameters in this template
  /// parameter list.
  unsigned NumParams;

  TemplateParameterList(SourceLocation TemplateLoc, SourceLocation LAngleLoc,
                        Decl **Params, unsigned NumParams,
                        SourceLocation RAngleLoc);

public:
  static TemplateParameterList *Create(ASTContext &C, 
                                       SourceLocation TemplateLoc,
                                       SourceLocation LAngleLoc,
                                       Decl **Params,
                                       unsigned NumParams,
                                       SourceLocation RAngleLoc);

  /// iterator - Iterates through the template parameters in this list.
  typedef Decl** iterator;

  /// const_iterator - Iterates through the template parameters in this list.
  typedef Decl* const* const_iterator;

  iterator begin() { return reinterpret_cast<Decl **>(this + 1); }
  const_iterator begin() const {
    return reinterpret_cast<Decl * const *>(this + 1);
  }
  iterator end() { return begin() + NumParams; }
  const_iterator end() const { return begin() + NumParams; }

  unsigned size() const { return NumParams; }

  SourceLocation getTemplateLoc() const { return TemplateLoc; }
  SourceLocation getLAngleLoc() const { return LAngleLoc; }
  SourceLocation getRAngleLoc() const { return RAngleLoc; }
};

//===----------------------------------------------------------------------===//
// Kinds of Templates
//===----------------------------------------------------------------------===//

/// TemplateDecl - The base class of all kinds of template declarations (e.g.,
/// class, function, etc.). The TemplateDecl class stores the list of template
/// parameters and a reference to the templated scoped declaration: the
/// underlying AST node.
class TemplateDecl : public NamedDecl {
protected:
  // This is probably never used.
  TemplateDecl(Kind DK, DeclContext *DC, SourceLocation L,
               DeclarationName Name)
    : NamedDecl(DK, DC, L, Name), TemplatedDecl(0), TemplateParams(0)
  { }

  // Construct a template decl with the given name and parameters.
  // Used when there is not templated element (tt-params, alias?).
  TemplateDecl(Kind DK, DeclContext *DC, SourceLocation L,
               DeclarationName Name, TemplateParameterList *Params)
    : NamedDecl(DK, DC, L, Name), TemplatedDecl(0), TemplateParams(Params)
  { }

  // Construct a template decl with name, parameters, and templated element.
  TemplateDecl(Kind DK, DeclContext *DC, SourceLocation L,
               DeclarationName Name, TemplateParameterList *Params,
               NamedDecl *Decl)
    : NamedDecl(DK, DC, L, Name), TemplatedDecl(Decl),
      TemplateParams(Params) { }
public:
  ~TemplateDecl();

  /// Get the list of template parameters
  TemplateParameterList *getTemplateParameters() const {
    return TemplateParams;
  }

  /// Get the underlying, templated declaration.
  NamedDecl *getTemplatedDecl() const { return TemplatedDecl; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
      return D->getKind() >= TemplateFirst && D->getKind() <= TemplateLast;
  }
  static bool classof(const TemplateDecl *D) { return true; }
  static bool classof(const FunctionTemplateDecl *D) { return true; }
  static bool classof(const ClassTemplateDecl *D) { return true; }
  static bool classof(const TemplateTemplateParmDecl *D) { return true; }

protected:
  NamedDecl *TemplatedDecl;
  TemplateParameterList* TemplateParams;
};

/// Declaration of a template function.
class FunctionTemplateDecl : public TemplateDecl {
protected:
  FunctionTemplateDecl(DeclContext *DC, SourceLocation L, DeclarationName Name,
                       TemplateParameterList *Params, NamedDecl *Decl)
    : TemplateDecl(FunctionTemplate, DC, L, Name, Params, Decl) { }
public:
  /// Get the underling function declaration of the template.
  FunctionDecl *getTemplatedDecl() const {
    return static_cast<FunctionDecl*>(TemplatedDecl);
  }

  /// Create a template function node.
  static FunctionTemplateDecl *Create(ASTContext &C, DeclContext *DC,
                                      SourceLocation L,
                                      DeclarationName Name,
                                      TemplateParameterList *Params,
                                      NamedDecl *Decl);

  // Implement isa/cast/dyncast support
  static bool classof(const Decl *D)
  { return D->getKind() == FunctionTemplate; }
  static bool classof(const FunctionTemplateDecl *D)
  { return true; }
};

/// Declaration of a template class.
class ClassTemplateDecl : public TemplateDecl {
protected:
  ClassTemplateDecl(DeclContext *DC, SourceLocation L, DeclarationName Name,
                    TemplateParameterList *Params, NamedDecl *Decl)
    : TemplateDecl(ClassTemplate, DC, L, Name, Params, Decl) { }
public:
  /// Get the underlying class declarations of the template.
  CXXRecordDecl *getTemplatedDecl() const {
    return static_cast<CXXRecordDecl *>(TemplatedDecl);
  }

  /// Create a class teplate node.
  static ClassTemplateDecl *Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L,
                                   DeclarationName Name,
                                   TemplateParameterList *Params,
                                   NamedDecl *Decl);

  // Implement isa/cast/dyncast support
  static bool classof(const Decl *D)
  { return D->getKind() == ClassTemplate; }
  static bool classof(const ClassTemplateDecl *D)
  { return true; }
};

//===----------------------------------------------------------------------===//
// Kinds of Template Parameters
//===----------------------------------------------------------------------===//


/// The TemplateParmPosition class defines the position of a template parameter
/// within a template parameter list. Because template parameter can be listed
/// sequentially for out-of-line template members, each template parameter is
/// given a Depth - the nesting of template parameter scopes - and a Position -
/// the occurrence within the parameter list.
/// This class is inheritedly privately by different kinds of template
/// parameters and is not part of the Decl hierarchy. Just a facility.
class TemplateParmPosition
{
protected:
  // FIXME: This should probably never be called, but it's here as
  TemplateParmPosition()
    : Depth(0), Position(0)
  { /* assert(0 && "Cannot create positionless template parameter"); */ }

  TemplateParmPosition(unsigned D, unsigned P)
    : Depth(D), Position(P)
  { }

  // FIXME: These probably don't need to be ints. int:5 for depth, int:8 for
  // position? Maybe?
  unsigned Depth;
  unsigned Position;

public:
  /// Get the nesting depth of the template parameter.
  unsigned getDepth() const { return Depth; }

  /// Get the position of the template parameter within its parameter list.
  unsigned getPosition() const { return Position; }
};

/// TemplateTypeParmDecl - Declaration of a template type parameter,
/// e.g., "T" in
/// @code
/// template<typename T> class vector;
/// @endcode
class TemplateTypeParmDecl : public TypeDecl {
  /// \brief Whether this template type parameter was declaration with
  /// the 'typename' keyword. If false, it was declared with the
  /// 'class' keyword.
  bool Typename : 1;

  /// \brief Whether this template type parameter inherited its
  /// default argument.
  bool InheritedDefault : 1;

  /// \brief The location of the default argument, if any.
  SourceLocation DefaultArgumentLoc;

  /// \brief The default template argument, if any.
  QualType DefaultArgument;

  TemplateTypeParmDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id, 
                       bool Typename, QualType Type)
    : TypeDecl(TemplateTypeParm, DC, L, Id), Typename(Typename),
      InheritedDefault(false), DefaultArgument() { 
    TypeForDecl = Type.getTypePtr();
  }

public:
  static TemplateTypeParmDecl *Create(ASTContext &C, DeclContext *DC,
                                      SourceLocation L, unsigned D, unsigned P,
                                      IdentifierInfo *Id, bool Typename);

  /// \brief Whether this template type parameter was declared with
  /// the 'typename' keyword. If not, it was declared with the 'class'
  /// keyword.
  bool wasDeclaredWithTypename() const { return Typename; }

  /// \brief Determine whether this template parameter has a default
  /// argument.
  bool hasDefaultArgument() const { return !DefaultArgument.isNull(); }

  /// \brief Retrieve the default argument, if any.
  QualType getDefaultArgument() const { return DefaultArgument; }

  /// \brief Retrieve the location of the default argument, if any.
  SourceLocation getDefaultArgumentLoc() const { return DefaultArgumentLoc; }

  /// \brief Determines whether the default argument was inherited
  /// from a previous declaration of this template.
  bool defaultArgumentWasInherited() const { return InheritedDefault; }

  /// \brief Set the default argument for this template parameter, and
  /// whether that default argument was inherited from another
  /// declaration.
  void setDefaultArgument(QualType DefArg, SourceLocation DefArgLoc,
                          bool Inherited) {
    DefaultArgument = DefArg;
    DefaultArgumentLoc = DefArgLoc;
    InheritedDefault = Inherited;
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == TemplateTypeParm;
  }
  static bool classof(const TemplateTypeParmDecl *D) { return true; }

protected:
  /// Serialize this TemplateTypeParmDecl.  Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;

  /// Deserialize a TemplateTypeParmDecl.  Called by Decl::Create.
  static TemplateTypeParmDecl* CreateImpl(llvm::Deserializer& D, ASTContext& C);

  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

/// NonTypeTemplateParmDecl - Declares a non-type template parameter,
/// e.g., "Size" in
/// @code
/// template<int Size> class array { };
/// @endcode
class NonTypeTemplateParmDecl
  : public VarDecl, protected TemplateParmPosition {
  /// \brief The default template argument, if any.
  Expr *DefaultArgument;

  NonTypeTemplateParmDecl(DeclContext *DC, SourceLocation L, unsigned D,
                          unsigned P, IdentifierInfo *Id, QualType T,
                          SourceLocation TSSL = SourceLocation())
    : VarDecl(NonTypeTemplateParm, DC, L, Id, T, VarDecl::None, TSSL),
      TemplateParmPosition(D, P), DefaultArgument(0) 
  { }

public:
  static NonTypeTemplateParmDecl *
  Create(ASTContext &C, DeclContext *DC, SourceLocation L, unsigned D,
         unsigned P, IdentifierInfo *Id, QualType T,
         SourceLocation TypeSpecStartLoc = SourceLocation());

  using TemplateParmPosition::getDepth;
  using TemplateParmPosition::getPosition;

  /// \brief Determine whether this template parameter has a default
  /// argument.
  bool hasDefaultArgument() const { return DefaultArgument; }

  /// \brief Retrieve the default argument, if any.
  Expr *getDefaultArgument() const { return DefaultArgument; }

  /// \brief Retrieve the location of the default argument, if any.
  SourceLocation getDefaultArgumentLoc() const;

  /// \brief Set the default argument for this template parameter.
  void setDefaultArgument(Expr *DefArg) {
    DefaultArgument = DefArg;
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == NonTypeTemplateParm;
  }
  static bool classof(const NonTypeTemplateParmDecl *D) { return true; }

protected:
  /// EmitImpl - Serialize this TemplateTypeParmDecl.  Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;

  /// CreateImpl - Deserialize a TemplateTypeParmDecl.  Called by Decl::Create.
  static NonTypeTemplateParmDecl* CreateImpl(llvm::Deserializer& D,
                                             ASTContext& C);

  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

/// TemplateTemplateParmDecl - Declares a template template parameter,
/// e.g., "T" in
/// @code
/// template <template <typename> class T> class container { };
/// @endcode
/// A template template parameter is a TemplateDecl because it defines the
/// name of a template and the template parameters allowable for substitution.
class TemplateTemplateParmDecl
  : public TemplateDecl, protected TemplateParmPosition {

  /// \brief The default template argument, if any.
  Expr *DefaultArgument;

  TemplateTemplateParmDecl(DeclContext *DC, SourceLocation L,
                           unsigned D, unsigned P,
                           IdentifierInfo *Id, TemplateParameterList *Params)
    : TemplateDecl(TemplateTemplateParm, DC, L, Id, Params),
      TemplateParmPosition(D, P), DefaultArgument(0)
    { }

public:
  static TemplateTemplateParmDecl *Create(ASTContext &C, DeclContext *DC,
                                          SourceLocation L, unsigned D,
                                          unsigned P, IdentifierInfo *Id,
                                          TemplateParameterList *Params);

  using TemplateParmPosition::getDepth;
  using TemplateParmPosition::getPosition;

  /// \brief Determine whether this template parameter has a default
  /// argument.
  bool hasDefaultArgument() const { return DefaultArgument; }

  /// \brief Retrieve the default argument, if any.
  Expr *getDefaultArgument() const { return DefaultArgument; }

  /// \brief Retrieve the location of the default argument, if any.
  SourceLocation getDefaultArgumentLoc() const;

  /// \brief Set the default argument for this template parameter.
  void setDefaultArgument(Expr *DefArg) {
    DefaultArgument = DefArg;
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == TemplateTemplateParm;
  }
  static bool classof(const TemplateTemplateParmDecl *D) { return true; }

protected:
  /// EmitImpl - Serialize this TemplateTypeParmDecl.  Called by Decl::Emit.
  virtual void EmitImpl(llvm::Serializer& S) const;

  /// CreateImpl - Deserialize a TemplateTypeParmDecl.  Called by Decl::Create.
  static TemplateTemplateParmDecl* CreateImpl(llvm::Deserializer& D,
                                              ASTContext& C);

  friend Decl* Decl::Create(llvm::Deserializer& D, ASTContext& C);
};

} /* end of namespace clang */

#endif
