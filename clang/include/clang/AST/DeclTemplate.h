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
#include "clang/AST/TemplateBase.h"
#include "llvm/ADT/PointerUnion.h"
#include <limits>

namespace clang {

class TemplateParameterList;
class TemplateDecl;
class FunctionTemplateDecl;
class ClassTemplateDecl;
class ClassTemplatePartialSpecializationDecl;
class TemplateTypeParmDecl;
class NonTypeTemplateParmDecl;
class TemplateTemplateParmDecl;

/// \brief Stores a template parameter of any kind.
typedef llvm::PointerUnion3<TemplateTypeParmDecl*, NonTypeTemplateParmDecl*,
                            TemplateTemplateParmDecl*> TemplateParameter;

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
                        NamedDecl **Params, unsigned NumParams,
                        SourceLocation RAngleLoc);

public:
  static TemplateParameterList *Create(ASTContext &C,
                                       SourceLocation TemplateLoc,
                                       SourceLocation LAngleLoc,
                                       NamedDecl **Params,
                                       unsigned NumParams,
                                       SourceLocation RAngleLoc);

  /// iterator - Iterates through the template parameters in this list.
  typedef NamedDecl** iterator;

  /// const_iterator - Iterates through the template parameters in this list.
  typedef NamedDecl* const* const_iterator;

  iterator begin() { return reinterpret_cast<NamedDecl **>(this + 1); }
  const_iterator begin() const {
    return reinterpret_cast<NamedDecl * const *>(this + 1);
  }
  iterator end() { return begin() + NumParams; }
  const_iterator end() const { return begin() + NumParams; }

  unsigned size() const { return NumParams; }

  NamedDecl* getParam(unsigned Idx) {
    assert(Idx < size() && "Template parameter index out-of-range");
    return begin()[Idx];
  }

  const NamedDecl* getParam(unsigned Idx) const {
    assert(Idx < size() && "Template parameter index out-of-range");
    return begin()[Idx];
  }

  /// \btief Returns the minimum number of arguments needed to form a
  /// template specialization. This may be fewer than the number of
  /// template parameters, if some of the parameters have default
  /// arguments or if there is a parameter pack.
  unsigned getMinRequiredArguments() const;

  /// \brief Get the depth of this template parameter list in the set of
  /// template parameter lists.
  ///
  /// The first template parameter list in a declaration will have depth 0,
  /// the second template parameter list will have depth 1, etc.
  unsigned getDepth() const;
  
  SourceLocation getTemplateLoc() const { return TemplateLoc; }
  SourceLocation getLAngleLoc() const { return LAngleLoc; }
  SourceLocation getRAngleLoc() const { return RAngleLoc; }

  SourceRange getSourceRange() const {
    return SourceRange(TemplateLoc, RAngleLoc);
  }
};

/// \brief A helper class for making template argument lists.
class TemplateArgumentListBuilder {
  TemplateArgument *StructuredArgs;
  unsigned MaxStructuredArgs;
  unsigned NumStructuredArgs;

  TemplateArgument *FlatArgs;
  unsigned MaxFlatArgs;
  unsigned NumFlatArgs;

  bool AddingToPack;
  unsigned PackBeginIndex;

public:
  TemplateArgumentListBuilder(const TemplateParameterList *Parameters,
                              unsigned NumTemplateArgs)
  : StructuredArgs(0), MaxStructuredArgs(Parameters->size()),
  NumStructuredArgs(0), FlatArgs(0),
  MaxFlatArgs(std::max(MaxStructuredArgs, NumTemplateArgs)), NumFlatArgs(0),
  AddingToPack(false), PackBeginIndex(0) { }

  void Append(const TemplateArgument& Arg);
  void BeginPack();
  void EndPack();

  void ReleaseArgs();

  unsigned flatSize() const {
    return NumFlatArgs;
  }
  const TemplateArgument *getFlatArguments() const {
    return FlatArgs;
  }

  unsigned structuredSize() const {
    // If we don't have any structured args, just reuse the flat size.
    if (!StructuredArgs)
      return flatSize();

    return NumStructuredArgs;
  }
  const TemplateArgument *getStructuredArguments() const {
    // If we don't have any structured args, just reuse the flat args.
    if (!StructuredArgs)
      return getFlatArguments();

    return StructuredArgs;
  }
};

/// \brief A template argument list.
///
/// FIXME: In the future, this class will be extended to support
/// variadic templates and member templates, which will make some of
/// the function names below make more sense.
class TemplateArgumentList {
  /// \brief The template argument list.
  ///
  /// The integer value will be non-zero to indicate that this
  /// template argument list does not own the pointer.
  llvm::PointerIntPair<const TemplateArgument *, 1> FlatArguments;

  /// \brief The number of template arguments in this template
  /// argument list.
  unsigned NumFlatArguments;

  llvm::PointerIntPair<const TemplateArgument *, 1> StructuredArguments;
  unsigned NumStructuredArguments;

public:
  TemplateArgumentList(ASTContext &Context,
                       TemplateArgumentListBuilder &Builder,
                       bool TakeArgs);

  /// \brief Produces a shallow copy of the given template argument list
  TemplateArgumentList(const TemplateArgumentList &Other);
  
  ~TemplateArgumentList();

  /// \brief Retrieve the template argument at a given index.
  const TemplateArgument &get(unsigned Idx) const {
    assert(Idx < NumFlatArguments && "Invalid template argument index");
    return getFlatArgumentList()[Idx];
  }

  /// \brief Retrieve the template argument at a given index.
  const TemplateArgument &operator[](unsigned Idx) const { return get(Idx); }

  /// \brief Retrieve the number of template arguments in this
  /// template argument list.
  unsigned size() const { return NumFlatArguments; }

  /// \brief Retrieve the number of template arguments in the
  /// flattened template argument list.
  unsigned flat_size() const { return NumFlatArguments; }

  /// \brief Retrieve the flattened template argument list.
  const TemplateArgument *getFlatArgumentList() const {
    return FlatArguments.getPointer();
  }
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
    : NamedDecl(DK, DC, L, Name), TemplatedDecl(0), TemplateParams(0) { }

  // Construct a template decl with the given name and parameters.
  // Used when there is not templated element (tt-params, alias?).
  TemplateDecl(Kind DK, DeclContext *DC, SourceLocation L,
               DeclarationName Name, TemplateParameterList *Params)
    : NamedDecl(DK, DC, L, Name), TemplatedDecl(0), TemplateParams(Params) { }

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

/// \brief Provides information about a function template specialization,
/// which is a FunctionDecl that has been explicitly specialization or
/// instantiated from a function template.
class FunctionTemplateSpecializationInfo : public llvm::FoldingSetNode {
public:
  /// \brief The function template specialization that this structure
  /// describes.
  FunctionDecl *Function;

  /// \brief The function template from which this function template
  /// specialization was generated.
  ///
  /// The two bits are contain the top 4 values of TemplateSpecializationKind.
  llvm::PointerIntPair<FunctionTemplateDecl *, 2> Template;

  /// \brief The template arguments used to produce the function template
  /// specialization from the function template.
  const TemplateArgumentList *TemplateArguments;

  /// \brief The point at which this function template specialization was
  /// first instantiated. 
  SourceLocation PointOfInstantiation;
  
  /// \brief Retrieve the template from which this function was specialized.
  FunctionTemplateDecl *getTemplate() const { return Template.getPointer(); }

  /// \brief Determine what kind of template specialization this is.
  TemplateSpecializationKind getTemplateSpecializationKind() const {
    return (TemplateSpecializationKind)(Template.getInt() + 1);
  }

  /// \brief Set the template specialization kind.
  void setTemplateSpecializationKind(TemplateSpecializationKind TSK) {
    assert(TSK != TSK_Undeclared &&
         "Cannot encode TSK_Undeclared for a function template specialization");
    Template.setInt(TSK - 1);
  }

  /// \brief Retrieve the first point of instantiation of this function
  /// template specialization.
  ///
  /// The point of instantiation may be an invalid source location if this
  /// function has yet to be instantiated.
  SourceLocation getPointOfInstantiation() const { 
    return PointOfInstantiation; 
  }
  
  /// \brief Set the (first) point of instantiation of this function template
  /// specialization.
  void setPointOfInstantiation(SourceLocation POI) {
    PointOfInstantiation = POI;
  }
  
  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, TemplateArguments->getFlatArgumentList(),
            TemplateArguments->flat_size(),
            Function->getASTContext());
  }

  static void
  Profile(llvm::FoldingSetNodeID &ID, const TemplateArgument *TemplateArgs,
          unsigned NumTemplateArgs, ASTContext &Context) {
    ID.AddInteger(NumTemplateArgs);
    for (unsigned Arg = 0; Arg != NumTemplateArgs; ++Arg)
      TemplateArgs[Arg].Profile(ID, Context);
  }
};

/// \brief Provides information a specialization of a member of a class 
/// template, which may be a member function, static data member, or
/// member class.
class MemberSpecializationInfo {
  // The member declaration from which this member was instantiated, and the
  // manner in which the instantiation occurred (in the lower two bits).
  llvm::PointerIntPair<NamedDecl *, 2> MemberAndTSK;
  
  // The point at which this member was first instantiated.
  SourceLocation PointOfInstantiation;
  
public:
  explicit 
  MemberSpecializationInfo(NamedDecl *IF, TemplateSpecializationKind TSK)
    : MemberAndTSK(IF, TSK - 1), PointOfInstantiation() {
    assert(TSK != TSK_Undeclared && 
           "Cannot encode undeclared template specializations for members");
  }
  
  /// \brief Retrieve the member declaration from which this member was
  /// instantiated.
  NamedDecl *getInstantiatedFrom() const { return MemberAndTSK.getPointer(); }
  
  /// \brief Determine what kind of template specialization this is.
  TemplateSpecializationKind getTemplateSpecializationKind() const {
    return (TemplateSpecializationKind)(MemberAndTSK.getInt() + 1);
  }
  
  /// \brief Set the template specialization kind.
  void setTemplateSpecializationKind(TemplateSpecializationKind TSK) {
    assert(TSK != TSK_Undeclared && 
           "Cannot encode undeclared template specializations for members");
    MemberAndTSK.setInt(TSK - 1);
  }
  
  /// \brief Retrieve the first point of instantiation of this member. 
  /// If the point of instantiation is an invalid location, then this member
  /// has not yet been instantiated.
  SourceLocation getPointOfInstantiation() const { 
    return PointOfInstantiation; 
  }
  
  /// \brief Set the first point of instantiation.
  void setPointOfInstantiation(SourceLocation POI) {
    PointOfInstantiation = POI;
  }
};
  
/// Declaration of a template function.
class FunctionTemplateDecl : public TemplateDecl {
protected:
  /// \brief Data that is common to all of the declarations of a given
  /// function template.
  struct Common {
    Common() : InstantiatedFromMember(0, false) { }

    /// \brief The function template specializations for this function
    /// template, including explicit specializations and instantiations.
    llvm::FoldingSet<FunctionTemplateSpecializationInfo> Specializations;

    /// \brief The member function template from which this was most
    /// directly instantiated (or null).
    ///
    /// The boolean value indicates whether this member function template
    /// was explicitly specialized.
    llvm::PointerIntPair<FunctionTemplateDecl*, 1, bool> InstantiatedFromMember;
  };

  /// \brief A pointer to the previous declaration (if this is a redeclaration)
  /// or to the data that is common to all declarations of this function
  /// template.
  llvm::PointerUnion<Common*, FunctionTemplateDecl*> CommonOrPrev;

  /// \brief Retrieves the "common" pointer shared by all
  /// (re-)declarations of the same function template. Calling this routine
  /// may implicitly allocate memory for the common pointer.
  Common *getCommonPtr();

  FunctionTemplateDecl(DeclContext *DC, SourceLocation L, DeclarationName Name,
                       TemplateParameterList *Params, NamedDecl *Decl)
    : TemplateDecl(FunctionTemplate, DC, L, Name, Params, Decl),
      CommonOrPrev((Common*)0) { }

public:
  void Destroy(ASTContext &C);

  /// Get the underlying function declaration of the template.
  FunctionDecl *getTemplatedDecl() const {
    return static_cast<FunctionDecl*>(TemplatedDecl);
  }

  /// \brief Retrieve the set of function template specializations of this
  /// function template.
  llvm::FoldingSet<FunctionTemplateSpecializationInfo> &getSpecializations() {
    return getCommonPtr()->Specializations;
  }

  /// \brief Retrieve the previous declaration of this function template, or
  /// NULL if no such declaration exists.
  const FunctionTemplateDecl *getPreviousDeclaration() const {
    return CommonOrPrev.dyn_cast<FunctionTemplateDecl*>();
  }

  /// \brief Retrieve the previous declaration of this function template, or
  /// NULL if no such declaration exists.
  FunctionTemplateDecl *getPreviousDeclaration() {
    return CommonOrPrev.dyn_cast<FunctionTemplateDecl*>();
  }

  /// \brief Set the previous declaration of this function template.
  void setPreviousDeclaration(FunctionTemplateDecl *Prev) {
    if (Prev)
      CommonOrPrev = Prev;
  }

  virtual FunctionTemplateDecl *getCanonicalDecl();

  /// \brief Retrieve the member function template that this function template
  /// was instantiated from.
  ///
  /// This routine will return non-NULL for member function templates of
  /// class templates.  For example, given:
  ///
  /// \code
  /// template <typename T>
  /// struct X {
  ///   template <typename U> void f();
  /// };
  /// \endcode
  ///
  /// X<int>::A<float> is a CXXMethodDecl (whose parent is X<int>, a
  /// ClassTemplateSpecializationDecl) for which getPrimaryTemplate() will
  /// return X<int>::f, a FunctionTemplateDecl (whose parent is again
  /// X<int>) for which getInstantiatedFromMemberTemplate() will return
  /// X<T>::f, a FunctionTemplateDecl (whose parent is X<T>, a
  /// ClassTemplateDecl).
  ///
  /// \returns NULL if this is not an instantiation of a member function
  /// template.
  FunctionTemplateDecl *getInstantiatedFromMemberTemplate() {
    return getCommonPtr()->InstantiatedFromMember.getPointer();
  }

  void setInstantiatedFromMemberTemplate(FunctionTemplateDecl *FTD) {
    assert(!getCommonPtr()->InstantiatedFromMember.getPointer());
    getCommonPtr()->InstantiatedFromMember.setPointer(FTD);
  }

  /// \brief Determines whether this template was a specialization of a 
  /// member template.
  ///
  /// In the following example, the function template \c X<int>::f is a 
  /// member specialization.
  ///
  /// \code
  /// template<typename T>
  /// struct X {
  ///   template<typename U> void f(T, U);
  /// };
  ///
  /// template<> template<typename T>
  /// void X<int>::f(int, T);
  /// \endcode
  bool isMemberSpecialization() {
    return getCommonPtr()->InstantiatedFromMember.getInt();
  }
  
  /// \brief Note that this member template is a specialization.
  void setMemberSpecialization() {
    assert(getCommonPtr()->InstantiatedFromMember.getPointer() &&
           "Only member templates can be member template specializations");
    getCommonPtr()->InstantiatedFromMember.setInt(true);
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
class TemplateParmPosition {
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

  /// Get the index of the template parameter within its parameter list.
  unsigned getIndex() const { return Position; }
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

  /// \brief Whether this is a parameter pack.
  bool ParameterPack : 1;

  /// \brief The default template argument, if any.
  DeclaratorInfo *DefaultArgument;

  TemplateTypeParmDecl(DeclContext *DC, SourceLocation L, IdentifierInfo *Id,
                       bool Typename, QualType Type, bool ParameterPack)
    : TypeDecl(TemplateTypeParm, DC, L, Id), Typename(Typename),
      InheritedDefault(false), ParameterPack(ParameterPack), DefaultArgument() {
    TypeForDecl = Type.getTypePtr();
  }

public:
  static TemplateTypeParmDecl *Create(ASTContext &C, DeclContext *DC,
                                      SourceLocation L, unsigned D, unsigned P,
                                      IdentifierInfo *Id, bool Typename,
                                      bool ParameterPack);

  /// \brief Whether this template type parameter was declared with
  /// the 'typename' keyword. If not, it was declared with the 'class'
  /// keyword.
  bool wasDeclaredWithTypename() const { return Typename; }

  /// \brief Determine whether this template parameter has a default
  /// argument.
  bool hasDefaultArgument() const { return DefaultArgument != 0; }

  /// \brief Retrieve the default argument, if any.
  QualType getDefaultArgument() const { return DefaultArgument->getType(); }

  /// \brief Retrieves the default argument's source information, if any.
  DeclaratorInfo *getDefaultArgumentInfo() const { return DefaultArgument; }

  /// \brief Retrieves the location of the default argument declaration.
  SourceLocation getDefaultArgumentLoc() const;

  /// \brief Determines whether the default argument was inherited
  /// from a previous declaration of this template.
  bool defaultArgumentWasInherited() const { return InheritedDefault; }

  /// \brief Set the default argument for this template parameter, and
  /// whether that default argument was inherited from another
  /// declaration.
  void setDefaultArgument(DeclaratorInfo *DefArg, bool Inherited) {
    DefaultArgument = DefArg;
    InheritedDefault = Inherited;
  }

  /// \brief Removes the default argument of this template parameter.
  void removeDefaultArgument() {
    DefaultArgument = 0;
    InheritedDefault = false;
  }

  /// \brief Retrieve the depth of the template parameter.
  unsigned getDepth() const;
  
  /// \brief Retrieve the index of the template parameter.
  unsigned getIndex() const;

  /// \brief Returns whether this is a parameter pack.
  bool isParameterPack() const { return ParameterPack; }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == TemplateTypeParm;
  }
  static bool classof(const TemplateTypeParmDecl *D) { return true; }
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
                          DeclaratorInfo *DInfo)
    : VarDecl(NonTypeTemplateParm, DC, L, Id, T, DInfo, VarDecl::None),
      TemplateParmPosition(D, P), DefaultArgument(0)
  { }

public:
  static NonTypeTemplateParmDecl *
  Create(ASTContext &C, DeclContext *DC, SourceLocation L, unsigned D,
         unsigned P, IdentifierInfo *Id, QualType T, DeclaratorInfo *DInfo);

  using TemplateParmPosition::getDepth;
  using TemplateParmPosition::getPosition;
  using TemplateParmPosition::getIndex;

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
  using TemplateParmPosition::getIndex;

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
};

/// \brief Represents a class template specialization, which refers to
/// a class template with a given set of template arguments.
///
/// Class template specializations represent both explicit
/// specialization of class templates, as in the example below, and
/// implicit instantiations of class templates.
///
/// \code
/// template<typename T> class array;
///
/// template<>
/// class array<bool> { }; // class template specialization array<bool>
/// \endcode
class ClassTemplateSpecializationDecl
  : public CXXRecordDecl, public llvm::FoldingSetNode {

  /// \brief Structure that stores information about a class template
  /// specialization that was instantiated from a class template partial
  /// specialization.
  struct SpecializedPartialSpecialization {
    /// \brief The class template partial specialization from which this
    /// class template specialization was instantiated.
    ClassTemplatePartialSpecializationDecl *PartialSpecialization;

    /// \brief The template argument list deduced for the class template
    /// partial specialization itself.
    TemplateArgumentList *TemplateArgs;
  };

  /// \brief The template that this specialization specializes
  llvm::PointerUnion<ClassTemplateDecl *, SpecializedPartialSpecialization *>
    SpecializedTemplate;

  /// \brief The template arguments used to describe this specialization.
  TemplateArgumentList TemplateArgs;

  /// \brief The point where this template was instantiated (if any)
  SourceLocation PointOfInstantiation;

  /// \brief The kind of specialization this declaration refers to.
  /// Really a value of type TemplateSpecializationKind.
  unsigned SpecializationKind : 3;

protected:
  ClassTemplateSpecializationDecl(ASTContext &Context, Kind DK,
                                  DeclContext *DC, SourceLocation L,
                                  ClassTemplateDecl *SpecializedTemplate,
                                  TemplateArgumentListBuilder &Builder,
                                  ClassTemplateSpecializationDecl *PrevDecl);

public:
  static ClassTemplateSpecializationDecl *
  Create(ASTContext &Context, DeclContext *DC, SourceLocation L,
         ClassTemplateDecl *SpecializedTemplate,
         TemplateArgumentListBuilder &Builder,
         ClassTemplateSpecializationDecl *PrevDecl);

  virtual void Destroy(ASTContext& C);

  virtual void getNameForDiagnostic(std::string &S,
                                    const PrintingPolicy &Policy,
                                    bool Qualified) const;

  /// \brief Retrieve the template that this specialization specializes.
  ClassTemplateDecl *getSpecializedTemplate() const;

  /// \brief Retrieve the template arguments of the class template
  /// specialization.
  const TemplateArgumentList &getTemplateArgs() const {
    return TemplateArgs;
  }

  /// \brief Determine the kind of specialization that this
  /// declaration represents.
  TemplateSpecializationKind getSpecializationKind() const {
    return static_cast<TemplateSpecializationKind>(SpecializationKind);
  }

  void setSpecializationKind(TemplateSpecializationKind TSK) {
    SpecializationKind = TSK;
  }

  /// \brief Get the point of instantiation (if any), or null if none.
  SourceLocation getPointOfInstantiation() const {
    return PointOfInstantiation;
  }

  void setPointOfInstantiation(SourceLocation Loc) {
    assert(Loc.isValid() && "point of instantiation must be valid!");
    PointOfInstantiation = Loc;
  }

  /// \brief If this class template specialization is an instantiation of
  /// a template (rather than an explicit specialization), return the
  /// class template or class template partial specialization from which it
  /// was instantiated.
  llvm::PointerUnion<ClassTemplateDecl *,
                     ClassTemplatePartialSpecializationDecl *>
  getInstantiatedFrom() const {
    if (getSpecializationKind() != TSK_ImplicitInstantiation &&
        getSpecializationKind() != TSK_ExplicitInstantiationDefinition &&
        getSpecializationKind() != TSK_ExplicitInstantiationDeclaration)
      return (ClassTemplateDecl*)0;

    if (SpecializedPartialSpecialization *PartialSpec
          = SpecializedTemplate.dyn_cast<SpecializedPartialSpecialization*>())
      return PartialSpec->PartialSpecialization;

    return const_cast<ClassTemplateDecl*>(
                             SpecializedTemplate.get<ClassTemplateDecl*>());
  }

  /// \brief Retrieve the set of template arguments that should be used
  /// to instantiate members of the class template or class template partial
  /// specialization from which this class template specialization was
  /// instantiated.
  ///
  /// \returns For a class template specialization instantiated from the primary
  /// template, this function will return the same template arguments as
  /// getTemplateArgs(). For a class template specialization instantiated from
  /// a class template partial specialization, this function will return the
  /// deduced template arguments for the class template partial specialization
  /// itself.
  const TemplateArgumentList &getTemplateInstantiationArgs() const {
    if (SpecializedPartialSpecialization *PartialSpec
        = SpecializedTemplate.dyn_cast<SpecializedPartialSpecialization*>())
      return *PartialSpec->TemplateArgs;

    return getTemplateArgs();
  }

  /// \brief Note that this class template specialization is actually an
  /// instantiation of the given class template partial specialization whose
  /// template arguments have been deduced.
  void setInstantiationOf(ClassTemplatePartialSpecializationDecl *PartialSpec,
                          TemplateArgumentList *TemplateArgs) {
    SpecializedPartialSpecialization *PS
      = new (getASTContext()) SpecializedPartialSpecialization();
    PS->PartialSpecialization = PartialSpec;
    PS->TemplateArgs = TemplateArgs;
    SpecializedTemplate = PS;
  }

  /// \brief Sets the type of this specialization as it was written by
  /// the user. This will be a class template specialization type.
  void setTypeAsWritten(QualType T) {
    TypeForDecl = T.getTypePtr();
  }

  void Profile(llvm::FoldingSetNodeID &ID) const {
    Profile(ID, TemplateArgs.getFlatArgumentList(), TemplateArgs.flat_size(),
            getASTContext());
  }

  static void
  Profile(llvm::FoldingSetNodeID &ID, const TemplateArgument *TemplateArgs,
          unsigned NumTemplateArgs, ASTContext &Context) {
    ID.AddInteger(NumTemplateArgs);
    for (unsigned Arg = 0; Arg != NumTemplateArgs; ++Arg)
      TemplateArgs[Arg].Profile(ID, Context);
  }

  static bool classof(const Decl *D) {
    return D->getKind() == ClassTemplateSpecialization ||
           D->getKind() == ClassTemplatePartialSpecialization;
  }

  static bool classof(const ClassTemplateSpecializationDecl *) {
    return true;
  }

  static bool classof(const ClassTemplatePartialSpecializationDecl *) {
    return true;
  }
};

class ClassTemplatePartialSpecializationDecl
  : public ClassTemplateSpecializationDecl {
  /// \brief The list of template parameters
  TemplateParameterList* TemplateParams;

  /// \brief The source info for the template arguments as written.
  TemplateArgumentLoc *ArgsAsWritten;
  unsigned NumArgsAsWritten;

  /// \brief The class template partial specialization from which this 
  /// class template partial specialization was instantiated.
  ///
  /// The boolean value will be true to indicate that this class template
  /// partial specialization was specialized at this level.
  llvm::PointerIntPair<ClassTemplatePartialSpecializationDecl *, 1, bool>
      InstantiatedFromMember;
    
  ClassTemplatePartialSpecializationDecl(ASTContext &Context,
                                         DeclContext *DC, SourceLocation L,
                                         TemplateParameterList *Params,
                                         ClassTemplateDecl *SpecializedTemplate,
                                         TemplateArgumentListBuilder &Builder,
                                         TemplateArgumentLoc *ArgInfos,
                                         unsigned NumArgInfos,
                               ClassTemplatePartialSpecializationDecl *PrevDecl)
    : ClassTemplateSpecializationDecl(Context,
                                      ClassTemplatePartialSpecialization,
                                      DC, L, SpecializedTemplate, Builder,
                                      PrevDecl),
      TemplateParams(Params), ArgsAsWritten(ArgInfos),
      NumArgsAsWritten(NumArgInfos), InstantiatedFromMember(0, false) { }

public:
  static ClassTemplatePartialSpecializationDecl *
  Create(ASTContext &Context, DeclContext *DC, SourceLocation L,
         TemplateParameterList *Params,
         ClassTemplateDecl *SpecializedTemplate,
         TemplateArgumentListBuilder &Builder,
         TemplateArgumentLoc *ArgInfos,
         unsigned NumArgInfos,
         ClassTemplatePartialSpecializationDecl *PrevDecl);

  /// Get the list of template parameters
  TemplateParameterList *getTemplateParameters() const {
    return TemplateParams;
  }

  /// Get the template arguments as written.
  TemplateArgumentLoc *getTemplateArgsAsWritten() const {
    return ArgsAsWritten;
  }

  /// Get the number of template arguments as written.
  unsigned getNumTemplateArgsAsWritten() const {
    return NumArgsAsWritten;
  }

  /// \brief Retrieve the member class template partial specialization from
  /// which this particular class template partial specialization was
  /// instantiated.
  ///
  /// \code
  /// template<typename T>
  /// struct Outer {
  ///   template<typename U> struct Inner;
  ///   template<typename U> struct Inner<U*> { }; // #1
  /// };
  ///
  /// Outer<float>::Inner<int*> ii;
  /// \endcode
  ///
  /// In this example, the instantiation of \c Outer<float>::Inner<int*> will
  /// end up instantiating the partial specialization 
  /// \c Outer<float>::Inner<U*>, which itself was instantiated from the class 
  /// template partial specialization \c Outer<T>::Inner<U*>. Given 
  /// \c Outer<float>::Inner<U*>, this function would return
  /// \c Outer<T>::Inner<U*>.
  ClassTemplatePartialSpecializationDecl *getInstantiatedFromMember() {
    ClassTemplatePartialSpecializationDecl *First
      = cast<ClassTemplatePartialSpecializationDecl>(getFirstDeclaration());
    return First->InstantiatedFromMember.getPointer();
  }
  
  void setInstantiatedFromMember(
                          ClassTemplatePartialSpecializationDecl *PartialSpec) {
    ClassTemplatePartialSpecializationDecl *First
      = cast<ClassTemplatePartialSpecializationDecl>(getFirstDeclaration());
    First->InstantiatedFromMember.setPointer(PartialSpec);
  }
    
  /// \brief Determines whether this class template partial specialization 
  /// template was a specialization of a member partial specialization.
  ///
  /// In the following example, the member template partial specialization
  /// \c X<int>::Inner<T*> is a member specialization.
  ///
  /// \code
  /// template<typename T>
  /// struct X {
  ///   template<typename U> struct Inner;
  ///   template<typename U> struct Inner<U*>;
  /// };
  ///
  /// template<> template<typename T>
  /// struct X<int>::Inner<T*> { /* ... */ };
  /// \endcode
  bool isMemberSpecialization() {
    ClassTemplatePartialSpecializationDecl *First
      = cast<ClassTemplatePartialSpecializationDecl>(getFirstDeclaration());
    return First->InstantiatedFromMember.getInt();
  }
  
  /// \brief Note that this member template is a specialization.
  void setMemberSpecialization() {
    ClassTemplatePartialSpecializationDecl *First
      = cast<ClassTemplatePartialSpecializationDecl>(getFirstDeclaration());
    assert(First->InstantiatedFromMember.getPointer() &&
           "Only member templates can be member template specializations");
    return First->InstantiatedFromMember.setInt(true);
  }
    
  // FIXME: Add Profile support!

  static bool classof(const Decl *D) {
    return D->getKind() == ClassTemplatePartialSpecialization;
  }

  static bool classof(const ClassTemplatePartialSpecializationDecl *) {
    return true;
  }
};

/// Declaration of a class template.
class ClassTemplateDecl : public TemplateDecl {
protected:
  /// \brief Data that is common to all of the declarations of a given
  /// class template.
  struct Common {
    Common() : InstantiatedFromMember(0, 0) {}

    /// \brief The class template specializations for this class
    /// template, including explicit specializations and instantiations.
    llvm::FoldingSet<ClassTemplateSpecializationDecl> Specializations;

    /// \brief The class template partial specializations for this class
    /// template.
    llvm::FoldingSet<ClassTemplatePartialSpecializationDecl>
      PartialSpecializations;

    /// \brief The injected-class-name type for this class template.
    QualType InjectedClassNameType;

    /// \brief The templated member class from which this was most
    /// directly instantiated (or null).
    ///
    /// The boolean value indicates whether this member class template
    /// was explicitly specialized.
    llvm::PointerIntPair<ClassTemplateDecl *, 1, bool> InstantiatedFromMember;
  };

  // FIXME: Combine PreviousDeclaration with CommonPtr, as in 
  // FunctionTemplateDecl.
  
  /// \brief Previous declaration of this class template.
  ClassTemplateDecl *PreviousDeclaration;

  /// \brief Pointer to the data that is common to all of the
  /// declarations of this class template.
  ///
  /// The first declaration of a class template (e.g., the declaration
  /// with no "previous declaration") owns this pointer.
  Common *CommonPtr;

  ClassTemplateDecl(DeclContext *DC, SourceLocation L, DeclarationName Name,
                    TemplateParameterList *Params, NamedDecl *Decl,
                    ClassTemplateDecl *PrevDecl, Common *CommonPtr)
    : TemplateDecl(ClassTemplate, DC, L, Name, Params, Decl),
      PreviousDeclaration(PrevDecl), CommonPtr(CommonPtr) { }

  ~ClassTemplateDecl();

public:
  /// Get the underlying class declarations of the template.
  CXXRecordDecl *getTemplatedDecl() const {
    return static_cast<CXXRecordDecl *>(TemplatedDecl);
  }

  /// \brief Retrieve the previous declaration of this template.
  ClassTemplateDecl *getPreviousDeclaration() const {
    return PreviousDeclaration;
  }

  virtual ClassTemplateDecl *getCanonicalDecl();

  /// Create a class template node.
  static ClassTemplateDecl *Create(ASTContext &C, DeclContext *DC,
                                   SourceLocation L,
                                   DeclarationName Name,
                                   TemplateParameterList *Params,
                                   NamedDecl *Decl,
                                   ClassTemplateDecl *PrevDecl);

  /// \brief Retrieve the set of specializations of this class template.
  llvm::FoldingSet<ClassTemplateSpecializationDecl> &getSpecializations() {
    return CommonPtr->Specializations;
  }

  /// \brief Retrieve the set of partial specializations of this class
  /// template.
  llvm::FoldingSet<ClassTemplatePartialSpecializationDecl> &
  getPartialSpecializations() {
    return CommonPtr->PartialSpecializations;
  }

  /// \brief Find a class template partial specialization with the given
  /// type T.
  ///
  /// \brief A dependent type that names a specialization of this class
  /// template.
  ///
  /// \returns the class template partial specialization that exactly matches
  /// the type \p T, or NULL if no such partial specialization exists.
  ClassTemplatePartialSpecializationDecl *findPartialSpecialization(QualType T);

  /// \brief Retrieve the type of the injected-class-name for this
  /// class template.
  ///
  /// The injected-class-name for a class template \c X is \c
  /// X<template-args>, where \c template-args is formed from the
  /// template arguments that correspond to the template parameters of
  /// \c X. For example:
  ///
  /// \code
  /// template<typename T, int N>
  /// struct array {
  ///   typedef array this_type; // "array" is equivalent to "array<T, N>"
  /// };
  /// \endcode
  QualType getInjectedClassNameType(ASTContext &Context);

  /// \brief Retrieve the member class template that this class template was
  /// derived from.
  ///
  /// This routine will return non-NULL for templated member classes of
  /// class templates.  For example, given:
  ///
  /// \code
  /// template <typename T>
  /// struct X {
  ///   template <typename U> struct A {};
  /// };
  /// \endcode
  ///
  /// X<int>::A<float> is a ClassTemplateSpecializationDecl (whose parent
  /// is X<int>, also a CTSD) for which getSpecializedTemplate() will
  /// return X<int>::A<U>, a TemplateClassDecl (whose parent is again
  /// X<int>) for which getInstantiatedFromMemberTemplate() will return
  /// X<T>::A<U>, a TemplateClassDecl (whose parent is X<T>, also a TCD).
  ///
  /// \returns null if this is not an instantiation of a member class template.
  ClassTemplateDecl *getInstantiatedFromMemberTemplate() const {
    return CommonPtr->InstantiatedFromMember.getPointer();
  }

  void setInstantiatedFromMemberTemplate(ClassTemplateDecl *CTD) {
    assert(!CommonPtr->InstantiatedFromMember.getPointer());
    CommonPtr->InstantiatedFromMember.setPointer(CTD);
  }

  /// \brief Determines whether this template was a specialization of a 
  /// member template.
  ///
  /// In the following example, the member template \c X<int>::Inner is a 
  /// member specialization.
  ///
  /// \code
  /// template<typename T>
  /// struct X {
  ///   template<typename U> struct Inner;
  /// };
  ///
  /// template<> template<typename T>
  /// struct X<int>::Inner { /* ... */ };
  /// \endcode
  bool isMemberSpecialization() {
    return CommonPtr->InstantiatedFromMember.getInt();
  }
  
  /// \brief Note that this member template is a specialization.
  void setMemberSpecialization() {
    assert(CommonPtr->InstantiatedFromMember.getPointer() &&
           "Only member templates can be member template specializations");
    CommonPtr->InstantiatedFromMember.setInt(true);
  }
  
  // Implement isa/cast/dyncast support
  static bool classof(const Decl *D)
  { return D->getKind() == ClassTemplate; }
  static bool classof(const ClassTemplateDecl *D)
  { return true; }

  virtual void Destroy(ASTContext& C);
};

/// Declaration of a friend template.  For example:
///
/// template <typename T> class A {
///   friend class MyVector<T>; // not a friend template
///   template <typename U> friend class B; // friend template
///   template <typename U> friend class Foo<T>::Nested; // friend template
class FriendTemplateDecl : public Decl {
public:
  typedef llvm::PointerUnion<NamedDecl*,Type*> FriendUnion;

private:
  // The number of template parameters;  always non-zero.
  unsigned NumParams;

  // The parameter list.
  TemplateParameterList **Params;

  // The declaration that's a friend of this class.
  FriendUnion Friend;

  // Location of the 'friend' specifier.
  SourceLocation FriendLoc;


  FriendTemplateDecl(DeclContext *DC, SourceLocation Loc,
                     unsigned NParams, 
                     TemplateParameterList **Params,
                     FriendUnion Friend,
                     SourceLocation FriendLoc)
    : Decl(Decl::FriendTemplate, DC, Loc),
      NumParams(NParams),
      Params(Params),
      Friend(Friend),
      FriendLoc(FriendLoc)
  {}

public:
  static FriendTemplateDecl *Create(ASTContext &Context,
                                    DeclContext *DC, SourceLocation Loc,
                                    unsigned NParams, 
                                    TemplateParameterList **Params,
                                    FriendUnion Friend,
                                    SourceLocation FriendLoc);

  /// If this friend declaration names a templated type (or
  /// a dependent member type of a templated type), return that
  /// type;  otherwise return null.
  Type *getFriendType() const {
    return Friend.dyn_cast<Type*>();
  }

  /// If this friend declaration names a templated function (or
  /// a member function of a templated type), return that type;
  /// otherwise return null.
  NamedDecl *getFriendDecl() const {
    return Friend.dyn_cast<NamedDecl*>();
  }

  /// Retrieves the location of the 'friend' keyword.
  SourceLocation getFriendLoc() const {
    return FriendLoc;
  }

  TemplateParameterList *getTemplateParameterList(unsigned i) const {
    assert(i <= NumParams);
    return Params[i];
  }

  unsigned getNumTemplateParameters() const {
    return NumParams;
  }

  // Implement isa/cast/dyncast/etc.
  static bool classof(const Decl *D) {
    return D->getKind() == Decl::FriendTemplate;
  }
  static bool classof(const FriendTemplateDecl *D) { return true; }
};

/// Implementation of inline functions that require the template declarations
inline AnyFunctionDecl::AnyFunctionDecl(FunctionTemplateDecl *FTD)
  : Function(FTD) { }

} /* end of namespace clang */

#endif
