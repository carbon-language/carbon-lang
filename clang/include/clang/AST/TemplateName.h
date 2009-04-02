//===--- TemplateName.h - C++ Template Name Representation-------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TemplateName interface and subclasses.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TEMPLATENAME_H
#define LLVM_CLANG_AST_TEMPLATENAME_H

#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/PointerUnion.h"

namespace llvm {
  class raw_ostream;
}

namespace clang {

class DependentTemplateName;
class IdentifierInfo;
class NestedNameSpecifier;
class QualifiedTemplateName;
class TemplateDecl;

/// \brief Represents a C++ template name within the type system.
///
/// A C++ template name refers to a template within the C++ type
/// system. In most cases, a template name is simply a reference to a
/// class template, e.g.
///
/// \code
/// template<typename T> class X { };
///
/// X<int> xi;
/// \endcode
///
/// Here, the 'X' in \c X<int> is a template name that refers to the
/// declaration of the class template X, above. Template names can
/// also refer to function templates, C++0x template aliases, etc.
///
/// Some template names are dependent. For example, consider:
///
/// \code
/// template<typename MetaFun, typename T1, typename T2> struct apply2 {
///   typedef typename MetaFun::template apply<T1, T2>::type type;
/// };
/// \endcode
///
/// Here, "apply" is treated as a template name within the typename
/// specifier in the typedef. "apply" is a nested template, and can
/// only be understood in the context of
class TemplateName {
  typedef llvm::PointerUnion3<TemplateDecl *, QualifiedTemplateName *, 
                              DependentTemplateName *> StorageType;

  StorageType Storage;

  explicit TemplateName(void *Ptr) {
    Storage = StorageType::getFromOpaqueValue(Ptr);
  }

public:
  TemplateName() : Storage() { }
  explicit TemplateName(TemplateDecl *Template) : Storage(Template) { }
  explicit TemplateName(QualifiedTemplateName *Qual) : Storage(Qual) { }
  explicit TemplateName(DependentTemplateName *Dep) : Storage(Dep) { }

  /// \brief Retrieve the the underlying template declaration that
  /// this template name refers to, if known.
  ///
  /// \returns The template declaration that this template name refers
  /// to, if any. If the template name does not refer to a specific
  /// declaration because it is a dependent name, returns NULL.
  TemplateDecl *getAsTemplateDecl() const;

  /// \brief Retrieve the underlying qualified template name
  /// structure, if any.
  QualifiedTemplateName *getAsQualifiedTemplateName() const {
    return Storage.dyn_cast<QualifiedTemplateName *>();
  }

  /// \brief Retrieve the underlying dependent template name
  /// structure, if any.
  DependentTemplateName *getAsDependentTemplateName() const {
    return Storage.dyn_cast<DependentTemplateName *>();
  }

  /// \brief Determines whether this is a dependent template name.
  bool isDependent() const;

  /// \brief Print the template name.
  ///
  /// \param OS the output stream to which the template name will be
  /// printed.
  ///
  /// \param SuppressNNS if true, don't print the
  /// nested-name-specifier that precedes the template name (if it has
  /// one).
  void print(llvm::raw_ostream &OS, bool SuppressNNS = false) const;

  /// \brief Debugging aid that dumps the template name to standard
  /// error.
  void dump() const;

  void Profile(llvm::FoldingSetNodeID &ID) {
    ID.AddPointer(Storage.getOpaqueValue());
  }

  /// \brief Retrieve the template name as a void pointer.
  void *getAsVoidPointer() const { return Storage.getOpaqueValue(); }

  /// \brief Build a template name from a void pointer.
  static TemplateName getFromVoidPointer(void *Ptr) { 
    return TemplateName(Ptr); 
  }
};

/// \brief Represents a template name that was expressed as a
/// qualified name.
///
/// This kind of template name refers to a template name that was
/// preceded by a nested name specifier, e.g., \c std::vector. Here,
/// the nested name specifier is "std::" and the template name is the
/// declaration for "vector". The QualifiedTemplateName class is only
/// used to provide "sugar" for template names that were expressed
/// with a qualified name, and has no semantic meaning. In this
/// manner, it is to TemplateName what QualifiedNameType is to Type,
/// providing extra syntactic sugar for downstream clients.
class QualifiedTemplateName : public llvm::FoldingSetNode {
  /// \brief The nested name specifier that qualifies the template name.
  ///
  /// The bit is used to indicate whether the "template" keyword was
  /// present before the template name itself. Note that the
  /// "template" keyword is always redundant in this case (otherwise,
  /// the template name would be a dependent name and we would express
  /// this name with DependentTemplateName).
  llvm::PointerIntPair<NestedNameSpecifier *, 1> Qualifier;

  /// \brief The template declaration that this qualified name refers
  /// to.
  TemplateDecl *Template;

  friend class ASTContext;

  QualifiedTemplateName(NestedNameSpecifier *NNS, bool TemplateKeyword,
                        TemplateDecl *Template)
    : Qualifier(NNS, TemplateKeyword? 1 : 0), Template(Template) { }

public:
  /// \brief Return the nested name specifier that qualifies this name.
  NestedNameSpecifier *getQualifier() const { return Qualifier.getPointer(); }

  /// \brief Whether the template name was prefixed by the "template"
  /// keyword.
  bool hasTemplateKeyword() const { return Qualifier.getInt(); }

  /// \brief The template declaration to which this qualified name
  /// refers.
  TemplateDecl *getTemplateDecl() const { return Template; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getQualifier(), hasTemplateKeyword(), getTemplateDecl());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, NestedNameSpecifier *NNS, 
                      bool TemplateKeyword, TemplateDecl *Template) {
    ID.AddPointer(NNS);
    ID.AddBoolean(TemplateKeyword);
    ID.AddPointer(Template);
  }
};

/// \brief Represents a dependent template name that cannot be
/// resolved prior to template instantiation.
///
/// This kind of template name refers to a dependent template name,
/// including its nested name specifier. For example,
/// DependentTemplateName can refer to "MetaFun::template apply",
/// where "MetaFun::" is the nested name specifier and "apply" is the
/// template name referenced. The "template" keyword is implied.
class DependentTemplateName : public llvm::FoldingSetNode {
  /// \brief The nested name specifier that qualifies the template
  /// name.
  NestedNameSpecifier *Qualifier;

  /// \brief The dependent template name.
  const IdentifierInfo *Name;

  /// \brief The canonical template name to which this dependent
  /// template name refers.
  ///
  /// The canonical template name for a dependent template name is
  /// another dependent template name whose nested name specifier is
  /// canonical.
  TemplateName CanonicalTemplateName;

  friend class ASTContext;

  DependentTemplateName(NestedNameSpecifier *Qualifier, 
                        const IdentifierInfo *Name)
    : Qualifier(Qualifier), Name(Name), CanonicalTemplateName(this) { }

  DependentTemplateName(NestedNameSpecifier *Qualifier, 
                        const IdentifierInfo *Name,
                        TemplateName Canon)
    : Qualifier(Qualifier), Name(Name), CanonicalTemplateName(Canon) { }

public:
  /// \brief Return the nested name specifier that qualifies this name.
  NestedNameSpecifier *getQualifier() const { return Qualifier; }

  /// \brief Return the name to which this dependent template name
  /// refers.
  const IdentifierInfo *getName() const { return Name; }

  void Profile(llvm::FoldingSetNodeID &ID) {
    Profile(ID, getQualifier(), getName());
  }

  static void Profile(llvm::FoldingSetNodeID &ID, NestedNameSpecifier *NNS, 
                      const IdentifierInfo *Name) {
    ID.AddPointer(NNS);
    ID.AddPointer(Name);
  }
};

} // end namespace clang.

namespace llvm {

/// \brief The clang::TemplateName class is effectively a pointer.
template<>
class PointerLikeTypeTraits<clang::TemplateName> {
public:
  static inline void *getAsVoidPointer(clang::TemplateName TN) {
    return TN.getAsVoidPointer();
  }

  static inline clang::TemplateName getFromVoidPointer(void *Ptr) {
    return clang::TemplateName::getFromVoidPointer(Ptr);
  }

  // No bits are available!
  enum { NumLowBitsAvailable = 0 };
};

} // end namespace llvm.

#endif
