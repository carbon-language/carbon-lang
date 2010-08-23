//===--- ParsedTemplate.h - Template Parsing Data Types -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file provides data structures that store the parsed representation of
//  templates.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_SEMA_PARSEDTEMPLATE_H
#define LLVM_CLANG_SEMA_PARSEDTEMPLATE_H

#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Ownership.h"
#include <cassert>

namespace clang {  
  /// \brief Represents the parsed form of a C++ template argument.
  class ParsedTemplateArgument {
  public:
    /// \brief Describes the kind of template argument that was parsed.
    enum KindType {
      /// \brief A template type parameter, stored as a type.
      Type,
      /// \brief A non-type template parameter, stored as an expression.
      NonType,
      /// \brief A template template argument, stored as a template name.
      Template
    };

    /// \brief Build an empty template argument. This template argument 
    ParsedTemplateArgument() : Kind(Type), Arg(0) { }
    
    /// \brief Create a template type argument or non-type template argument.
    ///
    /// \param Arg the template type argument or non-type template argument.
    /// \param Loc the location of the type.
    ParsedTemplateArgument(KindType Kind, void *Arg, SourceLocation Loc)
      : Kind(Kind), Arg(Arg), Loc(Loc) { }
    
    /// \brief Create a template template argument.
    ///
    /// \param SS the C++ scope specifier that precedes the template name, if
    /// any.
    ///
    /// \param Template the template to which this template template 
    /// argument refers.
    ///
    /// \param TemplateLoc the location of the template name.
    ParsedTemplateArgument(const CXXScopeSpec &SS,
                           ActionBase::TemplateTy Template, 
                           SourceLocation TemplateLoc) 
      : Kind(ParsedTemplateArgument::Template), Arg(Template.get()), 
        Loc(TemplateLoc), SS(SS) { }
    
    /// \brief Determine whether the given template argument is invalid.
    bool isInvalid() const { return Arg == 0; }
    
    /// \brief Determine what kind of template argument we have.
    KindType getKind() const { return Kind; }
    
    /// \brief Retrieve the template type argument's type.
    ActionBase::TypeTy *getAsType() const {
      assert(Kind == Type && "Not a template type argument");
      return Arg;
    }
    
    /// \brief Retrieve the non-type template argument's expression.
    Expr *getAsExpr() const {
      assert(Kind == NonType && "Not a non-type template argument");
      return static_cast<Expr*>(Arg);
    }
    
    /// \brief Retrieve the template template argument's template name.
    ActionBase::TemplateTy getAsTemplate() const {
      assert(Kind == Template && "Not a template template argument");
      return ActionBase::TemplateTy::make(Arg);
    }
    
    /// \brief Retrieve the location of the template argument.
    SourceLocation getLocation() const { return Loc; }
    
    /// \brief Retrieve the nested-name-specifier that precedes the template
    /// name in a template template argument.
    const CXXScopeSpec &getScopeSpec() const {
      assert(Kind == Template && 
             "Only template template arguments can have a scope specifier");
      return SS;
    }
    
  private:
    KindType Kind;
    
    /// \brief The actual template argument representation, which may be
    /// an \c ActionBase::TypeTy* (for a type), an ActionBase::ExprTy* (for an
    /// expression), or an ActionBase::TemplateTy (for a template).
    void *Arg;

    /// \brief the location of the template argument.
    SourceLocation Loc;
    
    /// \brief The nested-name-specifier that can accompany a template template
    /// argument.
    CXXScopeSpec SS;
  };
  
  /// \brief Information about a template-id annotation
  /// token.
  ///
  /// A template-id annotation token contains the template declaration, 
  /// template arguments, whether those template arguments were types, 
  /// expressions, or template names, and the source locations for important 
  /// tokens. All of the information about template arguments is allocated 
  /// directly after this structure.
  struct TemplateIdAnnotation {
    /// TemplateNameLoc - The location of the template name within the
    /// source.
    SourceLocation TemplateNameLoc;
    
    /// FIXME: Temporarily stores the name of a specialization
    IdentifierInfo *Name;
    
    /// FIXME: Temporarily stores the overloaded operator kind.
    OverloadedOperatorKind Operator;
    
    /// The declaration of the template corresponding to the
    /// template-name. This is an Action::TemplateTy.
    void *Template;
    
    /// The kind of template that Template refers to.
    TemplateNameKind Kind;
    
    /// The location of the '<' before the template argument
    /// list.
    SourceLocation LAngleLoc;
    
    /// The location of the '>' after the template argument
    /// list.
    SourceLocation RAngleLoc;
    
    /// NumArgs - The number of template arguments.
    unsigned NumArgs;
    
    /// \brief Retrieves a pointer to the template arguments
    ParsedTemplateArgument *getTemplateArgs() { 
      return reinterpret_cast<ParsedTemplateArgument *>(this + 1); 
    }
    
    static TemplateIdAnnotation* Allocate(unsigned NumArgs) {
      TemplateIdAnnotation *TemplateId
      = (TemplateIdAnnotation *)std::malloc(sizeof(TemplateIdAnnotation) +
                                      sizeof(ParsedTemplateArgument) * NumArgs);
      TemplateId->NumArgs = NumArgs;
      return TemplateId;
    }
    
    void Destroy() { free(this); }
  };
  
  
  inline const ParsedTemplateArgument &
  ASTTemplateArgsPtr::operator[](unsigned Arg) const { 
    return Args[Arg]; 
  }
}

#endif
