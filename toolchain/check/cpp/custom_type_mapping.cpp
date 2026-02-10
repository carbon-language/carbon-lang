// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/check/cpp/custom_type_mapping.h"

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Check {

// A small, lightweight library of AST matchers. Unlike clang's ASTMatchers,
// this avoids heap allocations and is suitable for one-off matching rather than
// matching against a whole AST.
namespace Matchers {
// A matcher for a type T is just a function that takes a T and returns whether
// it matched. Matchers should be invoked immediately, and are not expected to
// outlive the arguments of the call that created them.
// TODO: We could avoid the indirect calls by making the below functions be
// templated on the inner matcher.
template <typename T>
using Matcher = llvm::function_ref<auto(T)->bool>;

// Returns a matcher for class declarations that determines whether the given
// class is a class template specialization in namespace std with the specified
// name and template arguments matching the given predicate.
static auto StdClassTemplate(
    llvm::StringLiteral name,
    Matcher<const clang::TemplateArgumentList&> args_matcher
    [[clang::lifetimebound]]) -> auto {
  return [=](const clang::CXXRecordDecl* class_decl) -> bool {
    const auto* specialization =
        dyn_cast<clang::ClassTemplateSpecializationDecl>(class_decl);
    const auto* identifier = class_decl->getIdentifier();
    return specialization && identifier && identifier->isStr(name) &&
           specialization->isInStdNamespace() &&
           args_matcher(specialization->getTemplateArgs());
  };
}

// Returns a matcher that matches types if they are class types whose class
// matches the given matcher.
static auto Class(Matcher<const clang::CXXRecordDecl*> class_matcher
                  [[clang::lifetimebound]]) -> auto {
  return [=](clang::QualType type) -> bool {
    const auto* class_decl = type->getAsCXXRecordDecl();
    return !type.hasQualifiers() && class_decl && class_matcher(class_decl);
  };
}

// Returns a matcher that determines whether the given template argument is a
// type matching the given predicate.
static auto TypeTemplateArgument(Matcher<clang::QualType> type_matcher
                                 [[clang::lifetimebound]]) -> auto {
  return [=](clang::TemplateArgument arg) -> bool {
    return arg.getKind() == clang::TemplateArgument::Type &&
           type_matcher(arg.getAsType());
  };
}

// A matcher that determines whether the given type is `char`.
static auto Char(clang::QualType type) -> bool {
  return !type.hasQualifiers() && type->isCharType();
}

// Returns a matcher that determines whether the given template argument list
// matches the given sequence of template argument matchers.
static auto TemplateArgumentsAre(
    std::initializer_list<Matcher<clang::TemplateArgument>> arg_matchers
    [[clang::lifetimebound]]) -> auto {
  return [=](const clang::TemplateArgumentList& args) -> bool {
    if (args.size() != arg_matchers.size()) {
      return false;
    }
    for (auto [arg, matcher] : llvm::zip_equal(args.asArray(), arg_matchers)) {
      if (!matcher(arg)) {
        return false;
      }
    }
    return true;
  };
}

// A matcher for `std::char_traits<char>`.
static auto StdCharTraitsChar(clang::QualType type) -> bool {
  return Class(StdClassTemplate(
      "char_traits", TemplateArgumentsAre({TypeTemplateArgument(Char)})))(type);
}

// A matcher for `std::string_view`.
static auto StdStringView(const clang::CXXRecordDecl* record_decl) -> bool {
  return StdClassTemplate(
      "basic_string_view",
      TemplateArgumentsAre({TypeTemplateArgument(Char),
                            TypeTemplateArgument(StdCharTraitsChar)}))(
      record_decl);
}
}  // end namespace Matchers

auto GetCustomCppTypeMapping(const clang::CXXRecordDecl* record_decl)
    -> CustomCppTypeMapping {
  if (Matchers::StdStringView(record_decl)) {
    return CustomCppTypeMapping::Str;
  }

  return CustomCppTypeMapping::None;
}

}  // namespace Carbon::Check
