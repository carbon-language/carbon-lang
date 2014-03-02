//===-- PassByValue.h -------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the declaration of the PassByValueTransform
/// class.
///
//===----------------------------------------------------------------------===//

#ifndef CLANG_MODERNIZE_PASS_BY_VALUE_H
#define CLANG_MODERNIZE_PASS_BY_VALUE_H

#include "Core/IncludeDirectives.h"
#include "Core/Transform.h"

class ConstructorParamReplacer;

/// \brief Subclass of Transform that uses pass-by-value semantic when move
/// constructors are available to avoid copies.
///
/// When a class constructor accepts an object by const reference with the
/// intention of copying the object the copy can be avoided in certain
/// situations if the object has a move constructor. First, the constructor is
/// changed to accept the object by value instead. Then this argument is moved
/// instead of copied into class-local storage. If an l-value is provided to the
/// constructor, there is no difference in the number of copies made. However,
/// if an r-value is passed, the copy is avoided completely.
///
/// For example, given:
/// \code
/// #include <string>
///
/// class A {
///   std::string S;
/// public:
///   A(const std::string &S) : S(S) {}
/// };
/// \endcode
/// the code is transformed to:
/// \code
/// #include <string>
///
/// class A {
///   std::string S;
/// public:
///   A(std::string S) : S(std::move(S)) {}
/// };
/// \endcode
class PassByValueTransform : public Transform {
public:
  PassByValueTransform(const TransformOptions &Options)
      : Transform("PassByValue", Options), Replacer(0) {}

  /// \see Transform::apply().
  virtual int apply(const clang::tooling::CompilationDatabase &Database,
                    const std::vector<std::string> &SourcePaths) override;

private:
  /// \brief Setups the \c IncludeDirectives for the replacer.
  virtual bool handleBeginSource(clang::CompilerInstance &CI,
                                 llvm::StringRef Filename) override;

  llvm::OwningPtr<IncludeDirectives> IncludeManager;
  ConstructorParamReplacer *Replacer;
};

#endif // CLANG_MODERNIZE_PASS_BY_VALUE_H
