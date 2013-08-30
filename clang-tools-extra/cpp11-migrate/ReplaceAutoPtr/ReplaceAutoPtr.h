//===-- ReplaceAutoPtr.h ------------ std::auto_ptr replacement -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file provides the declaration of the ReplaceAutoPtrTransform
/// class.
///
//===----------------------------------------------------------------------===//

#ifndef CPP11_MIGRATE_REPLACE_AUTO_PTR_H
#define CPP11_MIGRATE_REPLACE_AUTO_PTR_H

#include "Core/Transform.h"
#include "llvm/Support/Compiler.h"

/// \brief Subclass of Transform that transforms the deprecated \c std::auto_ptr
/// into the C++11 \c std::unique_ptr.
///
/// Note that both the \c std::auto_ptr type and the transfer of ownership are
/// transformed. \c std::auto_ptr provides two ways to transfer the ownership,
/// the copy-constructor and the assignment operator. Unlike most classes theses
/// operations do not 'copy' the resource but they 'steal' it.
/// \c std::unique_ptr uses move semantics instead, which makes the intent of
/// transferring the resource explicit. This difference between the two smart
/// pointers requires to wrap the copy-ctor and assign-operator with
/// \c std::move().
///
/// For example, given:
/// \code
///   std::auto_ptr<int> i, j;
///   i = j;
/// \endcode
/// the code is transformed to:
/// \code
///   std::unique_ptr<int> i, j;
///   i = std::move(j);
/// \endcode
class ReplaceAutoPtrTransform : public Transform {
public:
  ReplaceAutoPtrTransform(const TransformOptions &Options)
      : Transform("ReplaceAutoPtr", Options) {}

  /// \see Transform::run().
  virtual int apply(FileOverrides &InputStates,
                    const clang::tooling::CompilationDatabase &Database,
                    const std::vector<std::string> &SourcePaths) LLVM_OVERRIDE;
};

#endif // CPP11_MIGRATE_REPLACE_AUTO_PTR_H
