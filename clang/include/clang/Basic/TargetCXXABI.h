//===--- TargetCXXABI.h - C++ ABI Target Configuration ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the TargetCXXABI class, which abstracts details of the
/// C++ ABI that we're targeting.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TARGETCXXABI_H
#define LLVM_CLANG_TARGETCXXABI_H

#include "llvm/ADT/Triple.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {

/// \brief The basic abstraction for the target C++ ABI.
class TargetCXXABI {
public:
  /// \brief The basic C++ ABI kind.
  enum Kind {
    /// The generic Itanium ABI is the standard ABI of most open-source
    /// and Unix-like platforms.  It is the primary ABI targeted by
    /// many compilers, including Clang and GCC.
    ///
    /// It is documented here:
    ///   http://www.codesourcery.com/public/cxx-abi/
    GenericItanium,

    /// The generic ARM ABI is a modified version of the Itanium ABI
    /// proposed by ARM for use on ARM-based platforms.
    ///
    /// These changes include:
    ///   - the representation of member function pointers is adjusted
    ///     to not conflict with the 'thumb' bit of ARM function pointers;
    ///   - constructors and destructors return 'this';
    ///   - guard variables are smaller;
    ///   - inline functions are never key functions;
    ///   - array cookies have a slightly different layout;
    ///   - additional convenience functions are specified;
    ///   - and more!
    ///
    /// It is documented here:
    ///    http://infocenter.arm.com
    ///                    /help/topic/com.arm.doc.ihi0041c/IHI0041C_cppabi.pdf
    GenericARM,

    /// The iOS ABI is a partial implementation of the ARM ABI.
    /// Several of the features of the ARM ABI were not fully implemented
    /// in the compilers that iOS was launched with.
    ///
    /// Essentially, the iOS ABI includes the ARM changes to:
    ///   - member function pointers,
    ///   - guard variables,
    ///   - array cookies, and
    ///   - constructor/destructor signatures.
    iOS,

    /// The Microsoft ABI is the ABI used by Microsoft Visual Studio (and
    /// compatible compilers).
    ///
    /// FIXME: should this be split into Win32 and Win64 variants?
    ///
    /// Only scattered and incomplete official documentation exists.
    Microsoft
  };

private:
  // Right now, this class is passed around as a cheap value type.
  // If you add more members, especially non-POD members, please
  // audit the users to pass it by reference instead.
  Kind TheKind;

public:
  /// A bogus initialization of the platform ABI.
  TargetCXXABI() : TheKind(GenericItanium) {}

  TargetCXXABI(Kind kind) : TheKind(kind) {}

  void set(Kind kind) {
    TheKind = kind;
  }

  Kind getKind() const { return TheKind; }

  /// \brief Does this ABI generally fall into the Itanium family of ABIs?
  bool isItaniumFamily() const {
    switch (getKind()) {
    case GenericItanium:
    case GenericARM:
    case iOS:
      return true;

    case Microsoft:
      return false;
    }
    llvm_unreachable("bad ABI kind");
  }

  /// \brief Is this ABI an MSVC-compatible ABI?
  bool isMicrosoft() const {
    switch (getKind()) {
    case GenericItanium:
    case GenericARM:
    case iOS:
      return false;

    case Microsoft:
      return true;
    }
    llvm_unreachable("bad ABI kind");
  }

  /// \brief Is the default C++ member function calling convention
  /// the same as the default calling convention?
  bool isMemberFunctionCCDefault() const {
    // Right now, this is always true for Microsoft.
    return !isMicrosoft();
  }

  /// \brief Does this ABI have different entrypoints for complete-object
  /// and base-subobject constructors?
  bool hasConstructorVariants() const {
    return isItaniumFamily();
  }

  /// \brief Does this ABI have different entrypoints for complete-object
  /// and base-subobject destructors?
  bool hasDestructorVariants() const {
    return isItaniumFamily();
  }

  /// \brief Does this ABI allow virtual bases to be primary base classes?
  bool hasPrimaryVBases() const {
    return isItaniumFamily();
  }

  /// Try to parse an ABI name, returning false on error.
  bool tryParse(llvm::StringRef name);

  friend bool operator==(const TargetCXXABI &left, const TargetCXXABI &right) {
    return left.getKind() == right.getKind();
  }

  friend bool operator!=(const TargetCXXABI &left, const TargetCXXABI &right) {
    return !(left == right);
  }
};

}  // end namespace clang

#endif
