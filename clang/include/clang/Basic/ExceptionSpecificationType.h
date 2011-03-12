//===--- ExceptionSpecificationType.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the ExceptionSpecificationType enumeration and various
// utility functions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_CLANG_BASIC_EXCEPTIONSPECIFICATIONTYPE_H
#define LLVM_CLANG_BASIC_EXCEPTIONSPECIFICATIONTYPE_H

namespace clang {

/// \brief The various types of exception specifications that exist in C++0x.
enum ExceptionSpecificationType {
  EST_None,            ///< no exception specification
  EST_DynamicNone,     ///< throw()
  EST_Dynamic,         ///< throw(T1, T2)
  EST_MSAny,           ///< Microsoft throw(...) extension
  EST_BasicNoexcept,   ///< noexcept
  EST_ComputedNoexcept ///< noexcept(expression)
};

inline bool isDynamicExceptionSpec(ExceptionSpecificationType ESpecType) {
  return ESpecType >= EST_DynamicNone && ESpecType <= EST_MSAny;
}

inline bool isNoexceptExceptionSpec(ExceptionSpecificationType ESpecType) {
  return ESpecType == EST_BasicNoexcept || ESpecType == EST_ComputedNoexcept;
}

} // end namespace clang

#endif // LLVM_CLANG_BASIC_EXCEPTIONSPECIFICATIONTYPE_H
