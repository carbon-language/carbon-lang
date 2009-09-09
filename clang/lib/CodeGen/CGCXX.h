//===----- CGCXX.h - C++ related code CodeGen declarations ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// These classes wrap the information about a call or function
// definition used to handle ABI compliancy.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CODEGEN_CGCXX_H
#define CLANG_CODEGEN_CGCXX_H

namespace clang {

/// CXXCtorType - C++ constructor types
enum CXXCtorType {
    Ctor_Complete,          // Complete object ctor
    Ctor_Base,              // Base object ctor
    Ctor_CompleteAllocating // Complete object allocating ctor
};

/// CXXDtorType - C++ destructor types
enum CXXDtorType {
    Dtor_Deleting, // Deleting dtor
    Dtor_Complete, // Complete object dtor
    Dtor_Base      // Base object dtor
};

} // end namespace clang

#endif // CLANG_CODEGEN_CGCXX_H
