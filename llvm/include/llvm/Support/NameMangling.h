//===- llvm/Support/NameMangling.h - Name Mangling for LLVM ------*- C++ -*--=//
//
// This file implements a consistent scheme for name mangling symbols.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_NAMEMANGLING_H
#define LLVM_SUPPORT_NAMEMANGLING_H

#include <string>
class Type;
class Value;

// MangleTypeName - Implement a consistent name-mangling scheme for
//                  a given type.
// 
string MangleTypeName(const Type *type);

// MangleName - implement a consistent name-mangling scheme for all
// externally visible (i.e., global) objects.
//
// privateName should be unique within the module.
// 
string MangleName(const string &privateName, const Value *V);

#endif

