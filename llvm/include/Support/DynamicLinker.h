//===-- DynamicLinker.h - System-indep. DynamicLinker interface -*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// Lightweight interface to dynamic library linking and loading, and dynamic
// symbol lookup functionality, in whatever form the operating system
// provides it.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DYNAMICLINKER_H
#define SUPPORT_DYNAMICLINKER_H

#include <string>

/// LinkDynamicObject - Load the named file as a dynamic library
/// and link it with the currently running process. Returns false
/// on success, true if there is an error (and sets ErrorMessage
/// if it is not NULL). Analogous to dlopen().
///
bool LinkDynamicObject (const char *filename, std::string *ErrorMessage);

/// GetAddressOfSymbol - Returns the address of the named symbol in
/// the currently running process, as reported by the dynamic linker,
/// or NULL if the symbol does not exist or some other error has
/// occurred.
/// 
void *GetAddressOfSymbol (const char *symbolName);
void *GetAddressOfSymbol (const std::string &symbolName);

#endif // SUPPORT_DYNAMICLINKER_H
