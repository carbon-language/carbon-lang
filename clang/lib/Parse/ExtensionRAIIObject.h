//===--- ExtensionRAIIObject.h - Use RAII for __extension__ -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines and implements the ExtensionRAIIObject class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_PARSE_EXTENSION_RAII_OBJECT_H
#define LLVM_CLANG_PARSE_EXTENSION_RAII_OBJECT_H

#include "clang/Parse/ParseDiagnostic.h"

namespace clang {

  /// ExtensionRAIIObject - This saves the state of extension warnings when
  /// constructed and disables them.  When destructed, it restores them back to
  /// the way they used to be.  This is used to handle __extension__ in the
  /// parser.
  class ExtensionRAIIObject {
    void operator=(const ExtensionRAIIObject &);     // DO NOT IMPLEMENT
    ExtensionRAIIObject(const ExtensionRAIIObject&); // DO NOT IMPLEMENT
    Diagnostic &Diags;
  public:
    ExtensionRAIIObject(Diagnostic &diags) : Diags(diags) {
      Diags.IncrementAllExtensionsSilenced();
    }
    
    ~ExtensionRAIIObject() {
      Diags.DecrementAllExtensionsSilenced();
    }
  };
}

#endif
