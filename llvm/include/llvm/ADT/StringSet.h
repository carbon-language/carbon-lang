//===--- StringSet.h - The LLVM Compiler Driver -----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open
// Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  StringSet - A set-like wrapper for the StringMap.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_STRINGSET_H
#define LLVM_ADT_STRINGSET_H

#include "llvm/ADT/StringMap.h"

namespace llvm {

  /// StringSet - A wrapper for StringMap that provides set-like functionality.
  template <class AllocatorTy = llvm::MallocAllocator>
  class StringSet : public llvm::StringMap<char, AllocatorTy> {
    typedef llvm::StringMap<char, AllocatorTy> base;
  public:

    /// insert - Insert the specified key into the set.  If the key already
    /// exists in the set, return false and ignore the request, otherwise insert
    /// it and return true.
    bool insert(StringRef Key) {
      // Get or create the map entry for the key; if it doesn't exist the value
      // type will be default constructed which we use to detect insert.
      //
      // We use '+' as the sentinel value in the map.
      assert(!Key.empty());
      StringMapEntry<char> &Entry = this->GetOrCreateValue(Key);
      if (Entry.getValue() == '+')
        return false;
      Entry.setValue('+');
      return true;
    }
  };
}

#endif // LLVM_ADT_STRINGSET_H
