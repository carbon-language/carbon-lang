//===- llvm/ADT/SmallMap.h - 'Normally small' pointer set -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the SmallMap class.
// SmallMap is DenseMap compatible MultiImplMap. 
// It uses FlatArrayMap for small mode, and DenseMap for big mode. 
// See MultiMapImpl comments for more details on the algorithm is used. 
//
//===----------------------------------------------------------------------===//

#ifndef SMALLPTRMAP_H_
#define SMALLPTRMAP_H_

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FlatArrayMap.h"
#include "llvm/ADT/MultiImplMap.h"

namespace llvm {
  
  //===--------------------------------------------------------------------===//
  /// SmallMap is wrapper around MultiImplMap. It uses FlatArrayMap for
  /// small mode, and DenseMap for big mode. 
  template <typename KeyTy, typename MappedTy, unsigned N = 16>
  class SmallMap : public MultiImplMap<
                        FlatArrayMap<KeyTy, MappedTy, N>,
                        DenseMap<KeyTy, MappedTy>,
                        N, true> {
  };
}

#endif /* SMALLPTRMAP_H_ */
