//===- DenseMap.h - A dense map implmentation -------------------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file implements a dense map. A dense map template takes two
// types. The first is the mapped type and the second is a functor
// that maps its argument to a size_t. On instanciation a "null" value
// can be provided to be used as a "does not exist" indicator in the
// map. A member function grow() is provided that given the value of
// the maximally indexed key (the argument of the functor) makes sure
// the map has enough space for it.
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_DENSEMAP_H
#define SUPPORT_DENSEMAP_H

#include <vector>

namespace llvm {

template <typename T, typename ToIndexT>
class DenseMap {
    typedef typename ToIndexT::argument_type IndexT;
    typedef std::vector<T> StorageT;
    StorageT storage_;
    T nullVal_;
    ToIndexT toIndex_;

public:
    DenseMap() { }

    explicit DenseMap(const T& val) : nullVal_(val) { }

    typename StorageT::reference operator[](IndexT n) {
        assert(toIndex_(n) < storage_.size() && "index out of bounds!");
        return storage_[toIndex_(n)];
    }

    typename StorageT::const_reference operator[](IndexT n) const {
        assert(toIndex_(n) < storage_.size() && "index out of bounds!");
        return storage_[toIndex_(n)];
    }

    void clear() {
        storage_.assign(storage_.size(), nullVal_);
    }

    void grow(IndexT n) {
        storage_.resize(toIndex_(n) + 1, nullVal_);
    }
};

} // End llvm namespace

#endif
