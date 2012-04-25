//===- llvm/ADT/MultiImplMap.h - 'Normally small' pointer set ----*- C++ -*-==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MultiImplMap class.
// MultiImplMap presents map container interface.
// It has two modes, one for small amount of elements and one for big amount.
// User should set map implementation for both of them. User also should
// set the maximum possible number of elements for small mode.
// If user want to use MultiImplMap instead of DenseMap, he should pass
// DenseMapCompatible = true. Note that in this case map implementations should
// present additional DenseMap specific methods (see below).
// Initially MultiImplMap uses small mode and small map implementation.
// It triggered to the big mode when number of contained elements exceeds
// maximum possible elements for small mode.
//
// Types that should be defined in nested map class:
//
//    key_type;
//    mapped_type;
//    value_type; // std::pair<key_type, mapped_type>
//                // or std::pair<const key_type, mapped_type>
//    iterator;
//    const_iterator;
//
// Map implementation should provide the next interface:
//
//    // Constructors
//    (default constructor)
//    (copy constructor)
//
//    // Size
//    unsigned size() const;
//    bool empty() const;
//
//    // Iterators
//    iterator begin();
//    const_iterator begin();
//    iterator end();
//    const_iterator end();
//
//    // Modifiers
//    void clear();
//    std::pair<iterator, bool> insert(const value_type& KV);
//    template <typename IterT>
//      void insert(IterT I, IterT E);
//    void erase(key_type K);
//    void erase(iterator i);
//    void swap(MultiImplMap& rhs);
//
//    // Search operations
//    iterator find(const key_type& K);
//    const_iterator find(const key_type& K) const;
//    bool count(const key_type& K) const;
//    mapped_type &operator[](const key_type &Key);
//
//    // Other operations
//    self& operator=(const self& other);
//
//    // If DenseMapCompatible == true, you also should present next methods.
//    // See DenseMap comments for more details about its behavior.
//    bool isPointerIntoBucketsArray(const void *Ptr) const;
//    const void *getPointerIntoBucketsArray() const;
//    value_type& FindAndConstruct(const key_type &Key);
//
// The list of methods that should be implemented in nested map iterator class:
//
//    (conversion constructor from non-constant iterator)
//
//    bool operator==(const const_iterator& rhs) const;
//    bool operator!=(const const_iterator& rhs) const;
//    reference operator*() const;
//    pointer operator->() const;
//    inline self& operator++();
//
//
//===----------------------------------------------------------------------===//

#ifndef MULTIIMPLEMENTATIONMAP_H_
#define MULTIIMPLEMENTATIONMAP_H_


#include <algorithm>
#include <utility>
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/FlatArrayMap.h"
#include "llvm/Support/type_traits.h"

namespace llvm {

  template<class SmallMapTy, class BigMapTy, bool IsConst = false>
  class MultiImplMapIterator;

  template<class SmallMapTy, class BigMapTy>
    struct MultiImplMapIteratorsFactory;

  template<class SmallMapTy, class BigMapTy>
  struct MultiImplMapTypes {
    typedef typename SmallMapTy::key_type key_type;
    typedef typename SmallMapTy::mapped_type mapped_type;
    typedef typename std::pair<key_type, mapped_type> value_type;
  };

  //===--------------------------------------------------------------------===//
  /// MultiImplMap is map that has two modes, one for small amount of
  /// elements and one for big amount.
  /// User should set map implementation for both of them. User also should
  /// set the maximum possible number of elements for small mode.
  /// If user want to use MultiImplMap instead of DenseMap, he should pass
  /// DenseMapCompatible = true.
  /// Initially MultiImplMap uses small mode and small map implementation.
  /// It triggered to the big mode when number of contained elements exceeds
  /// maximum possible elements for small mode.
  template<class SmallMapTy, class BigMapTy, unsigned MaxSmallN,
           bool DenseMapCompatible = false,
           class ItFactory =
               MultiImplMapIteratorsFactory<SmallMapTy, BigMapTy> >
  class MultiImplMap {

  protected:
    SmallMapTy SmallMap;
    BigMapTy BigMap;
    bool UseSmall;
    enum { MaxSmallSize = MaxSmallN };

  public:
    typedef MultiImplMapTypes<SmallMapTy, BigMapTy> Types;

    typedef typename Types::key_type key_type;
    typedef typename Types::mapped_type mapped_type;
    typedef typename Types::value_type value_type;

    typedef typename ItFactory::iterator iterator;
    typedef typename ItFactory::const_iterator const_iterator;

    typedef std::pair<iterator, bool> ins_res;

    typedef typename std::pair<typename SmallMapTy::iterator, bool>
      small_ins_res;

    typedef typename std::pair<typename BigMapTy::iterator, bool>
      big_ins_res;

    typedef MultiImplMap<SmallMapTy, BigMapTy, MaxSmallN> self;

    MultiImplMap() : UseSmall(true) {}

    MultiImplMap(const self& other) {
      if (other.UseSmall) {
        SmallMap = other.SmallMap;
        UseSmall = true;
      } else {
        if (other.size() <= MaxSmallN) {
          SmallMap.insert(other.BigMap.begin(), other.BigMap.end());
          UseSmall = true;
        } else {
          BigMap = other.BigMap;
          UseSmall = false;
        }
      }
    }

    // Size

    unsigned size() const {
      if (UseSmall)
        return SmallMap.size();
      return BigMap.size();
    }

    bool empty() const {
      if (UseSmall)
        return SmallMap.empty();
      return BigMap.empty();
    }

    // Iterators

    iterator begin() {
      if (UseSmall)
        return ItFactory::begin(SmallMap);
      return ItFactory::begin(BigMap);
    }
    const_iterator begin() const {
      if (UseSmall)
        return ItFactory::begin(SmallMap);
      return ItFactory::begin(BigMap);
    }

    iterator end() {
      if (UseSmall)
        return ItFactory::end(SmallMap);
      return ItFactory::end(BigMap);
    }
    const_iterator end() const {
      if (UseSmall)
        return ItFactory::end(SmallMap);
      return ItFactory::end(BigMap);
    }

    // Modifiers

    void clear() {
      if (UseSmall)
        SmallMap.clear();
      else
        BigMap.clear();
    }

    std::pair<iterator, bool> insert(const value_type& KV) {
      if (UseSmall) {
        if (SmallMap.size() < MaxSmallSize) {
          small_ins_res Res = SmallMap.insert(KV);
          return std::make_pair(ItFactory::it(SmallMap, Res.first), Res.second);
        }

        // Move all to big map.
        BigMap.insert(SmallMap.begin(), SmallMap.end());
        SmallMap.clear();

        UseSmall = false;
      }
      big_ins_res Res = BigMap.insert(KV);
      return std::make_pair(ItFactory::it(BigMap, Res.first), Res.second);
    }

    template <typename OtherValTy>
    std::pair<iterator, bool> insert(const OtherValTy& OtherKV) {
      const value_type* KV = reinterpret_cast<const value_type*>(
          reinterpret_cast<const void*>(OtherKV));
      return insert(*KV);
    }

    template <typename IterT>
    void insert(IterT I, IterT E) {
      for (; I != E; ++I)
        insert(*I);
    }

    void erase(key_type K) {
      if (UseSmall)
        SmallMap.erase(K);
      else
        BigMap.erase(K);
    }

    void erase(iterator i) {
      erase(i->first);
    }

    void swap(MultiImplMap& rhs) {
      SmallMap.swap(rhs.SmallMap);
      BigMap.swap(rhs.BigMap);
      std::swap(UseSmall, rhs.UseSmall);
    }

    // Search operations

    iterator find(const key_type& K) {
      if (UseSmall)
        return ItFactory::it(SmallMap, SmallMap.find(K));
      return ItFactory::it(BigMap, BigMap.find(K));
    }

    const_iterator find(const key_type& K) const {
      if (UseSmall)
        return ItFactory::const_it(SmallMap, SmallMap.find(K));
      return ItFactory::const_it(BigMap, BigMap.find(K));
    }

    bool count(const key_type& K) const {
      return find(K) != end();
    }

    mapped_type &operator[](const key_type &Key) {
      ins_res res = insert(std::make_pair(Key, mapped_type()));
      return res.first->second;
    }

    // Other operations

    self& operator=(const self& other) {
      if (other.isSmall()) {
        SmallMap = other.SmallMap;
        if (!UseSmall) {
          BigMap.clear();
          UseSmall = true;
        }
        return *this;
      }
      if (UseSmall) {
        SmallMap.clear();
        UseSmall = false;
      }
      BigMap = other.BigMap;
      return *this;
    }

    // Utilities

    bool isSmall()const {
      return UseSmall;
    }

    SmallMapTy& getSmallMap() {
      return SmallMap;
    }

    const SmallMapTy& getSmallMap() const {
      return SmallMap;
    }

    BigMapTy& getBigMap() {
      return BigMap;
    }

    const BigMapTy& getBigMap() const {
      return BigMap;
    }
  };

  template<class SmallMapTy, class BigMapTy, unsigned MaxSmallN>
  class MultiImplMap<SmallMapTy, BigMapTy, MaxSmallN, true> :
        public MultiImplMap<SmallMapTy, BigMapTy, MaxSmallN, false>
  {
  public:
    typedef MultiImplMap<SmallMapTy, BigMapTy, MaxSmallN, false> ParentTy;
    typedef typename ParentTy::Types Types;

    typedef typename Types::key_type key_type;
    typedef typename Types::mapped_type mapped_type;
    typedef typename Types::value_type value_type;
    typedef typename ParentTy::iterator iterator;

    /// isPointerIntoBucketsArray - Return true if the specified pointer points
    /// somewhere into the DenseMap's array of buckets (i.e. either to a key or
    /// value).
    bool isPointerIntoBucketsArray(const void *Ptr) const {
      if (this->UseSmall)
        return this->SmallMap.isPointerIntoBucketsArray(Ptr);
      return this->BigMap.isPointerIntoBucketsArray(Ptr);
    }

    /// getPointerIntoBucketsArray() - Return an opaque pointer into the buckets
    /// array.  In conjunction with the previous method, this can be used to
    /// determine whether an insertion caused the map to reallocate data.
    const void *getPointerIntoBucketsArray() const {
      if (this->UseSmall)
        return this->SmallMap.getPointerIntoBucketsArray();
      return this->BigMap.getPointerIntoBucketsArray();
    }

    value_type& FindAndConstruct(const key_type &Key) {
      std::pair<iterator, bool> Res =
          this->insert(std::make_pair(Key, mapped_type()));
      return *Res.first;
    }
  };

  template<class SmallMapTy, class BigMapTy, bool IsConst>
  class MultiImplMapIterator {
  public:

    typedef MultiImplMapTypes<SmallMapTy, BigMapTy> Types;

    typedef typename Types::mapped_type mapped_type;

    typedef typename conditional<IsConst,
                                 const typename Types::value_type,
                                 typename Types::value_type>::type value_type;

    typedef typename conditional<IsConst,
                                 typename SmallMapTy::const_iterator,
                                 typename SmallMapTy::iterator>::type
                                 small_iterator;

    typedef typename conditional<IsConst,
                                 typename BigMapTy::const_iterator,
                                 typename BigMapTy::iterator>::type
                                 big_iterator;

    typedef typename conditional<IsConst, const void*, void*>::type void_ptr_ty;

    typedef value_type *pointer;
    typedef value_type &reference;

    typedef MultiImplMapIterator<SmallMapTy, BigMapTy, IsConst> self;

    typedef MultiImplMapIterator<SmallMapTy, BigMapTy, false> non_const_self;
    typedef MultiImplMapIterator<SmallMapTy, BigMapTy, true> const_self;

    friend class MultiImplMapIterator<SmallMapTy, BigMapTy, true>;
    friend class MultiImplMapIterator<SmallMapTy, BigMapTy, false>;

  protected:

    template <typename OtherValTy>
    static value_type* toValueTypePtr(OtherValTy& ValTyRef) {
      return reinterpret_cast<value_type*>(
               reinterpret_cast<void_ptr_ty>(&ValTyRef));
    }

    template <typename OtherValTy>
    static value_type& toValueTypeRef(OtherValTy& ValTyRef) {
      return *reinterpret_cast<value_type*>(
                reinterpret_cast<void_ptr_ty>(&ValTyRef));
    }

    small_iterator SmallIt;
    big_iterator BigIt;
    bool UseSmall;

  public:

    MultiImplMapIterator() : UseSmall(true) {}
    MultiImplMapIterator(small_iterator It) : SmallIt(It), UseSmall(true) {}
    MultiImplMapIterator(big_iterator It) : BigIt(It), UseSmall(false) {}
    MultiImplMapIterator(const non_const_self& src) :
      SmallIt(src.SmallIt), BigIt(src.BigIt), UseSmall(src.UseSmall) {}

    bool operator==(const const_self& rhs) const {
      if (UseSmall != rhs.UseSmall)
        return false;
      if (UseSmall)
        return SmallIt == rhs.SmallIt;
      return BigIt == rhs.BigIt;
    }

    bool operator!=(const const_self& rhs) const {
      if (UseSmall != rhs.UseSmall)
        return true;
      if (UseSmall)
        return SmallIt != rhs.SmallIt;
      return BigIt != rhs.BigIt;
    }

    reference operator*() const {
      return UseSmall ? toValueTypeRef(*SmallIt) : toValueTypeRef(*BigIt);;
    }

    pointer operator->() const {
      return UseSmall ? toValueTypePtr(*SmallIt) : toValueTypePtr(*BigIt);
    }

    // Preincrement
    inline self& operator++() {
      if (UseSmall) ++SmallIt;
      return *this;
    }

    // Postincrement
    self operator++(int) {
      self tmp = *this; ++*this; return tmp;
    }
  };

  template<class SmallMapTy, class BigMapTy>
  struct MultiImplMapIteratorsFactory {

    typedef MultiImplMapIterator<SmallMapTy, BigMapTy, false> iterator;
    typedef MultiImplMapIterator<SmallMapTy, BigMapTy, true> const_iterator;

    template<class MapImpl, class ItTy>
    static iterator it(MapImpl& impl, ItTy it) {
      return iterator(it);
    }
    template<class MapImpl, class ConstItTy>
    static const_iterator const_it(const MapImpl& impl, ConstItTy it) {
      return const_iterator(it);
    }
    template<class MapImpl>
    static iterator begin(MapImpl& impl) {
      return iterator(impl.begin());
    }
    template<class MapImpl>
    static const_iterator begin(const MapImpl& impl) {
      return const_iterator(impl.begin());
    }
    template<class MapImpl>
    static iterator end(MapImpl& impl) {
      return iterator(impl.end());
    }
    template<class MapImpl>
    static const_iterator end(const MapImpl& impl) {
      return const_iterator(impl.end());
    }
  };

  template<typename KeyTy, typename MappedTy, unsigned MaxArraySize,
            typename KeyInfoT>
  struct MultiImplMapIteratorsFactory<
          FlatArrayMap<KeyTy, MappedTy, MaxArraySize>,
          DenseMap<KeyTy, MappedTy, KeyInfoT> >
  {

    typedef FlatArrayMap<KeyTy, MappedTy, MaxArraySize> SmallMapTy;
    typedef DenseMap<KeyTy, MappedTy, KeyInfoT> BigMapTy;

    typedef DenseMapIterator<KeyTy, MappedTy, KeyInfoT, false>
      iterator;
    typedef DenseMapIterator<KeyTy, MappedTy, KeyInfoT, true>
      const_iterator;

    static iterator it(SmallMapTy& impl, typename SmallMapTy::iterator it) {
      return iterator(&(*it), &(*impl.end()));
    }
    static const_iterator const_it(
        const SmallMapTy& impl, typename SmallMapTy::const_iterator it) {
      return const_iterator(&(*it), &(*impl.end()));
    }
    static iterator it(BigMapTy& impl, typename BigMapTy::iterator it) {
      return it;
    }
    static const_iterator const_it(
        const BigMapTy& impl, typename BigMapTy::const_iterator it) {
      return it;
    }
    static iterator begin(SmallMapTy& impl) {
      return it(impl, impl.begin());
    }
    static const_iterator begin(const SmallMapTy& impl) {
      return it(impl, impl.begin());
    }
    static iterator begin(BigMapTy& impl) {
      return impl.begin();
    }
    static const_iterator begin(const BigMapTy& impl) {
      return impl.begin();
    }
    static iterator end(SmallMapTy& impl) {
      return it(impl, impl.end());
    }
    static const_iterator end(const SmallMapTy& impl) {
      return const_it(impl, impl.end());
    }
    static iterator end(BigMapTy& impl) {
      return impl.end();
    }
    static const_iterator end(const BigMapTy& impl) {
      return impl.end();
    }
  };
}

#endif /* MULTIIMPLEMENTATIONMAP_H_ */
