//===-- ubsan_type_hash.cc ------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of a hash table for fast checking of inheritance
// relationships. This file is only linked into C++ compilations, and is
// permitted to use language features which require a C++ ABI library.
//
//===----------------------------------------------------------------------===//

#include "ubsan_type_hash.h"

#include "sanitizer_common/sanitizer_common.h"

// The following are intended to be binary compatible with the definitions
// given in the Itanium ABI. We make no attempt to be ODR-compatible with
// those definitions, since existing ABI implementations aren't.

namespace std {
  class type_info {
  public:
    virtual ~type_info();

    const char *__type_name;
  };
}

namespace __cxxabiv1 {

/// Type info for classes with no bases, and base class for type info for
/// classes with bases.
class __class_type_info : public std::type_info {
  virtual ~__class_type_info();
};

/// Type info for classes with simple single public inheritance.
class __si_class_type_info : public __class_type_info {
public:
  virtual ~__si_class_type_info();

  const __class_type_info *__base_type;
};

class __base_class_type_info {
public:
  const __class_type_info *__base_type;
  long __offset_flags;

  enum __offset_flags_masks {
    __virtual_mask = 0x1,
    __public_mask = 0x2,
    __offset_shift = 8
  };
};

/// Type info for classes with multiple, virtual, or non-public inheritance.
class __vmi_class_type_info : public __class_type_info {
public:
  virtual ~__vmi_class_type_info();

  unsigned int flags;
  unsigned int base_count;
  __base_class_type_info base_info[1];
};

}

namespace abi = __cxxabiv1;

// We implement a simple two-level cache for type-checking results. For each
// (vptr,type) pair, a hash is computed. This hash is assumed to be globally
// unique; if it collides, we will get false negatives, but:
//  * such a collision would have to occur on the *first* bad access,
//  * the probability of such a collision is low (and for a 64-bit target, is
//    negligible), and
//  * the vptr, and thus the hash, can be affected by ASLR, so multiple runs
//    give better coverage.
//
// The first caching layer is a small hash table with no chaining; buckets are
// reused as needed. The second caching layer is a large hash table with open
// chaining. We can freely evict from either layer since this is just a cache.
//
// FIXME: Make these hash table accesses thread-safe. The races here are benign:
//        assuming the unsequenced loads and stores don't misbehave too badly,
//        the worst case is false negatives or poor cache behavior, not false
//        positives or crashes.

/// Find a bucket to store the given hash value in.
static __ubsan::HashValue *getTypeCacheHashTableBucket(__ubsan::HashValue V) {
  static const unsigned HashTableSize = 65537;
  static __ubsan::HashValue __ubsan_vptr_hash_set[HashTableSize];

  unsigned First = (V & 65535) ^ 1;
  unsigned Probe = First;
  for (int Tries = 5; Tries; --Tries) {
    if (!__ubsan_vptr_hash_set[Probe] || __ubsan_vptr_hash_set[Probe] == V)
      return &__ubsan_vptr_hash_set[Probe];
    Probe += ((V >> 16) & 65535) + 1;
    if (Probe >= HashTableSize)
      Probe -= HashTableSize;
  }
  // FIXME: Pick a random entry from the probe sequence to evict rather than
  //        just taking the first.
  return &__ubsan_vptr_hash_set[First];
}

/// A cache of recently-checked hashes. Mini hash table with "random" evictions.
__ubsan::HashValue
__ubsan::__ubsan_vptr_type_cache[__ubsan::VptrTypeCacheSize];

/// \brief Determine whether \p Derived has a \p Base base class subobject at
/// offset \p Offset.
static bool isDerivedFromAtOffset(const abi::__class_type_info *Derived,
                                  const abi::__class_type_info *Base,
                                  sptr Offset) {
  if (Derived->__type_name == Base->__type_name)
    return Offset == 0;

  if (const abi::__si_class_type_info *SI =
        dynamic_cast<const abi::__si_class_type_info*>(Derived))
    return isDerivedFromAtOffset(SI->__base_type, Base, Offset);

  const abi::__vmi_class_type_info *VTI =
    dynamic_cast<const abi::__vmi_class_type_info*>(Derived);
  if (!VTI)
    // No base class subobjects.
    return false;

  // Look for a base class which is derived from \p Base at the right offset.
  for (unsigned int base = 0; base != VTI->base_count; ++base) {
    // FIXME: Curtail the recursion if this base can't possibly contain the
    //        given offset.
    sptr OffsetHere = VTI->base_info[base].__offset_flags >>
                      abi::__base_class_type_info::__offset_shift;
    if (VTI->base_info[base].__offset_flags &
          abi::__base_class_type_info::__virtual_mask)
      // For now, just punt on virtual bases and say 'yes'.
      // FIXME: OffsetHere is the offset in the vtable of the virtual base
      //        offset. Read the vbase offset out of the vtable and use it.
      return true;
    if (isDerivedFromAtOffset(VTI->base_info[base].__base_type,
                              Base, Offset - OffsetHere))
      return true;
  }

  return false;
}

/// \brief Find the derived-most dynamic base class of \p Derived at offset
/// \p Offset.
static const abi::__class_type_info *findBaseAtOffset(
    const abi::__class_type_info *Derived, sptr Offset) {
  if (!Offset)
    return Derived;

  if (const abi::__si_class_type_info *SI =
        dynamic_cast<const abi::__si_class_type_info*>(Derived))
    return findBaseAtOffset(SI->__base_type, Offset);

  const abi::__vmi_class_type_info *VTI =
    dynamic_cast<const abi::__vmi_class_type_info*>(Derived);
  if (!VTI)
    // No base class subobjects.
    return 0;

  for (unsigned int base = 0; base != VTI->base_count; ++base) {
    sptr OffsetHere = VTI->base_info[base].__offset_flags >>
                      abi::__base_class_type_info::__offset_shift;
    if (VTI->base_info[base].__offset_flags &
          abi::__base_class_type_info::__virtual_mask)
      // FIXME: Can't handle virtual bases yet.
      continue;
    if (const abi::__class_type_info *Base =
          findBaseAtOffset(VTI->base_info[base].__base_type,
                           Offset - OffsetHere))
      return Base;
  }

  return 0;
}

namespace {

struct VtablePrefix {
  /// The offset from the vptr to the start of the most-derived object.
  /// This should never be greater than zero, and will usually be exactly
  /// zero.
  sptr Offset;
  /// The type_info object describing the most-derived class type.
  std::type_info *TypeInfo;
};
VtablePrefix *getVtablePrefix(void *Object) {
  VtablePrefix **VptrPtr = reinterpret_cast<VtablePrefix**>(Object);
  if (!*VptrPtr)
    return 0;
  VtablePrefix *Prefix = *VptrPtr - 1;
  if (Prefix->Offset > 0 || !Prefix->TypeInfo)
    // This can't possibly be a valid vtable.
    return 0;
  return Prefix;
}

}

bool __ubsan::checkDynamicType(void *Object, void *Type, HashValue Hash) {
  // A crash anywhere within this function probably means the vptr is corrupted.
  // FIXME: Perform these checks more cautiously.

  // Check whether this is something we've evicted from the cache.
  HashValue *Bucket = getTypeCacheHashTableBucket(Hash);
  if (*Bucket == Hash) {
    __ubsan_vptr_type_cache[Hash % VptrTypeCacheSize] = Hash;
    return true;
  }

  VtablePrefix *Vtable = getVtablePrefix(Object);
  if (!Vtable)
    return false;

  // Check that this is actually a type_info object for a class type.
  abi::__class_type_info *Derived =
    dynamic_cast<abi::__class_type_info*>(Vtable->TypeInfo);
  if (!Derived)
    return false;

  abi::__class_type_info *Base = (abi::__class_type_info*)Type;
  if (!isDerivedFromAtOffset(Derived, Base, -Vtable->Offset))
    return false;

  // Success. Cache this result.
  __ubsan_vptr_type_cache[Hash % VptrTypeCacheSize] = Hash;
  *Bucket = Hash;
  return true;
}

__ubsan::DynamicTypeInfo __ubsan::getDynamicTypeInfo(void *Object) {
  VtablePrefix *Vtable = getVtablePrefix(Object);
  if (!Vtable)
    return DynamicTypeInfo(0, 0, 0);
  const abi::__class_type_info *ObjectType = findBaseAtOffset(
    static_cast<const abi::__class_type_info*>(Vtable->TypeInfo),
    -Vtable->Offset);
  return DynamicTypeInfo(Vtable->TypeInfo->__type_name, -Vtable->Offset,
                         ObjectType ? ObjectType->__type_name : "<unknown>");
}
