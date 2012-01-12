//===------------------------ private_typeinfo.h --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __PRIVATE_TYPEINFO_H_
#define __PRIVATE_TYPEINFO_H_

#include <typeinfo>
#include <cstddef>

namespace __cxxabiv1
{

class __fundamental_type_info
    : public std::type_info
{
public:
    virtual ~__fundamental_type_info();
};

class __array_type_info
    : public std::type_info
{
public:
    virtual ~__array_type_info();
};

class __function_type_info
    : public std::type_info
{
public:
    virtual ~__function_type_info();
};

class __enum_type_info
    : public std::type_info
{
public:
    virtual ~__enum_type_info();
};

enum
{
    unknown = 0,
    public_path,
    not_public_path
};

class __class_type_info;

struct __dynamic_cast_info
{
    // const data supplied to the search

    const __class_type_info* const dst_type;
    const void* const static_ptr;
    const __class_type_info* const static_type;
    const std::ptrdiff_t src2dst_offset;

    // non-const data learned during the search

    // pointer to a dst_type which has (static_ptr, static_type) above it
    const void* dst_ptr_leading_to_static_ptr;
    // pointer to a dst_type which does not have (static_ptr, static_type) above it
    const void* dst_ptr_not_leading_to_static_ptr;
    // access of path from dst_ptr_leading_to_static_ptr to (static_ptr, static_type)
    int path_dst_ptr_to_static_ptr;
    // access of path from (dynamic_ptr, dynamic_type) to (static_ptr, static_type)
    //    when there is no dst_type along the path
    int path_dynamic_ptr_to_static_ptr;
    // access of path from (dynamic_ptr, dynamic_type) to dst_type
    //    (not used if there is a (static_ptr, static_type) above a dst_type).
    int path_dynamic_ptr_to_dst_ptr;
    // Number of dst_types below (static_ptr, static_type)
    int number_to_static_ptr;
    // Number of dst_types not below (static_ptr, static_type)
    int number_to_dst_ptr;
    // true when the search is above a dst_type, else false
    bool above_dst_ptr;
    // communicates to a dst_type node that (static_ptr, static_type) was found
    //    above it.
    bool found_static_ptr;
};

// Has no base class
class __class_type_info
    : public std::type_info
{
public:
    virtual ~__class_type_info();

    virtual int search1(__dynamic_cast_info*, const void*, int) const;
    virtual int search2(__dynamic_cast_info*, const void*, int) const;
#ifdef DEBUG
    virtual void display(const void* obj) const;
#endif
};

// Has one non-virtual public base class at offset zero
class __si_class_type_info
    : public __class_type_info
{
public:
    const __class_type_info* __base_type;

    virtual ~__si_class_type_info();

    virtual int search1(__dynamic_cast_info*, const void*, int) const;
    virtual int search2(__dynamic_cast_info*, const void*, int) const;
#ifdef DEBUG
    virtual void display(const void* obj) const;
#endif
};

struct __base_class_type_info
{
public:
    const __class_type_info* __base_type;
    long __offset_flags;

    enum __offset_flags_masks
    {
        __virtual_mask = 0x1,
        __public_mask  = 0x2, // base is public
        __offset_shift = 8
    };

    int search1(__dynamic_cast_info*, const void*, int) const;
    int search2(__dynamic_cast_info*, const void*, int) const;
#ifdef DEBUG
    void display(const void* obj) const;
#endif
};

// Has one or more base classes
class __vmi_class_type_info
    : public __class_type_info
{
public:
    unsigned int __flags;
    unsigned int __base_count;
    __base_class_type_info __base_info[1];

    enum __flags_masks
    {
        __non_diamond_repeat_mask = 0x1,  // has two or more distinct base class
                                          //    objects of the same type
        __diamond_shaped_mask     = 0x2   // has base class object with two or
                                          //    more derived objects
    };

    virtual ~__vmi_class_type_info();

    virtual int search1(__dynamic_cast_info*, const void*, int) const;
    virtual int search2(__dynamic_cast_info*, const void*, int) const;
#ifdef DEBUG
    virtual void display(const void* obj) const;
#endif
};

class __pbase_type_info
    : public std::type_info
{
public:
    unsigned int __flags;
    const std::type_info* __pointee;

    enum __masks
    {
        __const_mask            = 0x1,
        __volatile_mask         = 0x2,
        __restrict_mask         = 0x4,
        __incomplete_mask       = 0x8,
        __incomplete_class_mask = 0x10
    };

    virtual ~__pbase_type_info();
};

class __pointer_type_info
    : public __pbase_type_info
{
public:
    virtual ~__pointer_type_info();
};

class __pointer_to_member_type_info
    : public __pbase_type_info
{
public:
    const __class_type_info* __context;

    virtual ~__pointer_to_member_type_info();
};

}  // __cxxabiv1

#endif  // __PRIVATE_TYPEINFO_H_
