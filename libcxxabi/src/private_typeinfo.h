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

class __class_type_info
    : public std::type_info
{
public:
    virtual ~__class_type_info();
};

class __si_class_type_info
    : public __class_type_info
{
public:
    const __class_type_info* __base_type;

    virtual ~__si_class_type_info();
};

struct __base_class_type_info
{
public:
    const __class_type_info* __base_type;
    long __offset_flags;

    enum __offset_flags_masks
    {
        __virtual_mask = 0x1,
        __public_mask  = 0x2,
        __offset_shift = 8
    };
};

class __vmi_class_type_info
    : public __class_type_info
{
public:
    unsigned int __flags;
    unsigned int __base_count;
    __base_class_type_info __base_info[1];

    enum __flags_masks
    {
        __non_diamond_repeat_mask = 0x1,
        __diamond_shaped_mask     = 0x2
    };

    virtual ~__vmi_class_type_info();
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
