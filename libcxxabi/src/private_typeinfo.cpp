//===----------------------- private_typeinfo.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "private_typeinfo.h"

#include <iostream>

namespace std
{

type_info::~type_info()
{
}

}  // std

namespace __cxxabiv1
{

// __fundamental_type_info

// This miraculously (compiler magic) emits the type_info's for:
//   1. all of the fundamental types
//   2. pointers to all of the fundamental types
//   3. pointers to all of the const fundamental types
__fundamental_type_info::~__fundamental_type_info()
{
}

// __array_type_info

__array_type_info::~__array_type_info()
{
}

// __function_type_info

__function_type_info::~__function_type_info()
{
}

// __enum_type_info

__enum_type_info::~__enum_type_info()
{
}

// __class_type_info

__class_type_info::~__class_type_info()
{
}

void
__class_type_info::display(const void* obj) const
{
    std::cout << "\n__class_type_info::this = " << obj << "  " << name() << '\n';
}

// __si_class_type_info

__si_class_type_info::~__si_class_type_info()
{
}

void
__si_class_type_info::display(const void* obj) const
{
    std::cout << "\n__si_class_type_info::this = " << obj << "  " << name() << '\n';
    __base_type->display(obj);
}

// __vmi_class_type_info

__vmi_class_type_info::~__vmi_class_type_info()
{
}

void
__vmi_class_type_info::display(const void* obj) const
{
    std::cout << "\n__vmi_class_type_info::this = " << obj << "  " << name() << '\n';
    if (__flags & __non_diamond_repeat_mask)
        std::cout << "__non_diamond_repeat_mask\n";
    if (__flags & __diamond_shaped_mask)
        std::cout << "__diamond_shaped_mask\n";
    std::cout << "__base_count = " << __base_count << '\n';
    for (const __base_class_type_info* p = __base_info; p < __base_info + __base_count; ++p)
        p->display(obj);
}

void
__base_class_type_info::display(const void* obj) const
{
    if (__offset_flags & __virtual_mask)
        std::cout << "__virtual_mask\n";
    if (__offset_flags & __public_mask)
        std::cout << "__public_mask\n";
    ptrdiff_t offset_to_base = __offset_flags >> __offset_shift;
    if (__offset_flags & __virtual_mask)
    {
        char* vtable = *(char**)obj;
        offset_to_base = (ptrdiff_t)vtable[offset_to_base];
    }
    __base_type->display((char*)obj + offset_to_base);
}

// __pbase_type_info

__pbase_type_info::~__pbase_type_info()
{
}

// __pointer_type_info

__pointer_type_info::~__pointer_type_info()
{
}

// __pointer_to_member_type_info

__pointer_to_member_type_info::~__pointer_to_member_type_info()
{
}

// __dynamic_cast

// static_ptr: source address to be adjusted; nonnull, and since the
//   source object is polymorphic, *(void**)static_ptr is a virtual table pointer.
// static_type: static type of the source object.
// dst_type: destination type (the "T" in "dynamic_cast<T>(v)").
// src2dst_offset: a static hint about the location of the
//                 source subobject with respect to the complete object;
//                 special negative values are:
//                     -1: no hint
//                     -2: static_type is not a public base of dst_type
//                     -3: static_type is a multiple public base type but never a
//                         virtual base type
//                 otherwise, the static_type type is a unique public nonvirtual
//                 base type of dst_type at offset src2dst_offset from the
//                 origin of dst_type.
// Returns either one of:
//    1.  dynamic_ptr adjusted from static_ptr given a public path from
//        (static_ptr, static_type) to dst_type without ambiguity, or
//    2.  dynamic_ptr adjusted from static_ptr given a public path from
//        (static_ptr, static_type) to (dynamic_ptr, dynamic_type) and also
//        a public path from (dynamic_ptr, dynamic_type) to dst_type without
//        ambiguity.
// Things I think I know:
//     This is a DAG rooted at dynamic_type and going up.
//     Don't care about anything above static_type.
//     There can be only one dynamic_type and it is at the root.
//     A dst_type can never appear above another dst_type.
extern "C"
void*
__dynamic_cast(const void* static_ptr,
			   const __class_type_info* static_type,
			   const __class_type_info* dst_type,
			   std::ptrdiff_t src2dst_offset)
{
std::cout << "static_ptr = " << static_ptr << '\n';
std::cout << "static_type = " << static_type << '\n';
std::cout << "dst_type = " << dst_type << '\n';
std::cout << "src2dst_offset = " << src2dst_offset << '\n';
    void** vtable = *(void***)static_ptr;
    ptrdiff_t offset_to_derived = (ptrdiff_t)vtable[-2];
    const void* dynamic_ptr = (const char*)static_ptr + offset_to_derived;
    const __class_type_info* dynamic_type = (const __class_type_info*)vtable[-1];
std::cout << "dynamic_ptr = " << dynamic_ptr << '\n';
std::cout << "dynamic_type = " << dynamic_type << '\n';
dynamic_type->display(dynamic_ptr);
    return 0;
}

}  // __cxxabiv1
