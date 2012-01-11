//===----------------------- private_typeinfo.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "private_typeinfo.h"

// temporary headers
#include <iostream>
#include <cassert>

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

// __dynamic_cast notes:
// Up or above refers to base classes and base objects.
// Down or below refers to derived classes/objects.

// search1 is a search algorithm used by __dynamic_cast.
// If a static_type is found
//     If dynamic_ptr == static_ptr
//         Record the path to get here.
//         if the path to get here is public
//             stop search
//     Otherwise continue search only below this node
// Else
//     Continue search above and below this node.

// __class_type_info::search1
// There are no nodes to search above this node
// Returns:  1 if search should be continued, otherwise 0
int
__class_type_info::search1(__dynamic_cast_info* info, const void* dynamic_ptr,
                           int path_below) const
{
    if (this == info->static_type)
    {
        if (dynamic_ptr == info->static_ptr)
        {
            if (path_below == public_path)
            {
                info->path_dynamic_ptr_to_static_ptr = public_path;
                return 0;
            }
            info->path_dynamic_ptr_to_static_ptr = not_public_path;
        }
    }
    return 1;
}

// search2 is a search algorithm used by __dynamic_cast.
// if this is a dst_type
//     if this has already been classified then
//        If the path to get here is public, overwrite existing path_dynamic_ptr_to_dst_ptr.
//     else we haven't been to this dst_type before.
//        Record the path to get here in path_dynamic_ptr_to_dst_ptr.
//        Is (static_ptr, static_type) above this dst_type?
//           Yes:
//             Record it as dst_ptr_leading_to_static_ptr and increment the
//                number of such recordings.
//             If this is not the first of such recordings, then stop searching.
//             Otherwise continue searching both above and below this node.
//           No:
//             record it as dst_ptr_not_leading_to_static_ptr and increment
//                 the number of such recordings.
// else if this is a static_type
//     if this is *our* static_type
//        if we found it above a dst_type, record the path from the dst_type
//        else record the path from the dynamic_type being careful not to overwrite a
//           previous public path in this latter case.
//        Record that we found our static_type.
//     Continue searching only below this node
// else 
//     Continue searching above and below this node.
int
__class_type_info::search2(__dynamic_cast_info* info, const void* dynamic_ptr,
                           int path_below) const
{
    if (this == info->dst_type)
    {
        if (dynamic_ptr == info->dst_ptr_not_leading_to_static_ptr)
        {
            if (path_below == public_path)
                info->path_dynamic_ptr_to_dst_ptr = public_path;
        }
        else
        {
            info->path_dynamic_ptr_to_dst_ptr = path_below;
            info->dst_ptr_not_leading_to_static_ptr = dynamic_ptr;
            info->number_to_dst_ptr += 1;
        }
    }
    else if (this == info->static_type && dynamic_ptr == info->static_ptr)
    {
        if (info->above_dst_ptr)
            info->path_dst_ptr_to_static_ptr = path_below;
        else if (info->path_dynamic_ptr_to_static_ptr != public_path)
            info->path_dynamic_ptr_to_static_ptr = path_below;
        info->found_static_ptr = true;
    }
    return 1;
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

int
__si_class_type_info::search1(__dynamic_cast_info* info, const void* dynamic_ptr,
                              int path_below) const
{
    if (this == info->static_type)
    {
        if (dynamic_ptr == info->static_ptr)
        {
            if (path_below == public_path)
            {
                info->path_dynamic_ptr_to_static_ptr = public_path;
                return 0;
            }
        }
        return 1;
    }
    return __base_type->search1(info, dynamic_ptr, path_below);
}

int
__si_class_type_info::search2(__dynamic_cast_info* info, const void* dynamic_ptr,
                              int path_below) const
{
    if (this == info->dst_type)
    {
        if (dynamic_ptr == info->dst_ptr_leading_to_static_ptr ||
            dynamic_ptr == info->dst_ptr_not_leading_to_static_ptr)
        {
            if (path_below == public_path)
                info->path_dynamic_ptr_to_dst_ptr = public_path;
        }
        else
        {
            info->path_dynamic_ptr_to_dst_ptr = path_below;
            info->above_dst_ptr = true;
            (void)__base_type->search2(info, dynamic_ptr, public_path);
            info->above_dst_ptr = false;
            if (info->found_static_ptr)
            {
                info->found_static_ptr = false;
                info->dst_ptr_leading_to_static_ptr = dynamic_ptr;
                info->number_to_static_ptr += 1;
                return info->number_to_static_ptr == 1;
            }
            info->dst_ptr_not_leading_to_static_ptr = dynamic_ptr;
            info->number_to_dst_ptr += 1;
        }
        return 1;
    }
    else if (this == info->static_type)
    {
        if (dynamic_ptr == info->static_ptr)
        {
            if (info->above_dst_ptr)
                info->path_dst_ptr_to_static_ptr = path_below;
            else if (info->path_dynamic_ptr_to_static_ptr != public_path)
                info->path_dynamic_ptr_to_static_ptr = path_below;
            info->found_static_ptr = true;
        }
        return 1;
    }
    return __base_type->search2(info, dynamic_ptr, path_below);
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

int
__vmi_class_type_info::search1(__dynamic_cast_info* info, const void* dynamic_ptr,
                               int path_below) const
{
    typedef const __base_class_type_info* Iter;
    if (this == info->static_type)
    {
        if (dynamic_ptr == info->static_ptr)
        {
            if (path_below == public_path)
            {
                info->path_dynamic_ptr_to_static_ptr = public_path;
                return 0;
            }
        }
        return 1;
    }
    for (Iter p = __base_info, e = __base_info + __base_count; p < e; ++p)
    {
        int r = p->search1(info, dynamic_ptr, path_below);
        if (r == 0)
            return 0;
    }
    return 1;
}

int
__base_class_type_info::search1(__dynamic_cast_info* info, const void* dynamic_ptr,
                                int path_below) const
{
    ptrdiff_t offset_to_base = __offset_flags >> __offset_shift;
    if (__offset_flags & __virtual_mask)
    {
        char* vtable = *(char**)dynamic_ptr;
        offset_to_base = (ptrdiff_t)vtable[offset_to_base];
    }
    return __base_type->search1(info, (char*)dynamic_ptr + offset_to_base,
                                (__offset_flags & __public_mask) ? path_below :
                                                                   not_public_path);
}

int
__vmi_class_type_info::search2(__dynamic_cast_info* info, const void* dynamic_ptr,
                               int path_below) const
{
    typedef const __base_class_type_info* Iter;
    if (this == info->dst_type)
    {
        if (dynamic_ptr == info->dst_ptr_leading_to_static_ptr ||
            dynamic_ptr == info->dst_ptr_not_leading_to_static_ptr)
        {
            if (path_below == public_path)
                info->path_dynamic_ptr_to_dst_ptr = public_path;
        }
        else
        {
            info->path_dynamic_ptr_to_dst_ptr = path_below;
            for (Iter p = __base_info, e = __base_info + __base_count; p < e; ++p)
            {
                info->above_dst_ptr = true;
                int r = p->search2(info, dynamic_ptr, public_path);
                info->above_dst_ptr = false;
                if (r == 0)
                    return 0;
                if (info->found_static_ptr)
                {
                    info->found_static_ptr = false;
                    info->dst_ptr_leading_to_static_ptr = dynamic_ptr;
                    info->number_to_static_ptr += 1;
                    if (info->number_to_static_ptr != 1)
                        return 0;
                }
                else
                {
                    info->dst_ptr_not_leading_to_static_ptr = dynamic_ptr;
                    info->number_to_dst_ptr += 1;
                }
            }
        }
        return 1;
    }
    else if (this == info->static_type)
    {
        if (dynamic_ptr == info->static_ptr)
        {
            if (info->above_dst_ptr)
                info->path_dst_ptr_to_static_ptr = path_below;
            else if (info->path_dynamic_ptr_to_static_ptr != public_path)
                info->path_dynamic_ptr_to_static_ptr = path_below;
            info->found_static_ptr = true;
        }
        return 1;
    }
    for (Iter p = __base_info, e = __base_info + __base_count; p < e; ++p)
    {
        int r = p->search2(info, dynamic_ptr, path_below);
        if (r == 0)
            return 0;
    }
    return 1;
}

int
__base_class_type_info::search2(__dynamic_cast_info* info, const void* dynamic_ptr,
                                int path_below) const
{
    ptrdiff_t offset_to_base = __offset_flags >> __offset_shift;
    if (__offset_flags & __virtual_mask)
    {
        char* vtable = *(char**)dynamic_ptr;
        offset_to_base = (ptrdiff_t)vtable[offset_to_base];
    }
    return __base_type->search2(info, (char*)dynamic_ptr + offset_to_base,
                                (__offset_flags & __public_mask) ? path_below :
                                                                   not_public_path);
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
//        ambiguity, or
//    3.  nullptr
// Knowns:
//     (dynamic_ptr, dynamic_type) can be derived from static_ptr.
//     dynamic_ptr is a pointer to the complete run time object.
//     dynamic_type is the type of the complete run time object.
//     The type hierarchy is a DAG rooted at (dynamic_ptr, dynamic_type) and
//     traveling up.  Each node represents a distinct sub-object and is
//     uniquely identified by a (const void*, const __class_type_info*) pair.
//     __dynamic_cast is not called if dst_type == static_type, or if dst_type
//     appears as a node above (static_ptr, static_type), or if static_ptr is
//     nullptr.  In these cases the compiler already knows the answer.
//     So if dst_type appears in the DAG, it appears at the root
//     (dst_type == dynamic_type) or between the root and any static_types.
//     No type can appear above a node of the same type in the DAG.  Thus
//     there is only one node with type dynamic_type.
//     A node can appear above more than one node, as well as below more than
//     one node.
//     If dst_type == dynamic_type then there is only one dst_type in the
//        DAG and cases 1 and 2 collapse to the same degenerate case.  And
//        this degenerate case is easier/faster to search:
//        Returns dynamic_ptr if there exists a public path from
//           (dynamic_ptr, dynamic_type) to (static_ptr, static_type),
//           else returns nullptr.
//        This check is purely an optimization and does not impact correctness.
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

    const void* dst_ptr = 0;
    __dynamic_cast_info info = {dst_type, static_ptr, static_type, src2dst_offset, 0};
    if (dynamic_type == dst_type)
    {
        (void)dynamic_type->search1(&info, dynamic_ptr, public_path);
        if (info.path_dynamic_ptr_to_static_ptr == public_path)
            dst_ptr = dynamic_ptr;
    }
    else
    {
        (void)dynamic_type->search2(&info, dynamic_ptr, public_path);
        switch (info.number_to_static_ptr)
        {
        case 0:
            if (info.number_to_dst_ptr == 1 &&
                    info.path_dynamic_ptr_to_static_ptr == public_path &&
                    info.path_dynamic_ptr_to_dst_ptr == public_path)
                dst_ptr = info.dst_ptr_not_leading_to_static_ptr;
            break;
        case 1:
            if (info.path_dst_ptr_to_static_ptr == public_path)
                dst_ptr = info.dst_ptr_leading_to_static_ptr;
            break;
        }
    }
    return const_cast<void*>(dst_ptr);
}

}  // __cxxabiv1
