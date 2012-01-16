//===----------------------- private_typeinfo.cpp -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "private_typeinfo.h"

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

// __si_class_type_info

__si_class_type_info::~__si_class_type_info()
{
}

// __vmi_class_type_info

__vmi_class_type_info::~__vmi_class_type_info()
{
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
//
// (dynamic_ptr, dynamic_type) are the run time type of the complete object and
// a pointer to it.  These can be found from static_ptr for polymorphic types.
// static_type is guaranteed to be a polymorphic type.
//
// There are two classes of dst_types:
//    1.  Those that lead to (static_ptr, static_type).
//    2.  Those that do not lead to (static_ptr, static_type).
// If there is exactly one dst_type of type 1, and
//    If there is a public path from that dst_type to (static_ptr, static_type), or
//    If there are 0 dst_types of type 2, and there is a public path from
//        (dynamic_ptr, dynamic_type) to (static_ptr, static_type) and a public
//        path from (dynamic_ptr, dynamic_type) to the one dst_type, then return
//        a pointer to that dst_type.
// Else if there are 0 dst_types of type 1 and exactly 1 dst_type of type 2, and
//    if there is a public path (dynamic_ptr, dynamic_type) to
//    (static_ptr, static_type) and a public path from (dynamic_ptr, dynamic_type)
//    to the one dst_type, then return a pointer to that one dst_type.
// Else return nullptr.
//
// If dynamic_type == dst_type, then the above algorithm collapses to the
// following cheaper algorithm:
//
// If there is a public path from (dynamic_ptr, dynamic_type) to
//    (static_ptr, static_type), then return dynamic_ptr.
// Else return nullptr.
extern "C"
void*
__dynamic_cast(const void* static_ptr,
			   const __class_type_info* static_type,
			   const __class_type_info* dst_type,
			   std::ptrdiff_t src2dst_offset)
{
    // TODO:  Take advantage of src2dst_offset
    void** vtable = *(void***)static_ptr;
    ptrdiff_t offset_to_derived = (ptrdiff_t)vtable[-2];
    const void* dynamic_ptr = (const char*)static_ptr + offset_to_derived;
    const __class_type_info* dynamic_type = (const __class_type_info*)vtable[-1];
    const void* dst_ptr = 0;
    __dynamic_cast_info info = {dst_type, static_ptr, static_type, src2dst_offset, 0};
    if (dynamic_type == dst_type)
    {
        info.number_of_dst_type = 1;
        dynamic_type->search_above_dst(&info, dynamic_ptr, dynamic_ptr, public_path);
        if (info.path_dst_ptr_to_static_ptr == public_path)
            dst_ptr = dynamic_ptr;
    }
    else
    {
        dynamic_type->search_below_dst(&info, dynamic_ptr, public_path);
        switch (info.number_to_static_ptr)
        {
        case 0:
            if (info.number_to_dst_ptr == 1 &&
                    info.path_dynamic_ptr_to_static_ptr == public_path &&
                    info.path_dynamic_ptr_to_dst_ptr == public_path)
                dst_ptr = info.dst_ptr_not_leading_to_static_ptr;
            break;
        case 1:
            if (info.path_dst_ptr_to_static_ptr == public_path ||
                   (
                       info.number_to_dst_ptr == 0 &&
                       info.path_dynamic_ptr_to_static_ptr == public_path &&
                       info.path_dynamic_ptr_to_dst_ptr == public_path
                   )
               )
                dst_ptr = info.dst_ptr_leading_to_static_ptr;
            break;
        }
    }
    return const_cast<void*>(dst_ptr);
}

// Call this function when you hit a static_type which is a base (above) a dst_type.
// Let caller know you hit a static_type.  But only start recording details if
// this is (static_ptr, static_type) -- the node we are casting from.
// If this is (static_ptr, static_type)
//   Record the path (public or not) from the dst_type to here.  There may be
//   multiple paths from the same dst_type to here, record the "most public" one.
//   Record the dst_ptr as pointing to (static_ptr, static_type).
//   If more than one (dst_ptr, dst_type) points to (static_ptr, static_type),
//   then mark this dyanmic_cast as ambiguous and stop the search.
void
__class_type_info::process_static_type_above_dst(__dynamic_cast_info* info,
                                                 const void* dst_ptr,
                                                 const void* current_ptr,
                                                 int path_below) const
{
    // Record that we found a static_type
    info->found_any_static_type = true;
    if (current_ptr == info->static_ptr)
    {
        // Record that we found (static_ptr, static_type)
        info->found_our_static_ptr = true;
        if (info->dst_ptr_leading_to_static_ptr == 0)
        {
            // First time here
            info->dst_ptr_leading_to_static_ptr = dst_ptr;
            info->path_dst_ptr_to_static_ptr = path_below;
            info->number_to_static_ptr = 1;
            // If there is only one dst_type in the entire tree and the path from
            //    there to here is public then we are done!
            if (info->number_of_dst_type == 1 && info->path_dst_ptr_to_static_ptr == public_path)
                info->search_done = true;
        }
        else if (info->dst_ptr_leading_to_static_ptr == dst_ptr)
        {
            // We've been here before.  Update path to "most public"
            if (info->path_dst_ptr_to_static_ptr == not_public_path)
                info->path_dst_ptr_to_static_ptr = path_below;
            // If there is only one dst_type in the entire tree and the path from
            //    there to here is public then we are done!
            if (info->number_of_dst_type == 1 && info->path_dst_ptr_to_static_ptr == public_path)
                info->search_done = true;
        }
        else
        {
            // We've detected an ambiguous cast from (static_ptr, static_type)
            //   to a dst_type
            info->number_to_static_ptr += 1;
            info->search_done = true;
        }
    }
}

// Call this function when you hit a static_type which is not a base (above) a dst_type.
// Let caller know you hit a static_type (this may not be necessary).
// But only start recording details if this is (static_ptr, static_type) -- the node we are casting from.
// If this is (static_ptr, static_type)
//   Record the path (public or not) from (dynamic_ptr, dynamic_type) to here.  There may be
//   multiple paths from (dynamic_ptr, dynamic_type) to here, record the "most public" one.
void
__class_type_info::process_static_type_below_dst(__dynamic_cast_info* info,
                                                 const void* current_ptr,
                                                 int path_below) const
{
    // Record that we found a static_type
    info->found_any_static_type = true;  // TODO: Consider removing, no one is currently listening
    if (current_ptr == info->static_ptr)
    {
        // Record that we found (static_ptr, static_type)
        info->found_our_static_ptr = true;  // TODO: Consider removing, no one is currently listening
        // Record the most public path from (dynamic_ptr, dynamic_type) to
        //                                  (static_ptr, static_type)
        if (info->path_dynamic_ptr_to_static_ptr != public_path)
            info->path_dynamic_ptr_to_static_ptr = path_below;
    }
}

// Call this function when searching below a dst_type node.  This function searches
// for a path to (static_ptr, static_type) and for paths to one or more dst_type nodes.
// If it finds a static_type node, there is no need to further search base classes
// above.
// If it finds a dst_type node it should search base classes using search_above_dst
// to find out if this dst_type points to (static_ptr, static_type) or not.
// Either way, the dst_type is recorded as one of two "classes":  one that does
// or does not point to (static_ptr, static_type).
// If this is neither a static_type nor a dst_type node, continue searching
// base classes above.
// All the hoopla surrounding the search code is doing nothing but looking for
// excuses to stop the search prematurely (break out of the for-loop).
void
__vmi_class_type_info::search_below_dst(__dynamic_cast_info* info,
                                        const void* current_ptr,
                                        int path_below) const
{
    typedef const __base_class_type_info* Iter;
    if (this == info->static_type)
        process_static_type_below_dst(info, current_ptr, path_below);
    else if (this == info->dst_type)
    {
        // We've been here before if we've recorded current_ptr in one of these
        //   two places:
        if (current_ptr == info->dst_ptr_leading_to_static_ptr ||
            current_ptr == info->dst_ptr_not_leading_to_static_ptr)
        {
            // We've seen this node before, and therefore have already searched
            // its base classes above.
            //  Update path to here that is "most public".
            if (path_below == public_path)
                info->path_dynamic_ptr_to_dst_ptr = public_path;
        }
        else  // We have haven't been here before
        {
            // Record the access path that got us here
            //   If there is more than one dst_type this path doesn't matter.
            info->path_dynamic_ptr_to_dst_ptr = path_below;
            // Only search above here if dst_type derives from static_type, or
            //    if it is unknown if dst_type derives from static_type.
            if (info->is_dst_type_derived_from_static_type != no)
            {
                // Set up flags to record results from all base classes
                bool is_dst_type_derived_from_static_type = false;
                bool does_dst_type_point_to_our_static_type = false;
                // We've found a dst_type with a potentially public path to here.
                // We have to assume the path is public because it may become
                //   public later (if we get back to here with a public path).
                // We can stop looking above if:
                //    1.  We've found a public path to (static_ptr, static_type).
                //    2.  We've found an ambiguous cast from (static_ptr, static_type) to a dst_type.
                //        This is detected at the (static_ptr, static_type).
                //    3.  We can prove that there is no public path to (static_ptr, static_type)
                //        above here.
                const Iter e = __base_info + __base_count;
                for (Iter p = __base_info; p < e; ++p)
                {
                    // Zero out found flags
                    info->found_our_static_ptr = false;
                    info->found_any_static_type = false;
                    p->search_above_dst(info, current_ptr, current_ptr, public_path);
                   if (info->search_done)
                        break;
                    if (info->found_any_static_type)
                    {
                        is_dst_type_derived_from_static_type = true;
                        if (info->found_our_static_ptr)
                        {
                            does_dst_type_point_to_our_static_type = true;
                            // If we found what we're looking for, stop looking above.
                            if (info->path_dst_ptr_to_static_ptr == public_path)
                                break;
                            // We found a private path to (static_ptr, static_type)
                            //   If there is no diamond then there is only one path
                            //   to (static_ptr, static_type) and we just found it.
                            if (!(__flags & __diamond_shaped_mask))
                                break;
                        }
                        else
                        {
                            // If we found a static_type that isn't the one we're looking
                            //    for, and if there are no repeated types above here,
                            //    then stop looking.
                            if (!(__flags & __non_diamond_repeat_mask))
                                break;
                        }
                    }
                }
                if (!does_dst_type_point_to_our_static_type)
                {
                    // We found a dst_type that doesn't point to (static_ptr, static_type)
                    // So record the address of this dst_ptr and increment the
                    // count of the number of such dst_types found in the tree.
                    info->dst_ptr_not_leading_to_static_ptr = current_ptr;
                    info->number_to_dst_ptr += 1;
                    // If there exists another dst with a private path to
                    //    (static_ptr, static_type), then the cast from 
                    //     (dynamic_ptr, dynamic_type) to dst_type is now ambiguous,
                    //      so stop search.
                    if (info->number_to_static_ptr == 1 &&
                            info->path_dst_ptr_to_static_ptr == not_public_path)
                        info->search_done = true;
                }
                // If we found no static_type,s then dst_type doesn't derive
                //   from static_type, else it does.  Record this result so that
                //   next time we hit a dst_type we will know not to search above
                //   it if it doesn't derive from static_type.
                if (is_dst_type_derived_from_static_type)
                    info->is_dst_type_derived_from_static_type = yes;
                else
                    info->is_dst_type_derived_from_static_type = no;
            }
        }
    }
    else
    {
        // This is not a static_type and not a dst_type.
        const Iter e = __base_info + __base_count;
        if ((__flags & __diamond_shaped_mask) || info->number_to_static_ptr == 1)
        {
            // If there are multiple paths to a base above from here, or if
            //    a dst_type pointing to (static_ptr, static_type) has been found,
            //    then there is no way to break out of this loop early unless
            //    something below detects the search is done.
            for (Iter p = __base_info; p < e; ++p)
            {
                p->search_below_dst(info, current_ptr, path_below);
                if (info->search_done)
                    break;
            }
        }
        else if (__flags & __non_diamond_repeat_mask)
        {
            // There are not multiple paths to any base class from here and a
            //   dst_type pointing to (static_ptr, static_type) has not yet been
            //   found.
            for (Iter p = __base_info; p < e; ++p)
            {
                p->search_below_dst(info, current_ptr, path_below);
                if (info->search_done)
                    break;
                // If we just found a dst_type with a public path to (static_ptr, static_type),
                //    then the only reason to continue the search is to make sure
                //    no other dst_type points to (static_ptr, static_type).
                //    If !diamond, then we don't need to search here.
                if (info->number_to_static_ptr == 1 &&
                          info->path_dst_ptr_to_static_ptr == public_path)
                    break;
            }
        }
        else
        {
            // There are no repeated types above this node.
            // There are no nodes with multiple parents above this node.
            // no dst_type has been found to (static_ptr, static_type)
            for (Iter p = __base_info; p < e; ++p)
            {
                p->search_below_dst(info, current_ptr, path_below);
                if (info->search_done)
                    break;
                // If we just found a dst_type with a public path to (static_ptr, static_type),
                //    then the only reason to continue the search is to make sure sure
                //    no other dst_type points to (static_ptr, static_type).
                //    If !diamond, then we don't need to search here.
                // if we just found a dst_type with a private path to (static_ptr, static_type),
                //    then we're only looking for a public path to (static_ptr, static_type)
                //    and to check for other dst_types.
                //    If !diamond & !repeat, then there is not a pointer to (static_ptr, static_type)
                //    and not a dst_type under here.
                if (info->number_to_static_ptr == 1)
                    break;
            }
        }
    }
}

// This is the same algorithm as __vmi_class_type_info::search_below_dst but
//   simplified to the case that there is only a single base class.
void
__si_class_type_info::search_below_dst(__dynamic_cast_info* info,
                                       const void* current_ptr,
                                       int path_below) const
{
    if (this == info->static_type)
        process_static_type_below_dst(info, current_ptr, path_below);
    else if (this == info->dst_type)
    {
        // We've been here before if we've recorded current_ptr in one of these
        //   two places:
        if (current_ptr == info->dst_ptr_leading_to_static_ptr ||
            current_ptr == info->dst_ptr_not_leading_to_static_ptr)
        {
            // We've seen this node before, and therefore have already searched
            // its base classes above.
            //  Update path to here that is "most public".
            if (path_below == public_path)
                info->path_dynamic_ptr_to_dst_ptr = public_path;
        }
        else  // We have haven't been here before
        {
            // Record the access path that got us here
            //   If there is more than one dst_type this path doesn't matter.
            info->path_dynamic_ptr_to_dst_ptr = path_below;
            // Only search above here if dst_type derives from static_type, or
            //    if it is unknown if dst_type derives from static_type.
            if (info->is_dst_type_derived_from_static_type != no)
            {
                // Set up flags to record results from all base classes
                bool is_dst_type_derived_from_static_type = false;
                bool does_dst_type_point_to_our_static_type = false;
                // Zero out found flags
                info->found_our_static_ptr = false;
                info->found_any_static_type = false;
                __base_type->search_above_dst(info, current_ptr, current_ptr, public_path);
                if (info->found_any_static_type)
                {
                    is_dst_type_derived_from_static_type = true;
                    if (info->found_our_static_ptr)
                        does_dst_type_point_to_our_static_type = true;
                }
                if (!does_dst_type_point_to_our_static_type)
                {
                    // We found a dst_type that doesn't point to (static_ptr, static_type)
                    // So record the address of this dst_ptr and increment the
                    // count of the number of such dst_types found in the tree.
                    info->dst_ptr_not_leading_to_static_ptr = current_ptr;
                    info->number_to_dst_ptr += 1;
                    // If there exists another dst with a private path to
                    //    (static_ptr, static_type), then the cast from 
                    //     (dynamic_ptr, dynamic_type) to dst_type is now ambiguous.
                    if (info->number_to_static_ptr == 1 &&
                            info->path_dst_ptr_to_static_ptr == not_public_path)
                        info->search_done = true;
                }
                // If we found no static_type,s then dst_type doesn't derive
                //   from static_type, else it does.  Record this result so that
                //   next time we hit a dst_type we will know not to search above
                //   it if it doesn't derive from static_type.
                if (is_dst_type_derived_from_static_type)
                    info->is_dst_type_derived_from_static_type = yes;
                else
                    info->is_dst_type_derived_from_static_type = no;
            }
        }
    }
    else
    {
        // This is not a static_type and not a dst_type
        __base_type->search_below_dst(info, current_ptr, path_below);
    }
}

// This is the same algorithm as __vmi_class_type_info::search_below_dst but
//   simplified to the case that there is no base class.
void
__class_type_info::search_below_dst(__dynamic_cast_info* info,
                                    const void* current_ptr,
                                    int path_below) const
{
    typedef const __base_class_type_info* Iter;
    if (this == info->static_type)
        process_static_type_below_dst(info, current_ptr, path_below);
    else if (this == info->dst_type)
    {
        // We've been here before if we've recorded current_ptr in one of these
        //   two places:
        if (current_ptr == info->dst_ptr_leading_to_static_ptr ||
            current_ptr == info->dst_ptr_not_leading_to_static_ptr)
        {
            // We've seen this node before, and therefore have already searched
            // its base classes above.
            //  Update path to here that is "most public".
            if (path_below == public_path)
                info->path_dynamic_ptr_to_dst_ptr = public_path;
        }
        else  // We have haven't been here before
        {
            // Record the access path that got us here
            //   If there is more than one dst_type this path doesn't matter.
            info->path_dynamic_ptr_to_dst_ptr = path_below;
            // We found a dst_type that doesn't point to (static_ptr, static_type)
            // So record the address of this dst_ptr and increment the
            // count of the number of such dst_types found in the tree.
            info->dst_ptr_not_leading_to_static_ptr = current_ptr;
            info->number_to_dst_ptr += 1;
            // If there exists another dst with a private path to
            //    (static_ptr, static_type), then the cast from 
            //     (dynamic_ptr, dynamic_type) to dst_type is now ambiguous.
            if (info->number_to_static_ptr == 1 &&
                    info->path_dst_ptr_to_static_ptr == not_public_path)
                info->search_done = true;
            // We found that dst_type does not derive from static_type
            info->is_dst_type_derived_from_static_type = no;
        }
    }
}

// Call this function when searching above a dst_type node.  This function searches
// for a public path to (static_ptr, static_type).
// This function is guaranteed not to find a node of type dst_type.
// Theoretically this is a very simple function which just stops if it finds a
// static_type node, else keeps searching with:
//
//             const Iter e = __base_info + __base_count;
//             for (Iter p = __base_info; p < e; ++p)
//                 p->search_above_dst(info, dst_ptr, current_ptr, path_below);
//
// All the hoopla surrounding the search code is doing nothing but looking for
// excuses to stop the search prematurely (break out of the for-loop).
void
__vmi_class_type_info::search_above_dst(__dynamic_cast_info* info,
                                        const void* dst_ptr,
                                        const void* current_ptr,
                                        int path_below) const
{
    if (this == info->static_type)
        process_static_type_above_dst(info, dst_ptr, current_ptr, path_below);
    else
    {
        typedef const __base_class_type_info* Iter;
        // This is not a static_type and not a dst_type
        // Save flags so they can be restored when returning to nodes below.
        bool found_our_static_ptr = info->found_our_static_ptr;
        bool found_any_static_type = info->found_any_static_type;
        // We've found a dst_type below with a path to here.  If the path
        //    to here is not public, there may be another path to here that
        //    is public.  So we have to assume that the path to here is public.
        //  We can stop looking above if:
        //    1.  We've found a public path to (static_ptr, static_type).
        //    2.  We've found an ambiguous cast from (static_ptr, static_type) to a dst_type.
        //        This is detected at the (static_ptr, static_type).
        //    3.  We can prove that there is no public path to (static_ptr, static_type)
        //        above here.
        const Iter e = __base_info + __base_count;
        for (Iter p = __base_info; p < e; ++p)
        {
            // Zero out found flags
            info->found_our_static_ptr = false;
            info->found_any_static_type = false;
            p->search_above_dst(info, dst_ptr, current_ptr, path_below);
            if (info->search_done)
                break;
            if (info->found_our_static_ptr)
            {
                // If we found what we're looking for, stop looking above.
                if (info->path_dst_ptr_to_static_ptr == public_path)
                    break;
                // We found a private path to (static_ptr, static_type)
                //   If there is no diamond then there is only one path
                //   to (static_ptr, static_type) from here and we just found it.
                if (!(__flags & __diamond_shaped_mask))
                    break;
            }
            else if (info->found_any_static_type)
            {
                // If we found a static_type that isn't the one we're looking
                //    for, and if there are no repeated types above here,
                //    then stop looking.
                if (!(__flags & __non_diamond_repeat_mask))
                    break;
            }
        }
        // Restore flags
        info->found_our_static_ptr = found_our_static_ptr;
        info->found_any_static_type = found_any_static_type;
    }
}

// This is the same algorithm as __vmi_class_type_info::search_above_dst but
//   simplified to the case that there is only a single base class.
void
__si_class_type_info::search_above_dst(__dynamic_cast_info* info,
                                       const void* dst_ptr,
                                       const void* current_ptr,
                                       int path_below) const
{
    if (this == info->static_type)
        process_static_type_above_dst(info, dst_ptr, current_ptr, path_below);
    else
        __base_type->search_above_dst(info, dst_ptr, current_ptr, path_below);
}

// This is the same algorithm as __vmi_class_type_info::search_above_dst but
//   simplified to the case that there is no base class.
void
__class_type_info::search_above_dst(__dynamic_cast_info* info,
                                    const void* dst_ptr,
                                    const void* current_ptr,
                                    int path_below) const
{
    if (this == info->static_type)
        process_static_type_above_dst(info, dst_ptr, current_ptr, path_below);
}

// The search functions for __base_class_type_info are simply convenience
//   functions for adjusting the current_ptr and path_below as the search is
//   passed up to the base class node.

void
__base_class_type_info::search_above_dst(__dynamic_cast_info* info,
                                         const void* dst_ptr,
                                         const void* current_ptr,
                                         int path_below) const
{
    ptrdiff_t offset_to_base = __offset_flags >> __offset_shift;
    if (__offset_flags & __virtual_mask)
    {
        char* vtable = *(char**)current_ptr;
        offset_to_base = *(ptrdiff_t*)(vtable + offset_to_base);
    }
    __base_type->search_above_dst(info, dst_ptr,
                                  (char*)current_ptr + offset_to_base,
                                  (__offset_flags & __public_mask) ?
                                      path_below :
                                      not_public_path);
}

void
__base_class_type_info::search_below_dst(__dynamic_cast_info* info,
                                         const void* current_ptr,
                                         int path_below) const
{
    ptrdiff_t offset_to_base = __offset_flags >> __offset_shift;
    if (__offset_flags & __virtual_mask)
    {
        char* vtable = *(char**)current_ptr;
        offset_to_base = *(ptrdiff_t*)(vtable + offset_to_base);
    }
    __base_type->search_below_dst(info,
                                  (char*)current_ptr + offset_to_base,
                                  (__offset_flags & __public_mask) ?
                                      path_below :
                                      not_public_path);
}

}  // __cxxabiv1
