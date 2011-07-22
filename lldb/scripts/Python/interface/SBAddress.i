//===-- SWIG Interface for SBAddress ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature("docstring",
"A section + offset based address class.

The SBAddress class allows addresses to be relative to a section
that can move during runtime due to images (executables, shared
libraries, bundles, frameworks) being loaded at different
addresses than the addresses found in the object file that
represents them on disk. There are currently two types of addresses
for a section:
    o file addresses
    o load addresses

File addresses represents the virtual addresses that are in the 'on
disk' object files. These virtual addresses are converted to be
relative to unique sections scoped to the object file so that
when/if the addresses slide when the images are loaded/unloaded
in memory, we can easily track these changes without having to
update every object (compile unit ranges, line tables, function
address ranges, lexical block and inlined subroutine address
ranges, global and static variables) each time an image is loaded or
unloaded.

Load addresses represents the virtual addresses where each section
ends up getting loaded at runtime. Before executing a program, it
is common for all of the load addresses to be unresolved. When a
DynamicLoader plug-in receives notification that shared libraries
have been loaded/unloaded, the load addresses of the main executable
and any images (shared libraries) will be  resolved/unresolved. When
this happens, breakpoints that are in one of these sections can be
set/cleared.

See docstring of SBFunction for example usage of SBAddress."
) SBAddress;
class SBAddress
{
public:

    SBAddress ();

    SBAddress (const lldb::SBAddress &rhs);

    // Create an address by resolving a load address using the supplied target
    SBAddress (lldb::addr_t load_addr, lldb::SBTarget &target);

    ~SBAddress ();

    bool
    IsValid () const;

    void
    Clear ();

    addr_t
    GetFileAddress () const;

    addr_t
    GetLoadAddress (const lldb::SBTarget &target) const;

	void
	SetLoadAddress (lldb::addr_t load_addr, 
                	lldb::SBTarget &target);

    bool
    OffsetAddress (addr_t offset);

    bool
    GetDescription (lldb::SBStream &description);

    SectionType
    GetSectionType ();

    lldb::SBModule
    GetModule ();
};

} // namespace lldb
