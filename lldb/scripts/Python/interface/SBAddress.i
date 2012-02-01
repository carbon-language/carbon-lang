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

    %feature("docstring", "
    Create an address by resolving a load address using the supplied target.
    ") SBAddress;
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

    lldb::SBSection
    GetSection ();

    lldb::addr_t
    SBAddress::GetOffset ();

    %feature("docstring", "
    //------------------------------------------------------------------
    /// GetSymbolContext() and the following can lookup symbol information for a given address.
    /// An address might refer to code or data from an existing module, or it
    /// might refer to something on the stack or heap. The following functions
    /// will only return valid values if the address has been resolved to a code
    /// or data address using 'void SBAddress::SetLoadAddress(...)' or 
    /// 'lldb::SBAddress SBTarget::ResolveLoadAddress (...)'. 
    //------------------------------------------------------------------
    ") GetSymbolContext;
    lldb::SBSymbolContext
    GetSymbolContext (uint32_t resolve_scope);

    %feature("docstring", "
    //------------------------------------------------------------------
    /// GetModule() and the following grab individual objects for a given address and
    /// are less efficient if you want more than one symbol related objects. 
    /// Use one of the following when you want multiple debug symbol related 
    /// objects for an address:
    ///    lldb::SBSymbolContext SBAddress::GetSymbolContext (uint32_t resolve_scope);
    ///    lldb::SBSymbolContext SBTarget::ResolveSymbolContextForAddress (const SBAddress &addr, uint32_t resolve_scope);
    /// One or more bits from the SymbolContextItem enumerations can be logically
    /// OR'ed together to more efficiently retrieve multiple symbol objects.
    //------------------------------------------------------------------
    ") GetModule;
    lldb::SBModule
    GetModule ();
    
    lldb::SBCompileUnit
    GetCompileUnit ();

    lldb::SBFunction
    GetFunction ();

    lldb::SBBlock
    GetBlock ();

    lldb::SBSymbol
    GetSymbol ();

    lldb::SBLineEntry
    GetLineEntry ();
    
    %pythoncode %{
        def __get_load_addr_property__ (self):
            '''Get the load address for a lldb.SBAddress using the current target.'''
            return self.GetLoadAddress (target)

        def __set_load_addr_property__ (self, load_addr):
            '''Set the load address for a lldb.SBAddress using the current target.'''
            return self.SetLoadAddress (load_addr, target)

        def __int__(self):
            '''Convert an address to a load address if there is a process and that
            process is alive, or to a file address otherwise.'''
            if process.is_alive:
                return self.GetLoadAddress (target)
            else:
                return self.GetFileAddress ()

        def __oct__(self):
            return '%o' % int(self)

        def __hex__(self):
            return '0x%x' % int(self)

        __swig_getmethods__["module"] = GetModule
        if _newclass: x = property(GetModule, None)

        __swig_getmethods__["compile_unit"] = GetCompileUnit
        if _newclass: x = property(GetCompileUnit, None)

        __swig_getmethods__["line_entry"] = GetLineEntry
        if _newclass: x = property(GetLineEntry, None)

        __swig_getmethods__["function"] = GetFunction
        if _newclass: x = property(GetFunction, None)

        __swig_getmethods__["block"] = GetBlock
        if _newclass: x = property(GetBlock, None)

        __swig_getmethods__["symbol"] = GetSymbol
        if _newclass: x = property(GetSymbol, None)

        __swig_getmethods__["offset"] = GetOffset
        if _newclass: x = property(GetOffset, None)

        __swig_getmethods__["section"] = GetSection
        if _newclass: x = property(GetSection, None)

        __swig_getmethods__["file_addr"] = GetFileAddress
        if _newclass: x = property(GetFileAddress, None)

        __swig_getmethods__["load_addr"] = __get_load_addr_property__
        __swig_setmethods__["load_addr"] = __set_load_addr_property__
        if _newclass: x = property(__get_load_addr_property__, __set_load_addr_property__)

    %}

};

} // namespace lldb
