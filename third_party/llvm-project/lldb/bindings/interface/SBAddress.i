//===-- SWIG Interface for SBAddress ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
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

* file addresses
* load addresses

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

    SBAddress (lldb::SBSection section,
               lldb::addr_t offset);

    %feature("docstring", "
    Create an address by resolving a load address using the supplied target.") SBAddress;
    SBAddress (lldb::addr_t load_addr, lldb::SBTarget &target);

    ~SBAddress ();

    bool
    IsValid () const;

    explicit operator bool() const;

#ifdef SWIGPYTHON
    // operator== is a free function, which swig does not handle, so we inject
    // our own equality operator here
    %pythoncode%{
    def __eq__(self, other):
      return not self.__ne__(other)
    %}
#endif

    bool operator!=(const SBAddress &rhs) const;

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

    void
    SetAddress (lldb::SBSection section,
                lldb::addr_t offset);

    %feature("docstring", "
    GetSymbolContext() and the following can lookup symbol information for a given address.
    An address might refer to code or data from an existing module, or it
    might refer to something on the stack or heap. The following functions
    will only return valid values if the address has been resolved to a code
    or data address using :py:class:`SBAddress.SetLoadAddress' or
    :py:class:`SBTarget.ResolveLoadAddress`.") GetSymbolContext;
    lldb::SBSymbolContext
    GetSymbolContext (uint32_t resolve_scope);

    %feature("docstring", "
    GetModule() and the following grab individual objects for a given address and
    are less efficient if you want more than one symbol related objects.
    Use :py:class:`SBAddress.GetSymbolContext` or
    :py:class:`SBTarget.ResolveSymbolContextForAddress` when you want multiple
    debug symbol related objects for an address.
    One or more bits from the SymbolContextItem enumerations can be logically
    OR'ed together to more efficiently retrieve multiple symbol objects.") GetModule;
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

    STRING_EXTENSION(SBAddress)

#ifdef SWIGPYTHON
    %pythoncode %{
        __runtime_error_str = 'This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'

        def __get_load_addr_property__ (self):
            '''Get the load address for a lldb.SBAddress using the current target. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'''
            if not target:
                raise RuntimeError(self.__runtime_error_str)
            return self.GetLoadAddress (target)

        def __set_load_addr_property__ (self, load_addr):
            '''Set the load address for a lldb.SBAddress using the current target. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'''
            if not target:
                raise RuntimeError(self.__runtime_error_str)
            return self.SetLoadAddress (load_addr, target)

        def __int__(self):
            '''Convert an address to a load address if there is a process and that process is alive, or to a file address otherwise. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'''
            if not process or not target:
                raise RuntimeError(self.__runtime_error_str)
            if process.is_alive:
                return self.GetLoadAddress (target)
            return self.GetFileAddress ()

        def __oct__(self):
            '''Convert the address to an octal string. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'''
            return '%o' % int(self)

        def __hex__(self):
            '''Convert the address to an hex string. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.'''
            return '0x%x' % int(self)

        module = property(GetModule, None, doc='''A read only property that returns an lldb object that represents the module (lldb.SBModule) that this address resides within.''')
        compile_unit = property(GetCompileUnit, None, doc='''A read only property that returns an lldb object that represents the compile unit (lldb.SBCompileUnit) that this address resides within.''')
        line_entry = property(GetLineEntry, None, doc='''A read only property that returns an lldb object that represents the line entry (lldb.SBLineEntry) that this address resides within.''')
        function = property(GetFunction, None, doc='''A read only property that returns an lldb object that represents the function (lldb.SBFunction) that this address resides within.''')
        block = property(GetBlock, None, doc='''A read only property that returns an lldb object that represents the block (lldb.SBBlock) that this address resides within.''')
        symbol = property(GetSymbol, None, doc='''A read only property that returns an lldb object that represents the symbol (lldb.SBSymbol) that this address resides within.''')
        offset = property(GetOffset, None, doc='''A read only property that returns the section offset in bytes as an integer.''')
        section = property(GetSection, None, doc='''A read only property that returns an lldb object that represents the section (lldb.SBSection) that this address resides within.''')
        file_addr = property(GetFileAddress, None, doc='''A read only property that returns file address for the section as an integer. This is the address that represents the address as it is found in the object file that defines it.''')
        load_addr = property(__get_load_addr_property__, __set_load_addr_property__, doc='''A read/write property that gets/sets the SBAddress using load address. This resolves the SBAddress using the SBTarget from lldb.target so this property can ONLY be used in the interactive script interpreter (i.e. under the lldb script command). For things like Python based commands and breakpoint callbacks use GetLoadAddress instead.''')
    %}
#endif

};

} // namespace lldb
