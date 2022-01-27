//===-- SWIG Interface for SBTypeEnumMember ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

namespace lldb {

%feature(
    "docstring",
    "Represents a member of an enum in lldb."
) SBTypeEnumMember;

class SBTypeEnumMember
{
public:
    SBTypeEnumMember ();

    SBTypeEnumMember (const SBTypeEnumMember& rhs);

    ~SBTypeEnumMember();

    bool
    IsValid() const;

    explicit operator bool() const;

    int64_t
    GetValueAsSigned();

    uint64_t
    GetValueAsUnsigned();

    const char *
    GetName ();

    lldb::SBType
    GetType ();

    bool
    GetDescription (lldb::SBStream &description,
                    lldb::DescriptionLevel description_level);

    STRING_EXTENSION_LEVEL(SBTypeEnumMember, lldb::eDescriptionLevelBrief)
#ifdef SWIGPYTHON
    %pythoncode %{
        name = property(GetName, None, doc='''A read only property that returns the name for this enum member as a string.''')
        type = property(GetType, None, doc='''A read only property that returns an lldb object that represents the type (lldb.SBType) for this enum member.''')
        signed = property(GetValueAsSigned, None, doc='''A read only property that returns the value of this enum member as a signed integer.''')
        unsigned = property(GetValueAsUnsigned, None, doc='''A read only property that returns the value of this enum member as a unsigned integer.''')
    %}
#endif

protected:
    friend class SBType;
    friend class SBTypeEnumMemberList;

    void
    reset (lldb_private::TypeEnumMemberImpl *);

    lldb_private::TypeEnumMemberImpl &
    ref ();

    const lldb_private::TypeEnumMemberImpl &
    ref () const;

    lldb::TypeEnumMemberImplSP m_opaque_sp;

    SBTypeEnumMember (const lldb::TypeEnumMemberImplSP &);
};

%feature("docstring",
"Represents a list of SBTypeEnumMembers.

SBTypeEnumMemberList supports SBTypeEnumMember iteration.
It also supports [] access either by index, or by enum
element name by doing: ::

  myType = target.FindFirstType('MyEnumWithElementA')
  members = myType.GetEnumMembers()
  first_elem = members[0]
  elem_A = members['A']

") SBTypeEnumMemberList;

class SBTypeEnumMemberList
{
public:
    SBTypeEnumMemberList();

    SBTypeEnumMemberList(const SBTypeEnumMemberList& rhs);

    ~SBTypeEnumMemberList();

    bool
    IsValid();

    explicit operator bool() const;

    void
    Append (SBTypeEnumMember entry);

    SBTypeEnumMember
    GetTypeEnumMemberAtIndex (uint32_t index);

    uint32_t
    GetSize();

#ifdef SWIGPYTHON
    %pythoncode %{
        def __iter__(self):
            '''Iterate over all members in a lldb.SBTypeEnumMemberList object.'''
            return lldb_iter(self, 'GetSize', 'GetTypeEnumMemberAtIndex')

        def __len__(self):
            '''Return the number of members in a lldb.SBTypeEnumMemberList object.'''
            return self.GetSize()

        def __getitem__(self, key):
          num_elements = self.GetSize()
          if type(key) is int:
              if key < num_elements:
                  return self.GetTypeEnumMemberAtIndex(key)
          elif type(key) is str:
              for idx in range(num_elements):
                  item = self.GetTypeEnumMemberAtIndex(idx)
                  if item.name == key:
                      return item
          return None
    %}
#endif

private:
    std::unique_ptr<lldb_private::TypeEnumMemberListImpl> m_opaque_ap;
};

} // namespace lldb
