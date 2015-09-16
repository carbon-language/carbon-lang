//===-- SymbolFileDWARFDwo.h ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_
#define SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "SymbolFileDWARF.h"

class SymbolFileDWARFDwo : public SymbolFileDWARF
{
public:
    SymbolFileDWARFDwo(lldb::ObjectFileSP objfile, DWARFCompileUnit* dwarf_cu);

    virtual
    ~SymbolFileDWARFDwo() = default;
    
    const lldb_private::DWARFDataExtractor&
    GetCachedSectionData(uint32_t got_flag,
                         lldb::SectionType sect_type,
                         lldb_private::DWARFDataExtractor &data) override;
    
    lldb::CompUnitSP
    ParseCompileUnit(DWARFCompileUnit* dwarf_cu, uint32_t cu_idx) override;

    DWARFCompileUnit*
    GetCompileUnit();

    DWARFCompileUnit*
    GetDWARFCompileUnit(lldb_private::CompileUnit *comp_unit) override;

    lldb_private::DWARFExpression::LocationListFormat
    GetLocationListFormat() const override;

protected:
    DIEToTypePtr&
    GetDIEToType() override;

    DIEToVariableSP&
    GetDIEToVariable() override;
    
    DIEToClangType&
    GetForwardDeclDieToClangType() override;

    ClangTypeToDIE&
    GetForwardDeclClangTypeToDie() override;

    lldb::TypeSP
    FindDefinitionTypeForDWARFDeclContext (const DWARFDeclContext &die_decl_ctx) override;

    SymbolFileDWARF*
    GetBaseSymbolFile();

    lldb::ObjectFileSP m_obj_file_sp;
    DWARFCompileUnit* m_base_dwarf_cu;
};

#endif  // SymbolFileDWARFDwo_SymbolFileDWARFDwo_h_
