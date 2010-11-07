//===-- SymbolFileDWARFDebugMap.cpp ----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "SymbolFileDWARFDebugMap.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleList.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/RegularExpression.h"
#include "lldb/Core/StreamFile.h"
#include "lldb/Core/Timer.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Symbol/SymbolVendor.h"
#include "lldb/Symbol/VariableList.h"

#include "SymbolFileDWARF.h"

using namespace lldb;
using namespace lldb_private;

void
SymbolFileDWARFDebugMap::Initialize()
{
    PluginManager::RegisterPlugin (GetPluginNameStatic(),
                                   GetPluginDescriptionStatic(),
                                   CreateInstance);
}

void
SymbolFileDWARFDebugMap::Terminate()
{
    PluginManager::UnregisterPlugin (CreateInstance);
}


const char *
SymbolFileDWARFDebugMap::GetPluginNameStatic()
{
    return "symbol-file.dwarf2-debugmap";
}

const char *
SymbolFileDWARFDebugMap::GetPluginDescriptionStatic()
{
    return "DWARF and DWARF3 debug symbol file reader (debug map).";
}

SymbolFile*
SymbolFileDWARFDebugMap::CreateInstance (ObjectFile* obj_file)
{
    return new SymbolFileDWARFDebugMap (obj_file);
}


SymbolFileDWARFDebugMap::SymbolFileDWARFDebugMap (ObjectFile* ofile) :
    SymbolFile(ofile),
    m_flags(),
    m_compile_unit_infos(),
    m_func_indexes(),
    m_glob_indexes()
{
}


SymbolFileDWARFDebugMap::~SymbolFileDWARFDebugMap()
{
}

void
SymbolFileDWARFDebugMap::InitOSO ()
{
    if (m_flags.test(kHaveInitializedOSOs))
        return;

    m_flags.set(kHaveInitializedOSOs);
    // In order to get the abilities of this plug-in, we look at the list of
    // N_OSO entries (object files) from the symbol table and make sure that
    // these files exist and also contain valid DWARF. If we get any of that
    // then we return the abilities of the first N_OSO's DWARF.

    Symtab* symtab = m_obj_file->GetSymtab();
    if (symtab)
    {
        //StreamFile s(0, 4, eByteOrderHost, stdout);
        std::vector<uint32_t> oso_indexes;
        const uint32_t oso_index_count = symtab->AppendSymbolIndexesWithType(eSymbolTypeObjectFile, oso_indexes);

        symtab->AppendSymbolIndexesWithType (eSymbolTypeCode, Symtab::eDebugYes, Symtab::eVisibilityAny, m_func_indexes);
        symtab->AppendSymbolIndexesWithType (eSymbolTypeData, Symtab::eDebugYes, Symtab::eVisibilityAny, m_glob_indexes);

        symtab->SortSymbolIndexesByValue(m_func_indexes, true);
        symtab->SortSymbolIndexesByValue(m_glob_indexes, true);

        if (oso_index_count > 0)
        {
            m_compile_unit_infos.resize(oso_index_count);
//          s.Printf("%s N_OSO symbols:\n", __PRETTY_FUNCTION__);
//          symtab->Dump(&s, oso_indexes);

            for (uint32_t i=0; i<oso_index_count; ++i)
            {
                m_compile_unit_infos[i].so_symbol = symtab->SymbolAtIndex(oso_indexes[i] - 1);
                if (m_compile_unit_infos[i].so_symbol->GetSiblingIndex() == 0)
                    m_compile_unit_infos[i].so_symbol = symtab->SymbolAtIndex(oso_indexes[i] - 2);
                m_compile_unit_infos[i].oso_symbol = symtab->SymbolAtIndex(oso_indexes[i]);
                uint32_t sibling_idx = m_compile_unit_infos[i].so_symbol->GetSiblingIndex();
                assert (sibling_idx != 0);
                assert (sibling_idx > i + 1);
                m_compile_unit_infos[i].last_symbol = symtab->SymbolAtIndex (sibling_idx - 1);
                m_compile_unit_infos[i].first_symbol_index = symtab->GetIndexForSymbol(m_compile_unit_infos[i].so_symbol);
                m_compile_unit_infos[i].last_symbol_index = symtab->GetIndexForSymbol(m_compile_unit_infos[i].last_symbol);
            }
        }
    }
}

Module *
SymbolFileDWARFDebugMap::GetModuleByOSOIndex (uint32_t oso_idx)
{
    const uint32_t cu_count = GetNumCompileUnits();
    if (oso_idx < cu_count)
        return GetModuleByCompUnitInfo (&m_compile_unit_infos[oso_idx]);
    return NULL;
}

Module *
SymbolFileDWARFDebugMap::GetModuleByCompUnitInfo (CompileUnitInfo *comp_unit_info)
{
    if (comp_unit_info->oso_module_sp.get() == NULL)
    {
        Symbol *oso_symbol = comp_unit_info->oso_symbol;
        if (oso_symbol)
        {
            FileSpec oso_file_spec(oso_symbol->GetMangled().GetName().AsCString(), true);

            ModuleList::GetSharedModule (oso_file_spec,
                                         m_obj_file->GetModule()->GetArchitecture(),
                                         NULL,  // UUID pointer
                                         NULL,  // object name
                                         0,     // object offset
                                         comp_unit_info->oso_module_sp,
                                         NULL,
                                         NULL);
            //comp_unit_info->oso_module_sp.reset(new Module (oso_file_spec, m_obj_file->GetModule()->GetArchitecture()));
        }
    }
    return comp_unit_info->oso_module_sp.get();
}


bool
SymbolFileDWARFDebugMap::GetFileSpecForSO (uint32_t oso_idx, FileSpec &file_spec)
{
    if (oso_idx < m_compile_unit_infos.size())
    {
        if (!m_compile_unit_infos[oso_idx].so_file)
        {

            if (m_compile_unit_infos[oso_idx].so_symbol == NULL)
                return false;

            std::string so_path (m_compile_unit_infos[oso_idx].so_symbol->GetMangled().GetName().AsCString());
            if (m_compile_unit_infos[oso_idx].so_symbol[1].GetType() == eSymbolTypeSourceFile)
                so_path += m_compile_unit_infos[oso_idx].so_symbol[1].GetMangled().GetName().AsCString();
            m_compile_unit_infos[oso_idx].so_file.SetFile(so_path.c_str(), true);
        }
        file_spec = m_compile_unit_infos[oso_idx].so_file;
        return true;
    }
    return false;
}



ObjectFile *
SymbolFileDWARFDebugMap::GetObjectFileByOSOIndex (uint32_t oso_idx)
{
    Module *oso_module = GetModuleByOSOIndex (oso_idx);
    if (oso_module)
        return oso_module->GetObjectFile();
    return NULL;
}

SymbolFileDWARF *
SymbolFileDWARFDebugMap::GetSymbolFile (const SymbolContext& sc)
{
    CompileUnitInfo *comp_unit_info = GetCompUnitInfo (sc);
    if (comp_unit_info)
        return GetSymbolFileByCompUnitInfo (comp_unit_info);
    return NULL;
}

ObjectFile *
SymbolFileDWARFDebugMap::GetObjectFileByCompUnitInfo (CompileUnitInfo *comp_unit_info)
{
    Module *oso_module = GetModuleByCompUnitInfo (comp_unit_info);
    if (oso_module)
        return oso_module->GetObjectFile();
    return NULL;
}

SymbolFileDWARF *
SymbolFileDWARFDebugMap::GetSymbolFileByOSOIndex (uint32_t oso_idx)
{
    if (oso_idx < m_compile_unit_infos.size())
        return GetSymbolFileByCompUnitInfo (&m_compile_unit_infos[oso_idx]);
    return NULL;
}

SymbolFileDWARF *
SymbolFileDWARFDebugMap::GetSymbolFileByCompUnitInfo (CompileUnitInfo *comp_unit_info)
{
    if (comp_unit_info->oso_symbol_vendor == NULL)
    {
        ObjectFile *oso_objfile = GetObjectFileByCompUnitInfo (comp_unit_info);

        if (oso_objfile)
        {
            comp_unit_info->oso_symbol_vendor = oso_objfile->GetModule()->GetSymbolVendor();
//          SymbolFileDWARF *oso_dwarf = new SymbolFileDWARF(oso_objfile);
//          comp_unit_info->oso_dwarf_sp.reset (oso_dwarf);
            if (comp_unit_info->oso_symbol_vendor)
            {
                // Set a a pointer to this class to set our OSO DWARF file know
                // that the DWARF is being used along with a debug map and that
                // it will have the remapped sections that we do below.
                ((SymbolFileDWARF *)comp_unit_info->oso_symbol_vendor->GetSymbolFile())->SetDebugMapSymfile(this);
                comp_unit_info->debug_map_sections_sp.reset(new SectionList);

                Symtab *exe_symtab = m_obj_file->GetSymtab();
                Module *oso_module = oso_objfile->GetModule();
                Symtab *oso_symtab = oso_objfile->GetSymtab();
//#define DEBUG_OSO_DMAP    // Do not check in with this defined...
#if defined(DEBUG_OSO_DMAP)
                StreamFile s(stdout);
                s << "OSO symtab:\n";
                oso_symtab->Dump(&s, NULL);
                s << "OSO sections before:\n";
                oso_objfile->GetSectionList()->Dump(&s, NULL, true);
#endif

                ///const uint32_t fun_resolve_flags = SymbolContext::Module | eSymbolContextCompUnit | eSymbolContextFunction;
                //SectionList *oso_sections = oso_objfile->Sections();
                // Now we need to make sections that map from zero based object
                // file addresses to where things eneded up in the main executable.
                uint32_t oso_start_idx = exe_symtab->GetIndexForSymbol (comp_unit_info->oso_symbol);
                assert (oso_start_idx != UINT32_MAX);
                oso_start_idx += 1;
                const uint32_t oso_end_idx = comp_unit_info->so_symbol->GetSiblingIndex();
                uint32_t sect_id = 0x10000;
                for (uint32_t idx = oso_start_idx; idx < oso_end_idx; ++idx)
                {
                    Symbol *exe_symbol = exe_symtab->SymbolAtIndex(idx);
                    if (exe_symbol)
                    {
                        if (exe_symbol->IsDebug() == false)
                            continue;

                        switch (exe_symbol->GetType())
                        {
                        case eSymbolTypeCode:
                            {
                                // For each N_FUN, or function that we run into in the debug map
                                // we make a new section that we add to the sections found in the
                                // .o file. This new section has the file address set to what the
                                // addresses are in the .o file, and the load address is adjusted
                                // to match where it ended up in the final executable! We do this
                                // before we parse any dwarf info so that when it goes get parsed
                                // all section/offset addresses that get registered will resolve
                                // correctly to the new addresses in the main executable.

                                // First we find the original symbol in the .o file's symbol table
                                Symbol *oso_fun_symbol = oso_symtab->FindFirstSymbolWithNameAndType(exe_symbol->GetMangled().GetName(), eSymbolTypeCode, Symtab::eDebugNo, Symtab::eVisibilityAny);
                                if (oso_fun_symbol)
                                {
                                    // If we found the symbol, then we
                                    Section* exe_fun_section = const_cast<Section *>(exe_symbol->GetAddressRangePtr()->GetBaseAddress().GetSection());
                                    Section* oso_fun_section = const_cast<Section *>(oso_fun_symbol->GetAddressRangePtr()->GetBaseAddress().GetSection());
                                    if (oso_fun_section)
                                    {
                                        // Now we create a section that we will add as a child of the
                                        // section in which the .o symbol (the N_FUN) exists.

                                        // We use the exe_symbol size because the one in the .o file
                                        // will just be a symbol with no size, and the exe_symbol
                                        // size will reflect any size changes (ppc has been known to
                                        // shrink function sizes when it gets rid of jump islands that
                                        // aren't needed anymore).
                                        SectionSP oso_fun_section_sp (new Section (const_cast<Section *>(oso_fun_symbol->GetAddressRangePtr()->GetBaseAddress().GetSection()),
                                                                                   oso_module,                         // Module (the .o file)
                                                                                   sect_id++,                          // Section ID starts at 0x10000 and increments so the section IDs don't overlap with the standard mach IDs
                                                                                   exe_symbol->GetMangled().GetName(), // Name the section the same as the symbol for which is was generated!
                                                                                   eSectionTypeDebug,
                                                                                   oso_fun_symbol->GetAddressRangePtr()->GetBaseAddress().GetOffset(),  // File VM address offset in the current section
                                                                                   exe_symbol->GetByteSize(),          // File size (we need the size from the executable)
                                                                                   0, 0, 0));

                                        oso_fun_section_sp->SetLinkedLocation (exe_fun_section,
                                                                               exe_symbol->GetValue().GetFileAddress() - exe_fun_section->GetFileAddress());
                                        oso_fun_section->GetChildren().AddSection(oso_fun_section_sp);
                                        comp_unit_info->debug_map_sections_sp->AddSection(oso_fun_section_sp);
                                    }
                                }
                            }
                            break;

                        case eSymbolTypeData:
                            {
                                // For each N_GSYM we remap the address for the global by making
                                // a new section that we add to the sections found in the .o file.
                                // This new section has the file address set to what the
                                // addresses are in the .o file, and the load address is adjusted
                                // to match where it ended up in the final executable! We do this
                                // before we parse any dwarf info so that when it goes get parsed
                                // all section/offset addresses that get registered will resolve
                                // correctly to the new addresses in the main executable. We
                                // initially set the section size to be 1 byte, but will need to
                                // fix up these addresses further after all globals have been
                                // parsed to span the gaps, or we can find the global variable
                                // sizes from the DWARF info as we are parsing.

#if 0
                                // First we find the non-stab entry that corresponds to the N_GSYM in the executable
                                Symbol *exe_gsym_symbol = exe_symtab->FindFirstSymbolWithNameAndType(exe_symbol->GetMangled().GetName(), eSymbolTypeData, Symtab::eDebugNo, Symtab::eVisibilityAny);
#else
                                // The mach-o object file parser already matches up the N_GSYM with with the non-stab
                                // entry, so we shouldn't have to do that. If this ever changes, enable the code above
                                // in the "#if 0" block. STSYM's always match the symbol as found below.
                                Symbol *exe_gsym_symbol = exe_symbol;
#endif
                                // Next we find the non-stab entry that corresponds to the N_GSYM in the .o file
                                Symbol *oso_gsym_symbol = oso_symtab->FindFirstSymbolWithNameAndType(exe_symbol->GetMangled().GetName(), eSymbolTypeData, Symtab::eDebugNo, Symtab::eVisibilityAny);
                                if (exe_gsym_symbol && oso_gsym_symbol)
                                {
                                    // If we found the symbol, then we
                                    Section* exe_gsym_section = const_cast<Section *>(exe_gsym_symbol->GetAddressRangePtr()->GetBaseAddress().GetSection());
                                    Section* oso_gsym_section = const_cast<Section *>(oso_gsym_symbol->GetAddressRangePtr()->GetBaseAddress().GetSection());
                                    if (oso_gsym_section)
                                    {
                                        SectionSP oso_gsym_section_sp (new Section (const_cast<Section *>(oso_gsym_symbol->GetAddressRangePtr()->GetBaseAddress().GetSection()),
                                                                                   oso_module,                         // Module (the .o file)
                                                                                   sect_id++,                          // Section ID starts at 0x10000 and increments so the section IDs don't overlap with the standard mach IDs
                                                                                   exe_symbol->GetMangled().GetName(), // Name the section the same as the symbol for which is was generated!
                                                                                   eSectionTypeDebug,
                                                                                   oso_gsym_symbol->GetAddressRangePtr()->GetBaseAddress().GetOffset(),  // File VM address offset in the current section
                                                                                   1,                                   // We don't know the size of the global, just do the main address for now.
                                                                                   0, 0, 0));

                                        oso_gsym_section_sp->SetLinkedLocation (exe_gsym_section,
                                                                               exe_gsym_symbol->GetValue().GetFileAddress() - exe_gsym_section->GetFileAddress());
                                        oso_gsym_section->GetChildren().AddSection(oso_gsym_section_sp);
                                        comp_unit_info->debug_map_sections_sp->AddSection(oso_gsym_section_sp);
                                    }
                                }
                            }
                            break;

//                        case eSymbolTypeStatic:
//                            {
//                                // For each N_STSYM we remap the address for the global by making
//                                // a new section that we add to the sections found in the .o file.
//                                // This new section has the file address set to what the
//                                // addresses are in the .o file, and the load address is adjusted
//                                // to match where it ended up in the final executable! We do this
//                                // before we parse any dwarf info so that when it goes get parsed
//                                // all section/offset addresses that get registered will resolve
//                                // correctly to the new addresses in the main executable. We
//                                // initially set the section size to be 1 byte, but will need to
//                                // fix up these addresses further after all globals have been
//                                // parsed to span the gaps, or we can find the global variable
//                                // sizes from the DWARF info as we are parsing.
//
//
//                                Symbol *exe_stsym_symbol = exe_symbol;
//                                // First we find the non-stab entry that corresponds to the N_STSYM in the .o file
//                                Symbol *oso_stsym_symbol = oso_symtab->FindFirstSymbolWithNameAndType(exe_symbol->GetMangled().GetName(), eSymbolTypeData);
//                                if (exe_stsym_symbol && oso_stsym_symbol)
//                                {
//                                    // If we found the symbol, then we
//                                    Section* exe_stsym_section = const_cast<Section *>(exe_stsym_symbol->GetAddressRangePtr()->GetBaseAddress().GetSection());
//                                    Section* oso_stsym_section = const_cast<Section *>(oso_stsym_symbol->GetAddressRangePtr()->GetBaseAddress().GetSection());
//                                    if (oso_stsym_section)
//                                    {
//                                        // The load address of the symbol will use the section in the
//                                        // executable that contains the debug map that corresponds to
//                                        // the N_FUN symbol. We set the offset to reflect the offset
//                                        // into that section since we are creating a new section.
//                                        AddressRange stsym_load_range(exe_stsym_section, exe_stsym_symbol->GetValue().GetFileAddress() - exe_stsym_section->GetFileAddress(), 1);
//                                        // We need the symbol's section offset address from the .o file, but
//                                        // we need a non-zero size.
//                                        AddressRange stsym_file_range(exe_stsym_symbol->GetAddressRangePtr()->GetBaseAddress().GetSection(), exe_stsym_symbol->GetAddressRangePtr()->GetBaseAddress().GetOffset(), 1);
//
//                                        // Now we create a section that we will add as a child of the
//                                        // section in which the .o symbol (the N_FUN) exists.
//
//// TODO: mimic what I did for N_FUN if that works...
////                                        // We use the 1 byte for the size because we don't know the
////                                        // size of the global symbol without seeing the DWARF.
////                                        SectionSP oso_fun_section_sp (new Section ( NULL, oso_module,                     // Module (the .o file)
////                                                                                        sect_id++,                      // Section ID starts at 0x10000 and increments so the section IDs don't overlap with the standard mach IDs
////                                                                                        exe_symbol->GetMangled().GetName(),// Name the section the same as the symbol for which is was generated!
////                                                                                       // &stsym_load_range,              // Load offset is the offset into the executable section for the N_FUN from the debug map
////                                                                                        &stsym_file_range,              // File section/offset is just the same os the symbol on the .o file
////                                                                                        0, 0, 0));
////
////                                        // Now we add the new section to the .o file's sections as a child
////                                        // of the section in which the N_SECT symbol exists.
////                                        oso_stsym_section->GetChildren().AddSection(oso_fun_section_sp);
////                                        comp_unit_info->debug_map_sections_sp->AddSection(oso_fun_section_sp);
//                                    }
//                                }
//                            }
//                            break;
                        }
                    }
                }
#if defined(DEBUG_OSO_DMAP)
                s << "OSO sections after:\n";
                oso_objfile->GetSectionList()->Dump(&s, NULL, true);
#endif
            }
        }
    }
    if (comp_unit_info->oso_symbol_vendor)
        return (SymbolFileDWARF *)comp_unit_info->oso_symbol_vendor->GetSymbolFile();
    return NULL;
}

uint32_t
SymbolFileDWARFDebugMap::GetAbilities ()
{
    // In order to get the abilities of this plug-in, we look at the list of
    // N_OSO entries (object files) from the symbol table and make sure that
    // these files exist and also contain valid DWARF. If we get any of that
    // then we return the abilities of the first N_OSO's DWARF.

    const uint32_t oso_index_count = GetNumCompileUnits();
    if (oso_index_count > 0)
    {
        const uint32_t dwarf_abilities = SymbolFile::CompileUnits |
                                         SymbolFile::Functions |
                                         SymbolFile::Blocks |
                                         SymbolFile::GlobalVariables |
                                         SymbolFile::LocalVariables |
                                         SymbolFile::VariableTypes |
                                         SymbolFile::LineTables;

        for (uint32_t oso_idx=0; oso_idx<oso_index_count; ++oso_idx)
        {
            SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (oso_idx);
            if (oso_dwarf)
            {
                uint32_t oso_abilities = oso_dwarf->GetAbilities();
                if ((oso_abilities & dwarf_abilities) == dwarf_abilities)
                    return oso_abilities;
            }
        }
    }
    return 0;
}

uint32_t
SymbolFileDWARFDebugMap::GetNumCompileUnits()
{
    InitOSO ();
    return m_compile_unit_infos.size();
}


CompUnitSP
SymbolFileDWARFDebugMap::ParseCompileUnitAtIndex(uint32_t cu_idx)
{
    CompUnitSP comp_unit_sp;
    const uint32_t cu_count = GetNumCompileUnits();

    if (cu_idx < cu_count)
    {
        if (m_compile_unit_infos[cu_idx].oso_compile_unit_sp.get() == NULL)
        {
            SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (cu_idx);
            if (oso_dwarf)
            {
                // There is only one compile unit for N_OSO entry right now, so
                // it will always exist at index zero.
                m_compile_unit_infos[cu_idx].oso_compile_unit_sp = m_compile_unit_infos[cu_idx].oso_symbol_vendor->GetCompileUnitAtIndex (0);
            }

            if (m_compile_unit_infos[cu_idx].oso_compile_unit_sp.get() == NULL)
            {
                // We weren't able to get the DWARF for this N_OSO entry (the
                // .o file may be missing or not at the specified path), make
                // one up as best we can from the debug map. We set the uid
                // of the compile unit to the symbol index with the MSBit set
                // so that it doesn't collide with any uid values from the DWARF
                Symbol *so_symbol = m_compile_unit_infos[cu_idx].so_symbol;
                if (so_symbol)
                {
                    m_compile_unit_infos[cu_idx].oso_compile_unit_sp.reset(new CompileUnit (m_obj_file->GetModule(),
                                                                                            NULL,
                                                                                            so_symbol->GetMangled().GetName().AsCString(),
                                                                                            cu_idx,
                                                                                            eLanguageTypeUnknown));

                    // Let our symbol vendor know about this compile unit
                    m_obj_file->GetModule()->GetSymbolVendor()->SetCompileUnitAtIndex (m_compile_unit_infos[cu_idx].oso_compile_unit_sp, 
                                                                                       cu_idx);
                }
            }
        }
        comp_unit_sp = m_compile_unit_infos[cu_idx].oso_compile_unit_sp;
    }

    return comp_unit_sp;
}

SymbolFileDWARFDebugMap::CompileUnitInfo *
SymbolFileDWARFDebugMap::GetCompUnitInfo (const SymbolContext& sc)
{
    const uint32_t cu_count = GetNumCompileUnits();
    for (uint32_t i=0; i<cu_count; ++i)
    {
        if (sc.comp_unit == m_compile_unit_infos[i].oso_compile_unit_sp.get())
            return &m_compile_unit_infos[i];
    }
    return NULL;
}

size_t
SymbolFileDWARFDebugMap::ParseCompileUnitFunctions (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseCompileUnitFunctions (sc);
    return 0;
}

bool
SymbolFileDWARFDebugMap::ParseCompileUnitLineTable (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseCompileUnitLineTable (sc);
    return false;
}

bool
SymbolFileDWARFDebugMap::ParseCompileUnitSupportFiles (const SymbolContext& sc, FileSpecList &support_files)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseCompileUnitSupportFiles (sc, support_files);
    return false;
}


size_t
SymbolFileDWARFDebugMap::ParseFunctionBlocks (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseFunctionBlocks (sc);
    return 0;
}


size_t
SymbolFileDWARFDebugMap::ParseTypes (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseTypes (sc);
    return 0;
}


size_t
SymbolFileDWARFDebugMap::ParseVariablesForContext (const SymbolContext& sc)
{
    SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
    if (oso_dwarf)
        return oso_dwarf->ParseTypes (sc);
    return 0;
}



Type*
SymbolFileDWARFDebugMap::ResolveTypeUID(lldb::user_id_t type_uid)
{
    return NULL;
}

lldb::clang_type_t
SymbolFileDWARFDebugMap::ResolveClangOpaqueTypeDefinition (lldb::clang_type_t clang_Type)
{
    // We have a struct/union/class/enum that needs to be fully resolved.
    return NULL;
}

uint32_t
SymbolFileDWARFDebugMap::ResolveSymbolContext (const Address& exe_so_addr, uint32_t resolve_scope, SymbolContext& sc)
{
    uint32_t resolved_flags = 0;
    Symtab* symtab = m_obj_file->GetSymtab();
    if (symtab)
    {
        const addr_t exe_file_addr = exe_so_addr.GetFileAddress();
        sc.symbol = symtab->FindSymbolContainingFileAddress (exe_file_addr, &m_func_indexes[0], m_func_indexes.size());

        if (sc.symbol != NULL)
        {
            resolved_flags |= eSymbolContextSymbol;

            uint32_t oso_idx = 0;
            CompileUnitInfo* comp_unit_info = GetCompileUnitInfoForSymbolWithID (sc.symbol->GetID(), &oso_idx);
            if (comp_unit_info)
            {
                SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (oso_idx);
                ObjectFile *oso_objfile = GetObjectFileByOSOIndex (oso_idx);
                if (oso_dwarf && oso_objfile)
                {
                    SectionList *oso_section_list = oso_objfile->GetSectionList();

                    SectionSP oso_symbol_section_sp (oso_section_list->FindSectionContainingLinkedFileAddress (exe_file_addr, UINT32_MAX));

                    if (oso_symbol_section_sp)
                    {
                        const addr_t linked_file_addr = oso_symbol_section_sp->GetLinkedFileAddress();
                        Address oso_so_addr (oso_symbol_section_sp.get(), exe_file_addr - linked_file_addr);
                        if (oso_so_addr.IsSectionOffset())
                            resolved_flags |= oso_dwarf->ResolveSymbolContext (oso_so_addr, resolve_scope, sc);
                    }
                }
            }
        }
    }
    return resolved_flags;
}


uint32_t
SymbolFileDWARFDebugMap::ResolveSymbolContext (const FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list)
{
    uint32_t initial = sc_list.GetSize();
    const uint32_t cu_count = GetNumCompileUnits();

    FileSpec so_file_spec;
    for (uint32_t i=0; i<cu_count; ++i)
    {
        if (GetFileSpecForSO (i, so_file_spec))
        {
            // By passing false to the comparison we will be able to match
            // and files given a filename only. If both file_spec and
            // so_file_spec have directories, we will still do a full match.
            if (FileSpec::Compare (file_spec, so_file_spec, false) == 0)
            {
                SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (i);

                oso_dwarf->ResolveSymbolContext(file_spec, line, check_inlines, resolve_scope, sc_list);
            }
        }
    }
    return sc_list.GetSize() - initial;
}

uint32_t
SymbolFileDWARFDebugMap::PrivateFindGlobalVariables
(
    const ConstString &name,
    const std::vector<uint32_t> &indexes,   // Indexes into the symbol table that match "name"
    uint32_t max_matches,
    VariableList& variables
)
{
    const uint32_t original_size = variables.GetSize();
    const size_t match_count = indexes.size();
    for (size_t i=0; i<match_count; ++i)
    {
        uint32_t oso_idx;
        CompileUnitInfo* comp_unit_info = GetCompileUnitInfoForSymbolWithIndex (indexes[i], &oso_idx);
        if (comp_unit_info)
        {
            SymbolFileDWARF *oso_dwarf = GetSymbolFileByOSOIndex (oso_idx);
            if (oso_dwarf)
            {
                if (oso_dwarf->FindGlobalVariables(name, true, max_matches, variables))
                    if (variables.GetSize() > max_matches)
                        break;
            }
        }
    }
    return variables.GetSize() - original_size;
}

uint32_t
SymbolFileDWARFDebugMap::FindGlobalVariables (const ConstString &name, bool append, uint32_t max_matches, VariableList& variables)
{

    // If we aren't appending the results to this list, then clear the list
    if (!append)
        variables.Clear();

    // Remember how many variables are in the list before we search in case
    // we are appending the results to a variable list.
    const uint32_t original_size = variables.GetSize();

    Symtab* symtab = m_obj_file->GetSymtab();
    if (symtab)
    {
        std::vector<uint32_t> indexes;
        const size_t match_count = m_obj_file->GetSymtab()->FindAllSymbolsWithNameAndType (name, eSymbolTypeData, Symtab::eDebugYes, Symtab::eVisibilityAny, indexes);
        if (match_count)
        {
            PrivateFindGlobalVariables (name, indexes, max_matches, variables);
        }
    }
    // Return the number of variable that were appended to the list
    return variables.GetSize() - original_size;
}


uint32_t
SymbolFileDWARFDebugMap::FindGlobalVariables (const RegularExpression& regex, bool append, uint32_t max_matches, VariableList& variables)
{
    return 0;
}


int
SymbolFileDWARFDebugMap::SymbolContainsSymbolWithIndex (uint32_t *symbol_idx_ptr, const CompileUnitInfo *comp_unit_info)
{
    const uint32_t symbol_idx = *symbol_idx_ptr;

    if (symbol_idx < comp_unit_info->first_symbol_index)
        return -1;

    if (symbol_idx <= comp_unit_info->last_symbol_index)
        return 0;

    return 1;
}


int
SymbolFileDWARFDebugMap::SymbolContainsSymbolWithID (user_id_t *symbol_idx_ptr, const CompileUnitInfo *comp_unit_info)
{
    const user_id_t symbol_id = *symbol_idx_ptr;

    if (symbol_id < comp_unit_info->so_symbol->GetID())
        return -1;

    if (symbol_id <= comp_unit_info->last_symbol->GetID())
        return 0;

    return 1;
}


SymbolFileDWARFDebugMap::CompileUnitInfo*
SymbolFileDWARFDebugMap::GetCompileUnitInfoForSymbolWithIndex (uint32_t symbol_idx, uint32_t *oso_idx_ptr)
{
    const uint32_t oso_index_count = m_compile_unit_infos.size();
    CompileUnitInfo *comp_unit_info = NULL;
    if (oso_index_count)
    {
        comp_unit_info = (CompileUnitInfo*)bsearch(&symbol_idx, &m_compile_unit_infos[0], m_compile_unit_infos.size(), sizeof(CompileUnitInfo), (comparison_function)SymbolContainsSymbolWithIndex);
    }

    if (oso_idx_ptr)
    {
        if (comp_unit_info != NULL)
            *oso_idx_ptr = comp_unit_info - &m_compile_unit_infos[0];
        else
            *oso_idx_ptr = UINT32_MAX;
    }
    return comp_unit_info;
}

SymbolFileDWARFDebugMap::CompileUnitInfo*
SymbolFileDWARFDebugMap::GetCompileUnitInfoForSymbolWithID (user_id_t symbol_id, uint32_t *oso_idx_ptr)
{
    const uint32_t oso_index_count = m_compile_unit_infos.size();
    CompileUnitInfo *comp_unit_info = NULL;
    if (oso_index_count)
    {
        comp_unit_info = (CompileUnitInfo*)bsearch(&symbol_id, &m_compile_unit_infos[0], m_compile_unit_infos.size(), sizeof(CompileUnitInfo), (comparison_function)SymbolContainsSymbolWithID);
    }

    if (oso_idx_ptr)
    {
        if (comp_unit_info != NULL)
            *oso_idx_ptr = comp_unit_info - &m_compile_unit_infos[0];
        else
            *oso_idx_ptr = UINT32_MAX;
    }
    return comp_unit_info;
}


static void
RemoveFunctionsWithModuleNotEqualTo (Module *module, SymbolContextList &sc_list, uint32_t start_idx)
{
    // We found functions in .o files. Not all functions in the .o files
    // will have made it into the final output file. The ones that did
    // make it into the final output file will have a section whose module
    // matches the module from the ObjectFile for this SymbolFile. When
    // the modules don't match, then we have something that was in a
    // .o file, but doesn't map to anything in the final executable.
    uint32_t i=start_idx;
    while (i < sc_list.GetSize())
    {
        SymbolContext sc;
        sc_list.GetContextAtIndex(i, sc);
        if (sc.function)
        {
            const Section *section = sc.function->GetAddressRange().GetBaseAddress().GetSection();
            if (section->GetModule() != module)
            {
                sc_list.RemoveContextAtIndex(i);
                continue;
            }
        }
        ++i;
    }
}

uint32_t
SymbolFileDWARFDebugMap::FindFunctions(const ConstString &name, uint32_t name_type_mask, bool append, SymbolContextList& sc_list)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolFileDWARFDebugMap::FindFunctions (name = %s)",
                        name.GetCString());

    uint32_t initial_size = 0;
    if (append)
        initial_size = sc_list.GetSize();
    else
        sc_list.Clear();

    uint32_t oso_idx = 0;
    SymbolFileDWARF *oso_dwarf;
    while ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx++)) != NULL)
    {
        uint32_t sc_idx = sc_list.GetSize();
        if (oso_dwarf->FindFunctions(name, name_type_mask, true, sc_list))
        {
            RemoveFunctionsWithModuleNotEqualTo (m_obj_file->GetModule(), sc_list, sc_idx);
        }
    }

    return sc_list.GetSize() - initial_size;
}


uint32_t
SymbolFileDWARFDebugMap::FindFunctions (const RegularExpression& regex, bool append, SymbolContextList& sc_list)
{
    Timer scoped_timer (__PRETTY_FUNCTION__,
                        "SymbolFileDWARFDebugMap::FindFunctions (regex = '%s')",
                        regex.GetText());

    uint32_t initial_size = 0;
    if (append)
        initial_size = sc_list.GetSize();
    else
        sc_list.Clear();

    uint32_t oso_idx = 0;
    SymbolFileDWARF *oso_dwarf;
    while ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx++)) != NULL)
    {
        uint32_t sc_idx = sc_list.GetSize();
        
        if (oso_dwarf->FindFunctions(regex, true, sc_list))
        {
            RemoveFunctionsWithModuleNotEqualTo (m_obj_file->GetModule(), sc_list, sc_idx);
        }
    }

    return sc_list.GetSize() - initial_size;
}

TypeSP
SymbolFileDWARFDebugMap::FindDefinitionTypeForDIE (
    DWARFCompileUnit* cu, 
    const DWARFDebugInfoEntry *die, 
    const ConstString &type_name
)
{
    TypeSP type_sp;
    SymbolFileDWARF *oso_dwarf;
    for (uint32_t oso_idx = 0; ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx)) != NULL); ++oso_idx)
    {
        type_sp = oso_dwarf->FindDefinitionTypeForDIE (cu, die, type_name);
        if (type_sp)
            break;
    }
    return type_sp;
}

uint32_t
SymbolFileDWARFDebugMap::FindTypes 
(
    const SymbolContext& sc, 
    const ConstString &name, 
    bool append, 
    uint32_t max_matches, 
    TypeList& types
)
{
    if (!append)
        types.Clear();

    const uint32_t initial_types_size = types.GetSize();
    SymbolFileDWARF *oso_dwarf;

    if (sc.comp_unit)
    {
        oso_dwarf = GetSymbolFile (sc);
        if (oso_dwarf)
            return oso_dwarf->FindTypes (sc, name, append, max_matches, types);
    }
    else
    {
        uint32_t oso_idx = 0;
        while ((oso_dwarf = GetSymbolFileByOSOIndex (oso_idx++)) != NULL)
            oso_dwarf->FindTypes (sc, name, append, max_matches, types);
    }

    return types.GetSize() - initial_types_size;
}

//
//uint32_t
//SymbolFileDWARFDebugMap::FindTypes (const SymbolContext& sc, const RegularExpression& regex, bool append, uint32_t max_matches, Type::Encoding encoding, lldb::user_id_t udt_uid, TypeList& types)
//{
//  SymbolFileDWARF *oso_dwarf = GetSymbolFile (sc);
//  if (oso_dwarf)
//      return oso_dwarf->FindTypes (sc, regex, append, max_matches, encoding, udt_uid, types);
//  return 0;
//}

//------------------------------------------------------------------
// PluginInterface protocol
//------------------------------------------------------------------
const char *
SymbolFileDWARFDebugMap::GetPluginName()
{
    return "SymbolFileDWARFDebugMap";
}

const char *
SymbolFileDWARFDebugMap::GetShortPluginName()
{
    return GetPluginNameStatic();
}

uint32_t
SymbolFileDWARFDebugMap::GetPluginVersion()
{
    return 1;
}

void
SymbolFileDWARFDebugMap::GetPluginCommandHelp (const char *command, Stream *strm)
{
}

Error
SymbolFileDWARFDebugMap::ExecutePluginCommand (Args &command, Stream *strm)
{
    Error error;
    error.SetErrorString("No plug-in command are currently supported.");
    return error;
}

Log *
SymbolFileDWARFDebugMap::EnablePluginLogging (Stream *strm, Args &command)
{
    return NULL;
}


void
SymbolFileDWARFDebugMap::SetCompileUnit (SymbolFileDWARF *oso_dwarf, const CompUnitSP &cu_sp)
{
    const uint32_t cu_count = GetNumCompileUnits();
    for (uint32_t i=0; i<cu_count; ++i)
    {
        if (m_compile_unit_infos[i].oso_symbol_vendor &&
            m_compile_unit_infos[i].oso_symbol_vendor->GetSymbolFile() == oso_dwarf)
        {
            if (m_compile_unit_infos[i].oso_compile_unit_sp)
            {
                assert (m_compile_unit_infos[i].oso_compile_unit_sp.get() == cu_sp.get());
            }
            else
            {
                m_compile_unit_infos[i].oso_compile_unit_sp = cu_sp;
            }
        }
    }
}

