//===-- SymbolFile.h --------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_SymbolFile_h_
#define liblldb_SymbolFile_h_

#include "lldb/lldb-private.h"
#include "lldb/Core/PluginInterface.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Symbol/ClangNamespaceDecl.h"
#include "lldb/Symbol/Type.h"

namespace lldb_private {

class SymbolFile :
    public PluginInterface
{
public:
    enum Abilities
    {
        Labels                              = (1 << 0),
        AddressAcceleratorTable             = (1 << 1),
        FunctionAcceleratorTable            = (1 << 2),
        TypeAcceleratorTable                = (1 << 3),
        MacroInformation                    = (1 << 4),
        CallFrameInformation                = (1 << 5),
        CompileUnits                        = (1 << 6),
        LineTables                          = (1 << 7),
        LineColumns                         = (1 << 8),
        Functions                           = (1 << 9),
        Blocks                              = (1 << 10),
        GlobalVariables                     = (1 << 11),
        LocalVariables                      = (1 << 12),
        VariableTypes                       = (1 << 13)
    };

    static SymbolFile *
    FindPlugin (ObjectFile* obj_file);

    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    SymbolFile(ObjectFile* obj_file) :
        m_obj_file(obj_file)
    {
    }

    virtual
    ~SymbolFile()
    {
    }

    virtual uint32_t        GetAbilities () = 0;

    //------------------------------------------------------------------
    // Compile Unit function calls
    //------------------------------------------------------------------
    // Approach 1 - iterator
    virtual uint32_t        GetNumCompileUnits() = 0;
    virtual lldb::CompUnitSP  ParseCompileUnitAtIndex(uint32_t index) = 0;

    virtual size_t          ParseCompileUnitFunctions (const SymbolContext& sc) = 0;
    virtual bool            ParseCompileUnitLineTable (const SymbolContext& sc) = 0;
    virtual bool            ParseCompileUnitSupportFiles (const SymbolContext& sc, FileSpecList& support_files) = 0;
    virtual size_t          ParseFunctionBlocks (const SymbolContext& sc) = 0;
    virtual size_t          ParseTypes (const SymbolContext& sc) = 0;
    virtual size_t          ParseVariablesForContext (const SymbolContext& sc) = 0;
    virtual Type*           ResolveTypeUID (lldb::user_id_t type_uid) = 0;
    virtual lldb::clang_type_t ResolveClangOpaqueTypeDefinition (lldb::clang_type_t clang_type) = 0;
    virtual clang::DeclContext* GetClangDeclContextForTypeUID (lldb::user_id_t type_uid) { return NULL; }
    virtual uint32_t        ResolveSymbolContext (const Address& so_addr, uint32_t resolve_scope, SymbolContext& sc) = 0;
    virtual uint32_t        ResolveSymbolContext (const FileSpec& file_spec, uint32_t line, bool check_inlines, uint32_t resolve_scope, SymbolContextList& sc_list) = 0;
    virtual uint32_t        FindGlobalVariables (const ConstString &name, bool append, uint32_t max_matches, VariableList& variables) = 0;
    virtual uint32_t        FindGlobalVariables (const RegularExpression& regex, bool append, uint32_t max_matches, VariableList& variables) = 0;
    virtual uint32_t        FindFunctions (const ConstString &name, uint32_t name_type_mask, bool append, SymbolContextList& sc_list) = 0;
    virtual uint32_t        FindFunctions (const RegularExpression& regex, bool append, SymbolContextList& sc_list) = 0;
    virtual uint32_t        FindTypes (const SymbolContext& sc, const ConstString &name, bool append, uint32_t max_matches, TypeList& types) = 0;
//  virtual uint32_t        FindTypes (const SymbolContext& sc, const RegularExpression& regex, bool append, uint32_t max_matches, TypeList& types) = 0;
    virtual TypeList *      GetTypeList ();
    virtual ClangNamespaceDecl
                            FindNamespace (const SymbolContext& sc, 
                                           const ConstString &name) = 0;

    ObjectFile*             GetObjectFile() { return m_obj_file; }
    const ObjectFile*       GetObjectFile() const { return m_obj_file; }
protected:
    ObjectFile*         m_obj_file; // The object file that symbols can be extracted from.

private:
    DISALLOW_COPY_AND_ASSIGN (SymbolFile);
};


} // namespace lldb_private

#endif  // liblldb_SymbolFile_h_
