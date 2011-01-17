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

    //------------------------------------------------------------------
    /// Get a mask of what this symbol file supports for the object file
    /// that it was constructed with.
    ///
    /// Each symbol file gets to respond with a mask of abilities that
    /// it supports for each object file. This happens when we are
    /// trying to figure out which symbol file plug-in will get used
    /// for a given object file. The plug-in that resoonds with the 
    /// best mix of "SymbolFile::Abilities" bits set, will get chosen to
    /// be the symbol file parser. This allows each plug-in to check for
    /// sections that contain data a symbol file plug-in would need. For
    /// example the DWARF plug-in requires DWARF sections in a file that
    /// contain debug information. If the DWARF plug-in doesn't find
    /// these sections, it won't respond with many ability bits set, and
    /// we will probably fall back to the symbol table SymbolFile plug-in
    /// which uses any information in the symbol table. Also, plug-ins 
    /// might check for some specific symbols in a symbol table in the
    /// case where the symbol table contains debug information (STABS
    /// and COFF). Not a lot of work should happen in these functions
    /// as the plug-in might not get selected due to another plug-in
    /// having more abilities. Any initialization work should be saved
    /// for "void SymbolFile::InitializeObject()" which will get called
    /// on the SymbolFile object with the best set of abilities.
    ///
    /// @return
    ///     A uint32_t mask containing bits from the SymbolFile::Abilities
    ///     enumeration. Any bits that are set represent an ability that
    ///     this symbol plug-in can parse from the object file.
    ///------------------------------------------------------------------
    virtual uint32_t        GetAbilities () = 0;
    
    //------------------------------------------------------------------
    /// Initialize the SymbolFile object.
    ///
    /// The SymbolFile object with the best set of abilities (detected
    /// in "uint32_t SymbolFile::GetAbilities()) will have this function
    /// called if it is chosen to parse an object file. More complete
    /// initialization can happen in this function which will get called
    /// prior to any other functions in the SymbolFile protocol.
    //------------------------------------------------------------------    
    virtual void            InitializeObject() {}

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
    virtual ClangASTContext &
                            GetClangASTContext ();
    virtual ClangNamespaceDecl
                            FindNamespace (const SymbolContext& sc, 
                                           const ConstString &name) = 0;

    ObjectFile*             GetObjectFile() { return m_obj_file; }
    const ObjectFile*       GetObjectFile() const { return m_obj_file; }
protected:
    ObjectFile*             m_obj_file; // The object file that symbols can be extracted from.

private:
    DISALLOW_COPY_AND_ASSIGN (SymbolFile);
};


} // namespace lldb_private

#endif  // liblldb_SymbolFile_h_
