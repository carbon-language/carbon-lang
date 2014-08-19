//===-- AppleObjCTypeEncodingParser.h ---------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AppleObjCTypeEncodingParser_h_
#define liblldb_AppleObjCTypeEncodingParser_h_

// C Includes
// C++ Includes

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/ObjCLanguageRuntime.h"

#include "clang/AST/ASTContext.h"

namespace lldb_utility {
    class StringLexer;
}

namespace lldb_private {

    class AppleObjCTypeEncodingParser : public ObjCLanguageRuntime::EncodingToType
    {
    public:
        AppleObjCTypeEncodingParser (ObjCLanguageRuntime& runtime);
        virtual ClangASTType RealizeType (clang::ASTContext &ast_ctx, const char* name, bool allow_unknownanytype);
        virtual ~AppleObjCTypeEncodingParser() {}
        
    private:
        struct StructElement {
            std::string name;
            clang::QualType type;
            uint32_t bitfield;
            
            StructElement ();
            ~StructElement () = default;
        };
        
        clang::QualType
        BuildType (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool allow_unknownanytype, uint32_t *bitfield_bit_size = nullptr);

        clang::QualType
        BuildStruct (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool allow_unknownanytype);
        
        clang::QualType
        BuildAggregate (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool allow_unknownanytype, char opener, char closer, uint kind);
        
        clang::QualType
        BuildUnion (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool allow_unknownanytype);
        
        clang::QualType
        BuildArray (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool allow_unknownanytype);
        
        std::string
        ReadStructName(lldb_utility::StringLexer& type);
        
        StructElement
        ReadStructElement (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool allow_unknownanytype);
        
        std::string
        ReadStructElementName(lldb_utility::StringLexer& type);
        
        uint
        ReadNumber (lldb_utility::StringLexer& type);

    };
    
} // namespace lldb_private

#endif  // liblldb_AppleObjCTypeEncodingParser_h_
