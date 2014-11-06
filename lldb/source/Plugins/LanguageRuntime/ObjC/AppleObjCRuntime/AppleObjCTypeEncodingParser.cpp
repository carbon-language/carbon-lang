//===-- AppleObjCTypeEncodingParser.cpp -------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "AppleObjCTypeEncodingParser.h"

#include "lldb/Symbol/ClangASTContext.h"
#include "lldb/Symbol/ClangASTType.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/StringLexer.h"

#include <vector>

using namespace lldb_private;
using namespace lldb_utility;

AppleObjCTypeEncodingParser::AppleObjCTypeEncodingParser (ObjCLanguageRuntime& runtime) :
    ObjCLanguageRuntime::EncodingToType(),
    m_runtime(runtime)
{
    if (!m_scratch_ast_ctx_ap)
        m_scratch_ast_ctx_ap.reset(new ClangASTContext(runtime.GetProcess()->GetTarget().GetArchitecture().GetTriple().str().c_str()));
}

std::string
AppleObjCTypeEncodingParser::ReadStructName(lldb_utility::StringLexer& type)
{
    StreamString buffer;
    while (type.HasAtLeast(1) && type.Peek() != '=')
        buffer.Printf("%c",type.Next());
    return buffer.GetString();
}

std::string
AppleObjCTypeEncodingParser::ReadQuotedString(lldb_utility::StringLexer& type)
{
    StreamString buffer;
    while (type.HasAtLeast(1) && type.Peek() != '"')
        buffer.Printf("%c",type.Next());
    return buffer.GetString();
}

uint32_t
AppleObjCTypeEncodingParser::ReadNumber (lldb_utility::StringLexer& type)
{
    uint32_t total = 0;
    while (type.HasAtLeast(1) && isdigit(type.Peek()))
           total = 10*total + (type.Next() - '0');
    return total;
}

// as an extension to the published grammar recent runtimes emit structs like this:
// "{CGRect=\"origin\"{CGPoint=\"x\"d\"y\"d}\"size\"{CGSize=\"width\"d\"height\"d}}"

AppleObjCTypeEncodingParser::StructElement::StructElement() :
name(""),
type(clang::QualType()),
bitfield(0)
{}

AppleObjCTypeEncodingParser::StructElement
AppleObjCTypeEncodingParser::ReadStructElement (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool for_expression)
{
    StructElement retval;
    if (type.NextIf('"'))
        retval.name = ReadQuotedString(type);
    if (!type.NextIf('"'))
        return retval;
    uint32_t bitfield_size = 0;
    retval.type = BuildType(ast_ctx, type, for_expression, &bitfield_size);
    retval.bitfield = bitfield_size;
    return retval;
}

clang::QualType
AppleObjCTypeEncodingParser::BuildStruct (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool for_expression)
{
    return BuildAggregate(ast_ctx, type, for_expression, '{', '}', clang::TTK_Struct);
}

clang::QualType
AppleObjCTypeEncodingParser::BuildUnion (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool for_expression)
{
    return BuildAggregate(ast_ctx, type, for_expression, '(', ')', clang::TTK_Union);
}

clang::QualType
AppleObjCTypeEncodingParser::BuildAggregate (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool for_expression, char opener, char closer, uint32_t kind)
{
    if (!type.NextIf(opener))
        return clang::QualType();
    std::string name(ReadStructName(type));
    if (!type.NextIf('='))
        return clang::QualType();
    bool in_union = true;
    std::vector<StructElement> elements;
    while (in_union && type.HasAtLeast(1))
    {
        if (type.NextIf(closer))
        {
            in_union = false;
            break;
        }
        else
        {
            auto element = ReadStructElement(ast_ctx, type, for_expression);
            if (element.type.isNull())
                break;
            else
                elements.push_back(element);
        }
    }
    if (in_union)
        return clang::QualType();
    ClangASTContext *lldb_ctx = ClangASTContext::GetASTContext(&ast_ctx);
    if (!lldb_ctx)
        return clang::QualType();
    ClangASTType union_type(lldb_ctx->CreateRecordType(nullptr, lldb::eAccessPublic, name.c_str(), kind, lldb::eLanguageTypeC));
    if (union_type)
    {
        union_type.StartTagDeclarationDefinition();
        
        unsigned int count = 0;
        for (auto element: elements)
        {
            if (element.name.empty())
            {
                StreamString elem_name;
                elem_name.Printf("__unnamed_%u",count);
                element.name = std::string(elem_name.GetData());
            }
            union_type.AddFieldToRecordType(element.name.c_str(), ClangASTType(&ast_ctx,element.type.getAsOpaquePtr()), lldb::eAccessPublic, element.bitfield);
            ++count;
        }
        
        union_type.CompleteTagDeclarationDefinition();
    }
    return union_type.GetQualType();
}

clang::QualType
AppleObjCTypeEncodingParser::BuildArray (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool for_expression)
{
    if (!type.NextIf('['))
        return clang::QualType();
    uint32_t size = ReadNumber(type);
    clang::QualType element_type(BuildType(ast_ctx, type, for_expression));
    if (!type.NextIf(']'))
        return clang::QualType();
    ClangASTContext *lldb_ctx = ClangASTContext::GetASTContext(&ast_ctx);
    if (!lldb_ctx)
        return clang::QualType();
    ClangASTType array_type(lldb_ctx->CreateArrayType(ClangASTType(&ast_ctx,element_type.getAsOpaquePtr()), size, false));
    return array_type.GetQualType();
}

// the runtime can emit these in the form of @"SomeType", giving more specifics
// this would be interesting for expression parser interop, but since we actually try
// to avoid exposing the ivar info to the expression evaluator, consume but ignore the type info
// and always return an 'id'; if anything, dynamic typing will resolve things for us anyway
clang::QualType
AppleObjCTypeEncodingParser::BuildObjCObjectPointerType (clang::ASTContext &ast_ctx, lldb_utility::StringLexer& type, bool for_expression)
{
    if (!type.NextIf('@'))
        return clang::QualType();
    
    std::string name;
    
    if (type.NextIf('"'))
        name = ReadQuotedString(type);
    
    if (for_expression && !name.empty())
    {
        size_t less_than_pos = name.find_first_of('<');
        
        if (less_than_pos != std::string::npos)
        {
            if (less_than_pos == 0)
                return ast_ctx.getObjCIdType();
            else
                name.erase(less_than_pos);
        }
        
        TypeVendor *type_vendor = m_runtime.GetTypeVendor();
        
        assert (type_vendor); // how are we parsing type encodings for expressions if a type vendor isn't in play?
        assert (type_vendor->GetClangASTContext() == &ast_ctx); // it doesn't make sense for us to be looking in other places
        
        const bool append = false;
        const uint32_t max_matches = 1;
        std::vector<ClangASTType> types;
        
        uint32_t num_types = type_vendor->FindTypes(ConstString(name),
                                                    append,
                                                    max_matches,
                                                    types);

        assert(num_types); // how can a type be mentioned in runtime type signatures and not be in the runtime?

        return types[0].GetPointerType().GetQualType();
    }
    else
    {
        // We're going to resolve this dynamically anyway, so just smile and wave.
        return ast_ctx.getObjCIdType();
    }
}

clang::QualType
AppleObjCTypeEncodingParser::BuildType (clang::ASTContext &ast_ctx, StringLexer& type, bool for_expression, uint32_t *bitfield_bit_size)
{
    if (!type.HasAtLeast(1))
        return clang::QualType();
    
    if (type.NextIf('c'))
        return ast_ctx.CharTy;
    if (type.NextIf('i'))
        return ast_ctx.IntTy;
    if (type.NextIf('s'))
        return ast_ctx.ShortTy;
    if (type.NextIf('l'))
    {
        ClangASTContext *lldb_ctx = ClangASTContext::GetASTContext(&ast_ctx);
        if (!lldb_ctx)
            return clang::QualType();
        return lldb_ctx->GetIntTypeFromBitSize(32, true).GetQualType();
    }
    if (type.NextIf('q'))
        return ast_ctx.LongLongTy;
    if (type.NextIf('C'))
        return ast_ctx.UnsignedCharTy;
    if (type.NextIf('I'))
        return ast_ctx.UnsignedIntTy;
    if (type.NextIf('S'))
        return ast_ctx.UnsignedShortTy;
    if (type.NextIf('L'))
    {
        ClangASTContext *lldb_ctx = ClangASTContext::GetASTContext(&ast_ctx);
        if (!lldb_ctx)
            return clang::QualType();
        return lldb_ctx->GetIntTypeFromBitSize(32, false).GetQualType();
    }
    if (type.NextIf('Q'))
        return ast_ctx.UnsignedLongLongTy;
    if (type.NextIf('f'))
        return ast_ctx.FloatTy;
    if (type.NextIf('d'))
        return ast_ctx.DoubleTy;
    if (type.NextIf('B'))
        return ast_ctx.BoolTy;
    if (type.NextIf('v'))
        return ast_ctx.VoidTy;
    if (type.NextIf('*'))
        return ast_ctx.getPointerType(ast_ctx.CharTy);
    if (type.NextIf('#'))
        return ast_ctx.getObjCClassType();
    if (type.NextIf(':'))
        return ast_ctx.getObjCSelType();
    
    if (type.NextIf('b'))
    {
        uint32_t size = ReadNumber(type);
        if (bitfield_bit_size)
        {
            *bitfield_bit_size = size;
            return ast_ctx.UnsignedIntTy; // FIXME: the spec is fairly vague here.
        }
        else
            return clang::QualType();
    }
    
    if (type.NextIf('r'))
    {
        clang::QualType target_type = BuildType(ast_ctx, type, for_expression);
        if (target_type.isNull())
            return clang::QualType();
        else if (target_type == ast_ctx.UnknownAnyTy)
            return ast_ctx.UnknownAnyTy;
        else
            return ast_ctx.getConstType(target_type);
    }
    
    if (type.NextIf('^'))
    {
        if (!for_expression && type.NextIf('?'))
        {
            // if we are not supporting the concept of unknownAny, but what is being created here is an unknownAny*, then
            // we can just get away with a void*
            // this is theoretically wrong (in the same sense as 'theoretically nothing exists') but is way better than outright failure
            // in many practical cases
            return ast_ctx.VoidPtrTy;
        }
        else
        {
            clang::QualType target_type = BuildType(ast_ctx, type, for_expression);
            if (target_type.isNull())
                return clang::QualType();
            else if (target_type == ast_ctx.UnknownAnyTy)
                return ast_ctx.UnknownAnyTy;
            else
                return ast_ctx.getPointerType(target_type);
        }
    }
    
    if (type.NextIf('?'))
        return for_expression ? ast_ctx.UnknownAnyTy : clang::QualType();
    
    if (type.Peek() == '{')
        return BuildStruct(ast_ctx, type, for_expression);
    
    if (type.Peek() == '[')
        return BuildArray(ast_ctx, type, for_expression);
    
    if (type.Peek() == '(')
        return BuildUnion(ast_ctx, type, for_expression);
    
    if (type.Peek() == '@')
        return BuildObjCObjectPointerType(ast_ctx, type, for_expression);
    
    return clang::QualType();
}

ClangASTType
AppleObjCTypeEncodingParser::RealizeType (clang::ASTContext &ast_ctx, const char* name, bool for_expression)
{
    if (name && name[0])
    {
        StringLexer lexer(name);
        clang::QualType qual_type = BuildType(ast_ctx, lexer, for_expression);
        return ClangASTType(&ast_ctx, qual_type.getAsOpaquePtr());
    }
    return ClangASTType();
}

