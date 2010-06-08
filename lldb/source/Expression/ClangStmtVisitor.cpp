//===-- ClangStmtVisitor.cpp ------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "lldb/Expression/ClangStmtVisitor.h"

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "clang/AST/RecordLayout.h"

#define NO_RTTI
#include "lldb/Core/dwarf.h"
#include "lldb/Core/Scalar.h"
#include "lldb/Core/StreamString.h"
#include "lldb/Expression/ClangExpressionDeclMap.h"
#include "lldb/Expression/ClangExpressionVariable.h"

//#define ENABLE_DEBUG_PRINTF // COMMENT THIS LINE OUT PRIOR TO CHECKIN
#ifdef ENABLE_DEBUG_PRINTF
#include <stdio.h>
#define DEBUG_PRINTF(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define DEBUG_PRINTF(fmt, ...)
#endif

// Project includes

static lldb_private::Scalar::Type
GetScalarTypeForClangType (clang::ASTContext &ast_context, clang::QualType clang_type, uint32_t &count)
{
    count = 1;

    switch (clang_type->getTypeClass())
    {
    case clang::Type::FunctionNoProto:
    case clang::Type::FunctionProto:
        break;

    case clang::Type::IncompleteArray:
    case clang::Type::VariableArray:
        break;

    case clang::Type::ConstantArray:
        break;

    case clang::Type::ExtVector:
    case clang::Type::Vector:
        // TODO: Set this to more than one???
        break;

    case clang::Type::Builtin:
        switch (cast<clang::BuiltinType>(clang_type)->getKind())
        {
        default: assert(0 && "Unknown builtin type!");
        case clang::BuiltinType::Void:
            break;

        case clang::BuiltinType::Bool:
        case clang::BuiltinType::Char_S:
        case clang::BuiltinType::SChar:
        case clang::BuiltinType::WChar:
        case clang::BuiltinType::Char16:
        case clang::BuiltinType::Char32:
        case clang::BuiltinType::Short:
        case clang::BuiltinType::Int:
        case clang::BuiltinType::Long:
        case clang::BuiltinType::LongLong:
        case clang::BuiltinType::Int128:
            return lldb_private::Scalar::GetValueTypeForSignedIntegerWithByteSize ((ast_context.getTypeSize(clang_type)/CHAR_BIT)/count);

        case clang::BuiltinType::Char_U:
        case clang::BuiltinType::UChar:
        case clang::BuiltinType::UShort:
        case clang::BuiltinType::UInt:
        case clang::BuiltinType::ULong:
        case clang::BuiltinType::ULongLong:
        case clang::BuiltinType::UInt128:
        case clang::BuiltinType::NullPtr:
            return lldb_private::Scalar::GetValueTypeForUnsignedIntegerWithByteSize ((ast_context.getTypeSize(clang_type)/CHAR_BIT)/count);

        case clang::BuiltinType::Float:
        case clang::BuiltinType::Double:
        case clang::BuiltinType::LongDouble:
            return lldb_private::Scalar::GetValueTypeForFloatWithByteSize ((ast_context.getTypeSize(clang_type)/CHAR_BIT)/count);
        }
        break;
    // All pointer types are represented as unsigned integer encodings.
    // We may nee to add a eEncodingPointer if we ever need to know the
    // difference
    case clang::Type::ObjCObjectPointer:
    case clang::Type::BlockPointer:
    case clang::Type::Pointer:
    case clang::Type::LValueReference:
    case clang::Type::RValueReference:
    case clang::Type::MemberPointer:
        return lldb_private::Scalar::GetValueTypeForUnsignedIntegerWithByteSize ((ast_context.getTypeSize(clang_type)/CHAR_BIT)/count);

    // Complex numbers are made up of floats
    case clang::Type::Complex:
        count = 2;
        return lldb_private::Scalar::GetValueTypeForFloatWithByteSize ((ast_context.getTypeSize(clang_type)/CHAR_BIT) / count);

    case clang::Type::ObjCInterface:            break;
    case clang::Type::Record:                   break;
    case clang::Type::Enum:
        return lldb_private::Scalar::GetValueTypeForSignedIntegerWithByteSize ((ast_context.getTypeSize(clang_type)/CHAR_BIT)/count);

    case clang::Type::Typedef:
        return GetScalarTypeForClangType(ast_context, cast<clang::TypedefType>(clang_type)->LookThroughTypedefs(), count);
        break;

    case clang::Type::TypeOfExpr:
    case clang::Type::TypeOf:
    case clang::Type::Decltype:
    //case clang::Type::QualifiedName:
    case clang::Type::TemplateSpecialization:   break;
    }
    count = 0;
    return lldb_private::Scalar::e_void;
}

//----------------------------------------------------------------------
// ClangStmtVisitor constructor
//----------------------------------------------------------------------
lldb_private::ClangStmtVisitor::ClangStmtVisitor
(
    clang::ASTContext &ast_context,
    lldb_private::ClangExpressionVariableList &variable_list,
    lldb_private::ClangExpressionDeclMap *decl_map,
    lldb_private::StreamString &strm
) :
    m_ast_context (ast_context),
    m_variable_list (variable_list),
    m_decl_map (decl_map),
    m_stream (strm)
{
}

//----------------------------------------------------------------------
// Destructor
//----------------------------------------------------------------------
lldb_private::ClangStmtVisitor::~ClangStmtVisitor()
{
}

CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitStmt (clang::Stmt *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);

    clang::Stmt::child_iterator pos;
    clang::Stmt::child_iterator begin = Node->child_begin();
    clang::Stmt::child_iterator end = Node->child_end();
    bool clear_before_next_stmt = false;
    for (pos = begin; pos != end; ++pos)
    {
#ifdef ENABLE_DEBUG_PRINTF
        pos->dump();
#endif
        clang::Stmt *child_stmt = *pos;
        uint32_t pre_visit_stream_offset = m_stream.GetSize();
        bool not_null_stmt = dyn_cast<clang::NullStmt>(child_stmt) == NULL;
        if (clear_before_next_stmt && not_null_stmt)
            m_stream.PutHex8(DW_OP_APPLE_clear);
        Visit (child_stmt);
        if (not_null_stmt)
            clear_before_next_stmt = pre_visit_stream_offset != m_stream.GetSize();
    }
}

CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitDeclStmt (clang::DeclStmt *decl_stmt)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
    clang::DeclGroupRef decl_group_ref = decl_stmt->getDeclGroup();
    clang::DeclGroupRef::iterator pos, end = decl_group_ref.end();
    for (pos = decl_group_ref.begin(); pos != end; ++pos)
    {
        clang::Decl *decl = *pos;
        if (decl)
        {
            clang::Decl::Kind decl_kind = decl->getKind();

            switch (decl_kind)
            {
            case clang::Decl::Namespace:
            case clang::Decl::Enum:
            case clang::Decl::Record:
            case clang::Decl::CXXRecord:
            case clang::Decl::ObjCMethod:
            case clang::Decl::ObjCInterface:
            case clang::Decl::ObjCCategory:
            case clang::Decl::ObjCProtocol:
            case clang::Decl::ObjCImplementation:
            case clang::Decl::ObjCCategoryImpl:
            case clang::Decl::LinkageSpec:
            case clang::Decl::Block:
            case clang::Decl::Function:
            case clang::Decl::CXXMethod:
            case clang::Decl::CXXConstructor:
            case clang::Decl::CXXDestructor:
            case clang::Decl::CXXConversion:
            case clang::Decl::Field:
            case clang::Decl::Typedef:
            case clang::Decl::EnumConstant:
            case clang::Decl::ImplicitParam:
            case clang::Decl::ParmVar:
            case clang::Decl::ObjCProperty:
                break;

            case clang::Decl::Var:
                {
                    const clang::VarDecl *var_decl = cast<clang::VarDecl>(decl)->getCanonicalDecl();
                    uint32_t expr_local_var_idx = UINT32_MAX;
                    if (m_variable_list.GetVariableForVarDecl (m_ast_context, var_decl, expr_local_var_idx, true))
                    {
                        const clang::Expr* var_decl_expr = var_decl->getAnyInitializer();
                        // If there is an inialization expression, then assign the
                        // variable.
                        if (var_decl_expr)
                        {
                            m_stream.PutHex8(DW_OP_APPLE_expr_local);
                            m_stream.PutULEB128(expr_local_var_idx);
                            Visit ((clang::Stmt *)var_decl_expr);
                            m_stream.PutHex8(DW_OP_APPLE_assign);
                        }
                    }
                }
                break;

            default:
                assert(!"decl unhandled");
                break;
            }
        }
    }
}

CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitLabelStmt (clang::LabelStmt *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitGotoStmt (clang::GotoStmt *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


// Exprs
CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitExpr (clang::Expr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitDeclRefExpr (clang::DeclRefExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
    clang::NamedDecl *decl = Node->getDecl();
    clang::QualType clang_type = Node->getType();

#ifdef ENABLE_DEBUG_PRINTF
    //decl->dump();
    //clang_type.dump("lldb_private::ClangStmtVisitor::VisitDeclRefExpr() -> clang_type.dump() = ");
#endif
    uint32_t expr_local_var_idx = UINT32_MAX;
    if (m_variable_list.GetVariableForVarDecl (m_ast_context, cast<clang::VarDecl>(decl)->getCanonicalDecl(), expr_local_var_idx, false) &&
        expr_local_var_idx != UINT32_MAX)
    {
        m_stream.PutHex8(DW_OP_APPLE_expr_local);
        m_stream.PutULEB128(expr_local_var_idx);
    }
    else if (m_decl_map &&
             m_decl_map->GetIndexForDecl(expr_local_var_idx, decl->getCanonicalDecl()))
    {
        m_stream.PutHex8(DW_OP_APPLE_extern);
        m_stream.PutULEB128(expr_local_var_idx);
    }
    else
    {
        m_stream.PutHex8 (DW_OP_APPLE_error);
    }
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitPredefinedExpr (clang::PredefinedExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitCharacterLiteral (clang::CharacterLiteral *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
    clang::QualType clang_type = Node->getType();
    uint64_t clang_type_size = m_ast_context.getTypeSize (clang_type);
    if (clang_type_size <= 64)
    {
        // Encode the integer into our DWARF expression
        if (clang_type->isSignedIntegerType())
            EncodeSInt64(Node->getValue(), clang_type_size);
        else
            EncodeUInt64(Node->getValue(), clang_type_size);
    }
    else
    {
        // TODO: eventually support integer math over 64 bits, probably using
        // APInt as the class.
        m_stream.PutHex8(DW_OP_APPLE_error);
    }
}

bool
lldb_private::ClangStmtVisitor::EncodeUInt64 (uint64_t uval, uint32_t bit_size)
{
    // If "bit_size" is zero, then encode "uval" in the most efficient way
    if (bit_size <= 8 || (bit_size == 0 && uval <= UINT8_MAX))
    {
        m_stream.PutHex8 (DW_OP_const1u);
        m_stream.PutHex8 (uval);
    }
    else if (bit_size <= 16 || (bit_size == 0 && uval <= UINT16_MAX))
    {
        m_stream.PutHex8 (DW_OP_const2u);
        m_stream.PutHex16 (uval);
    }
    else if (bit_size <= 32 || (bit_size == 0 && uval <= UINT32_MAX))
    {
        m_stream.PutHex8 (DW_OP_const4u);
        m_stream.PutHex32 (uval);
    }
    else if (bit_size <= 64 || (bit_size == 0))
    {
        m_stream.PutHex8 (DW_OP_const8u);
        m_stream.PutHex64 (uval);
    }
    else
    {
        m_stream.PutHex8 (DW_OP_APPLE_error);
        return false;
    }
    return true;
}

bool
lldb_private::ClangStmtVisitor::EncodeSInt64 (int64_t sval, uint32_t bit_size)
{
    if (bit_size <= 8 || (bit_size == 0 && INT8_MIN <= sval && sval <= INT8_MAX))
    {
        m_stream.PutHex8 (DW_OP_const1s);
        m_stream.PutHex8 (sval);
    }
    else if (bit_size <= 16 || (bit_size == 0 && INT16_MIN <= sval && sval <= INT16_MAX))
    {
        m_stream.PutHex8 (DW_OP_const2s);
        m_stream.PutHex16 (sval);
    }
    else if (bit_size <= 32 || (bit_size == 0 && INT32_MIN <= sval && sval <= INT32_MAX))
    {
        m_stream.PutHex8 (DW_OP_const4s);
        m_stream.PutHex32 (sval);
    }
    else if (bit_size <= 64 || (bit_size == 0))
    {
        m_stream.PutHex8 (DW_OP_const8s);
        m_stream.PutHex64 (sval);
    }
    else
    {
        m_stream.PutHex8 (DW_OP_APPLE_error);
        return false;
    }
    return true;
}

CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitIntegerLiteral (clang::IntegerLiteral *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
    const llvm::APInt &ap_int = Node->getValue();
    if (ap_int.getBitWidth() <= 64)
    {
        clang::QualType clang_type = Node->getType();
        uint64_t clang_type_size = m_ast_context.getTypeSize (clang_type);
        // Encode the integer into our DWARF expression
        if (clang_type->isSignedIntegerType())
            EncodeSInt64(ap_int.getLimitedValue(), clang_type_size);
        else
            EncodeUInt64(ap_int.getLimitedValue(), clang_type_size);
    }
    else
    {
        // TODO: eventually support integer math over 64 bits, probably using
        // APInt as the class.
        m_stream.PutHex8(DW_OP_APPLE_error);
    }
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitFloatingLiteral (clang::FloatingLiteral *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
    const llvm::APFloat &ap_float = Node->getValue();
    // Put the length of the float in bytes into a single byte
    llvm::APInt ap_int(ap_float.bitcastToAPInt());
    const unsigned byte_size = ap_int.getBitWidth() / CHAR_BIT;
    if (byte_size == sizeof(float))
    {
        if (sizeof(float) == 4)
        {
            m_stream.PutHex8(DW_OP_APPLE_constf);
            m_stream.PutHex8 (byte_size);
            m_stream.PutHex32 (ap_int.getLimitedValue());
            return;
        }
        else if (sizeof(float) == 8)
        {
            m_stream.PutHex8(DW_OP_APPLE_constf);
            m_stream.PutHex8 (byte_size);
            m_stream.PutHex64 (ap_int.getLimitedValue());
            return;
        }
    }
    else if (byte_size == sizeof(double))
    {
        if (sizeof(double) == 4)
        {
            m_stream.PutHex8(DW_OP_APPLE_constf);
            m_stream.PutHex8 (byte_size);
            m_stream.PutHex32 (ap_int.getLimitedValue());
            return;
        }
        else if (sizeof(double) == 8)
        {
            m_stream.PutHex8(DW_OP_APPLE_constf);
            m_stream.PutHex8 (byte_size);
            m_stream.PutHex64 (ap_int.getLimitedValue());
            return;
        }
    }
    else if (byte_size == sizeof(long double))
    {
        if (sizeof(long double) == 8)
        {
            m_stream.PutHex8(DW_OP_APPLE_constf);
            m_stream.PutHex8 (byte_size);
            m_stream.PutHex64 (ap_int.getLimitedValue());
            return;
        }
    }
    // TODO: eventually support float constants of all sizes using
    // APFloat as the class.
    m_stream.PutHex8(DW_OP_APPLE_error);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitStringLiteral (clang::StringLiteral *Str)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
    
    size_t byte_length = Str->getByteLength();
    bool is_wide = Str->isWide();
    
    size_t new_length = byte_length + (is_wide ? 1 : 2);
    
    uint8_t null_terminated_string[new_length];
    
    memcpy(&null_terminated_string[0], Str->getStrData(), byte_length);
    
    if(is_wide)
    {
        null_terminated_string[byte_length] = '\0';
        null_terminated_string[byte_length + 1] = '\0';
    }
    else 
    {
        null_terminated_string[byte_length] = '\0';
    }
    
    Value *val = new Value(null_terminated_string, new_length);
    val->SetContext(Value::eContextTypeOpaqueClangQualType, Str->getType().getAsOpaquePtr());
    
    uint32_t val_idx = m_variable_list.AppendValue(val);
    
    m_stream.PutHex8(DW_OP_APPLE_expr_local);
    m_stream.PutULEB128(val_idx);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitUnaryOperator (clang::UnaryOperator *unary_op)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);

    Visit(unary_op->getSubExpr());

    switch (unary_op->getOpcode())
    {
    case clang::UnaryOperator::PostInc:
        // Duplciate the top of stack value (which must be something that can
        // be assignable/incremented) and push its current value
        m_stream.PutHex8 (DW_OP_dup);               // x, x
        m_stream.PutHex8 (DW_OP_APPLE_value_of);    // x, val(x)
        m_stream.PutHex8 (DW_OP_swap);              // val(x), x
        m_stream.PutHex8 (DW_OP_dup);               // val(x), x, x
        m_stream.PutHex8 (DW_OP_lit1);              // val(x), x, x, 1
        m_stream.PutHex8 (DW_OP_plus);              // val(x), x, val(x)+1
        m_stream.PutHex8 (DW_OP_APPLE_assign);      // val(x), x
        m_stream.PutHex8 (DW_OP_drop);              // val(x)
        break;

    case clang::UnaryOperator::PostDec:
        // Duplciate the top of stack value (which must be something that can
        // be assignable/incremented) and push its current value
        m_stream.PutHex8 (DW_OP_dup);               // x, x
        m_stream.PutHex8 (DW_OP_APPLE_value_of);    // x, val(x)
        m_stream.PutHex8 (DW_OP_swap);              // val(x), x
        m_stream.PutHex8 (DW_OP_dup);               // val(x), x, x
        m_stream.PutHex8 (DW_OP_lit1);              // val(x), x, x, 1
        m_stream.PutHex8 (DW_OP_minus);             // val(x), x, val(x)-1
        m_stream.PutHex8 (DW_OP_APPLE_assign);      // val(x), x
        m_stream.PutHex8 (DW_OP_drop);              // val(x)
        break;

    case clang::UnaryOperator::PreInc:
        m_stream.PutHex8 (DW_OP_dup);               // x, x
        m_stream.PutHex8 (DW_OP_APPLE_value_of);    // x, val(x)
        m_stream.PutHex8 (DW_OP_lit1);              // x, val(x), 1
        m_stream.PutHex8 (DW_OP_plus);              // x, val(x)+1
        m_stream.PutHex8 (DW_OP_APPLE_assign);      // x with new value
        break;

    case clang::UnaryOperator::PreDec:
        m_stream.PutHex8 (DW_OP_dup);               // x, x
        m_stream.PutHex8 (DW_OP_APPLE_value_of);    // x, val(x)
        m_stream.PutHex8 (DW_OP_lit1);              // x, val(x), 1
        m_stream.PutHex8 (DW_OP_minus);             // x, val(x)-1
        m_stream.PutHex8 (DW_OP_APPLE_assign);      // x with new value
        break;

    case clang::UnaryOperator::AddrOf:
        m_stream.PutHex8 (DW_OP_APPLE_address_of);
        break;

    case clang::UnaryOperator::Deref:
        m_stream.PutHex8 (DW_OP_APPLE_deref_type);
        break;

    case clang::UnaryOperator::Plus:
        m_stream.PutHex8 (DW_OP_abs);
        break;

    case clang::UnaryOperator::Minus:
        m_stream.PutHex8 (DW_OP_neg);
        break;

    case clang::UnaryOperator::Not:
        m_stream.PutHex8 (DW_OP_not);
        break;

    case clang::UnaryOperator::LNot:
        m_stream.PutHex8 (DW_OP_lit0);
        m_stream.PutHex8 (DW_OP_eq);
        break;

    case clang::UnaryOperator::Real:
        m_stream.PutHex8(DW_OP_APPLE_error);
        break;

    case clang::UnaryOperator::Imag:
        m_stream.PutHex8(DW_OP_APPLE_error);
        break;

    case clang::UnaryOperator::Extension:
        m_stream.PutHex8(DW_OP_APPLE_error);
        break;

    case clang::UnaryOperator::OffsetOf:
        break;

    default:
        assert(!"Unknown unary operator!");
        break;
    }
}

CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitCastExpr (clang::CastExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
//    CastExpr::CastKind cast_kind = Node->getCastKind();
//    switch (cast_kind)
//    {
//    case CastExpr::CK_Unknown:
//    case CastExpr::CK_BitCast:        // Used for reinterpret_cast.
//    case CastExpr::CK_NoOp:           // Used for const_cast.
//    case CastExpr::CK_BaseToDerived: // Base to derived class casts.
//    case CastExpr::CK_DerivedToBase: // Derived to base class casts.
//    case CastExpr::CK_Dynamic: // Dynamic cast.
//    case CastExpr::CK_ToUnion: // Cast to union (GCC extension).
//    case CastExpr::CK_ArrayToPointerDecay: // Array to pointer decay.
//    case CastExpr::CK_FunctionToPointerDecay: // Function to pointer decay.
//    case CastExpr::CK_NullToMemberPointer: // Null pointer to member pointer.
//    case CastExpr::CK_BaseToDerivedMemberPointer: // Member pointer in base class to member pointer in derived class.
//    case CastExpr::CK_DerivedToBaseMemberPointer: // Member pointer in derived class to member pointer in base class.
//    case CastExpr::CK_UserDefinedConversion: // Conversion using a user defined type conversion function.
//    case CastExpr::CK_ConstructorConversion: // Conversion by constructor
//    case CastExpr::CK_IntegralToPointer: // Integral to pointer
//    case CastExpr::CK_PointerToIntegral: // Pointer to integral
//    case CastExpr::CK_ToVoid: // Cast to void
//    case CastExpr::CK_VectorSplat: // Casting from an integer/floating type to an extended
//                         // vector type with the same element type as the src type. Splats the
//                         // src expression into the destination expression.
//    case CastExpr::CK_IntegralCast: // Casting between integral types of different size.
//    case CastExpr::CK_IntegralToFloating: // Integral to floating point.
//    case CastExpr::CK_FloatingToIntegral: // Floating point to integral.
//    case CastExpr::CK_FloatingCast: // Casting between floating types of different size.
//        m_stream.PutHex8(DW_OP_APPLE_error);
//        break;
//    }
    uint32_t cast_type_count = 0;
    lldb_private::Scalar::Type cast_type_encoding = GetScalarTypeForClangType (m_ast_context, Node->getType(), cast_type_count);


    Visit (Node->getSubExpr());

    // Simple scalar cast
    if (cast_type_encoding != lldb_private::Scalar::e_void && cast_type_count == 1)
    {
        // Only cast if our scalar types mismatch
        uint32_t castee_type_count = 0;
        lldb_private::Scalar::Type castee_type_encoding = GetScalarTypeForClangType (m_ast_context, Node->getSubExpr()->getType(), castee_type_count);
        if (cast_type_encoding != castee_type_encoding &&
            castee_type_encoding != lldb_private::Scalar::e_void)
        {
            m_stream.PutHex8(DW_OP_APPLE_scalar_cast);
            m_stream.PutHex8(cast_type_encoding);
        }
    }
    else
    {
        // Handle more complex casts with clang types soon!
        m_stream.PutHex8(DW_OP_APPLE_error);
    }
}

CLANG_STMT_RESULT 
lldb_private::ClangStmtVisitor::VisitArraySubscriptExpr (clang::ArraySubscriptExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
    Visit (Node->getBase());
    Visit (Node->getIdx());
    m_stream.PutHex8(DW_OP_APPLE_array_ref);
}

//
//CLANG_STMT_RESULT
//lldb_private::ClangStmtVisitor::VisitImplicitCastExpr (clang::ImplicitCastExpr *Node)
//{
//    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
//    m_stream.PutHex8(DW_OP_APPLE_scalar_cast);
//    Visit (Node->getSubExpr());
//}

CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitSizeOfAlignOfExpr (clang::SizeOfAlignOfExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitMemberExpr (clang::MemberExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
    clang::Expr *parent = Node->getBase();
    Visit (parent);
    clang::QualType parent_clang_type = parent->getType();
    clang::NamedDecl *member_named_decl = cast<clang::NamedDecl>(Node->getMemberDecl()->getCanonicalDecl());

//  DeclarationName member_name = member->getDeclName();

    clang::Type::TypeClass parent_type_class = parent_clang_type->getTypeClass();
    if (parent_type_class == clang::Type::Pointer)
    {
        clang::PointerType *pointer_type = cast<clang::PointerType>(parent_clang_type.getTypePtr());
        parent_clang_type = pointer_type->getPointeeType();
        parent_type_class = parent_clang_type->getTypeClass();
    }

    switch (parent_type_class)
    {
    case clang::Type::Record:
        {
            const clang::RecordType *record_type = cast<clang::RecordType>(parent_clang_type.getTypePtr());
            const clang::RecordDecl *record_decl = record_type->getDecl();
            assert(record_decl);
            const clang::ASTRecordLayout &record_layout = m_ast_context.getASTRecordLayout(record_decl);
            uint32_t field_idx = 0;
            clang::RecordDecl::field_iterator field, field_end;
            for (field = record_decl->field_begin(), field_end = record_decl->field_end(); field != field_end; ++field, ++field_idx)
            {
                clang::NamedDecl *field_named_decl = cast<clang::NamedDecl>(field->getCanonicalDecl());
                if (field_named_decl == member_named_decl)
                {
                    std::pair<uint64_t, unsigned> field_type_info = m_ast_context.getTypeInfo(field->getType());
                    uint64_t field_bit_offset = record_layout.getFieldOffset (field_idx);
                    uint64_t field_byte_offset = field_bit_offset / 8;
                    uint32_t field_bitfield_bit_size = 0;
                    //uint32_t field_bitfield_bit_offset = field_bit_offset % 8;

                    if (field->isBitField())
                    {
                        clang::Expr* bit_width_expr = field->getBitWidth();
                        if (bit_width_expr)
                        {
                            llvm::APSInt bit_width_apsint;
                            if (bit_width_expr->isIntegerConstantExpr(bit_width_apsint, m_ast_context))
                            {
                                field_bitfield_bit_size = bit_width_apsint.getLimitedValue(UINT32_MAX);
                            }
                        }
                    }

                    if (Node->isArrow())
                    {
                        m_stream.PutHex8(DW_OP_deref);
                    }
                    else
                    {
                        m_stream.PutHex8(DW_OP_APPLE_address_of);
                    }

                    if (field_byte_offset)
                    {
                        if (EncodeUInt64(field_byte_offset, 0))
                        {
                            m_stream.PutHex8(DW_OP_plus);
                        }
                    }
                    m_stream.PutHex8(DW_OP_APPLE_clang_cast);
                    m_stream.PutPointer(field->getType().getAsOpaquePtr());
                    break;
                }
            }
        }
        break;

    default:
        assert(!"Unhandled MemberExpr");
        break;
    }
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitExtVectorElementExpr (clang::ExtVectorElementExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}

CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitParenExpr(clang::ParenExpr *paren_expr)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
    Visit (paren_expr->getSubExpr());
}

CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitInitListExpr (clang::InitListExpr *init_list_expr)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}

CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitBinaryOperator (clang::BinaryOperator *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);

    Visit(Node->getLHS());
    Visit(Node->getRHS());

    switch (Node->getOpcode())
    {
        default: assert(0 && "Unknown binary operator!");
        case clang::BinaryOperator::PtrMemD: m_stream.PutHex8(DW_OP_APPLE_error); break;
        case clang::BinaryOperator::PtrMemI: m_stream.PutHex8(DW_OP_APPLE_error); break;
        case clang::BinaryOperator::Mul:   m_stream.PutHex8(DW_OP_mul);    break;
        case clang::BinaryOperator::Div:   m_stream.PutHex8(DW_OP_div);    break;
        case clang::BinaryOperator::Rem:   m_stream.PutHex8(DW_OP_mod);    break;
        case clang::BinaryOperator::Add:   m_stream.PutHex8(DW_OP_plus);   break;
        case clang::BinaryOperator::Sub:   m_stream.PutHex8(DW_OP_minus);  break;
        case clang::BinaryOperator::Shl:   m_stream.PutHex8(DW_OP_shl);    break;
        case clang::BinaryOperator::Shr:   m_stream.PutHex8(DW_OP_shr);    break;
        case clang::BinaryOperator::LT:    m_stream.PutHex8(DW_OP_lt);     break;
        case clang::BinaryOperator::GT:    m_stream.PutHex8(DW_OP_gt);     break;
        case clang::BinaryOperator::LE:    m_stream.PutHex8(DW_OP_le);     break;
        case clang::BinaryOperator::GE:    m_stream.PutHex8(DW_OP_ge);     break;
        case clang::BinaryOperator::EQ:    m_stream.PutHex8(DW_OP_eq);     break;
        case clang::BinaryOperator::NE:    m_stream.PutHex8(DW_OP_ne);     break;
        case clang::BinaryOperator::And:   m_stream.PutHex8(DW_OP_and);    break;
        case clang::BinaryOperator::Xor:   m_stream.PutHex8(DW_OP_xor);    break;
        case clang::BinaryOperator::Or :   m_stream.PutHex8(DW_OP_or);     break;
        case clang::BinaryOperator::LAnd:
            // Do we need to call an operator here on objects? If so
            // we will need a DW_OP_apple_logical_and
            m_stream.PutHex8(DW_OP_lit0);
            m_stream.PutHex8(DW_OP_ne);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_lit0);
            m_stream.PutHex8(DW_OP_ne);
            m_stream.PutHex8(DW_OP_and);
            break;

        case clang::BinaryOperator::LOr :
            // Do we need to call an operator here on objects? If so
            // we will need a DW_OP_apple_logical_or
            m_stream.PutHex8(DW_OP_lit0);
            m_stream.PutHex8(DW_OP_ne);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_lit0);
            m_stream.PutHex8(DW_OP_ne);
            m_stream.PutHex8(DW_OP_or);
            break;

        case clang::BinaryOperator::Assign:
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::MulAssign:
            m_stream.PutHex8(DW_OP_over);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_mul);
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::DivAssign:
            m_stream.PutHex8(DW_OP_over);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_div);
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::RemAssign:
            m_stream.PutHex8(DW_OP_over);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_mod);
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::AddAssign:
            m_stream.PutHex8(DW_OP_over);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_plus);
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::SubAssign:
            m_stream.PutHex8(DW_OP_over);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_minus);
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::ShlAssign:
            m_stream.PutHex8(DW_OP_over);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_shl);
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::ShrAssign:
            m_stream.PutHex8(DW_OP_over);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_shr);
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::AndAssign:
            m_stream.PutHex8(DW_OP_over);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_and);
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::OrAssign:
            m_stream.PutHex8(DW_OP_over);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_or);
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::XorAssign:
            m_stream.PutHex8(DW_OP_over);
            m_stream.PutHex8(DW_OP_swap);
            m_stream.PutHex8(DW_OP_xor);
            m_stream.PutHex8(DW_OP_APPLE_assign);
            break;

        case clang::BinaryOperator::Comma:
            // Nothing needs to be done here right?
            break;
    }
}


//CLANG_STMT_RESULT
//lldb_private::ClangStmtVisitor::VisitCompoundAssignOperator (CompoundAssignOperator *Node)
//{
//    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
//
//}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitAddrLabelExpr (clang::AddrLabelExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitTypesCompatibleExpr (clang::TypesCompatibleExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}



    // C++
CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitCXXNamedCastExpr (clang::CXXNamedCastExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitCXXBoolLiteralExpr (clang::CXXBoolLiteralExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitCXXThisExpr (clang::CXXThisExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitCXXFunctionalCastExpr (clang::CXXFunctionalCastExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}



    // ObjC
CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitObjCEncodeExpr (clang::ObjCEncodeExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitObjCMessageExpr (clang::ObjCMessageExpr* Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitObjCSelectorExpr (clang::ObjCSelectorExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitObjCProtocolExpr (clang::ObjCProtocolExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitObjCPropertyRefExpr (clang::ObjCPropertyRefExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitObjCImplicitSetterGetterRefExpr (clang::ObjCImplicitSetterGetterRefExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitObjCIvarRefExpr (clang::ObjCIvarRefExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


CLANG_STMT_RESULT
lldb_private::ClangStmtVisitor::VisitObjCSuperExpr (clang::ObjCSuperExpr *Node)
{
    DEBUG_PRINTF("%s\n", __PRETTY_FUNCTION__);
}


