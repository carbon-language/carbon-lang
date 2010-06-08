//===-- ClangStmtVisitor.h --------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangStmtVisitor_h_
#define liblldb_ClangStmtVisitor_h_

// C Includes
// C++ Includes

// Other libraries and framework includes
#include "clang/AST/StmtVisitor.h"

// Project includes
#include "lldb/Core/ClangForward.h"

namespace lldb_private {

class StreamString;
class ClangExpressionDeclMap;
class ClangExpressionVariableList;

#define CLANG_STMT_RESULT void

class ClangStmtVisitor : public clang::StmtVisitor<ClangStmtVisitor, CLANG_STMT_RESULT>
{
public:
    //------------------------------------------------------------------
    // Constructors and Destructors
    //------------------------------------------------------------------
    ClangStmtVisitor (clang::ASTContext &ast_context, 
                      ClangExpressionVariableList &variable_list, 
                      ClangExpressionDeclMap *decl_map,
                      StreamString &strm);
    virtual ~ClangStmtVisitor ();

    // Stmts.
    CLANG_STMT_RESULT VisitStmt (clang::Stmt *Node);
    CLANG_STMT_RESULT VisitDeclStmt (clang::DeclStmt *Node);
    CLANG_STMT_RESULT VisitLabelStmt (clang::LabelStmt *Node);
    CLANG_STMT_RESULT VisitGotoStmt (clang::GotoStmt *Node);

    // Exprs
    CLANG_STMT_RESULT VisitExpr (clang::Expr *Node);
    CLANG_STMT_RESULT VisitDeclRefExpr (clang::DeclRefExpr *Node);
    CLANG_STMT_RESULT VisitPredefinedExpr (clang::PredefinedExpr *Node);
    CLANG_STMT_RESULT VisitCharacterLiteral (clang::CharacterLiteral *Node);
    CLANG_STMT_RESULT VisitIntegerLiteral (clang::IntegerLiteral *Node);
    CLANG_STMT_RESULT VisitFloatingLiteral (clang::FloatingLiteral *Node);
    CLANG_STMT_RESULT VisitStringLiteral (clang::StringLiteral *Str);
    CLANG_STMT_RESULT VisitUnaryOperator (clang::UnaryOperator *Node);
    CLANG_STMT_RESULT VisitSizeOfAlignOfExpr (clang::SizeOfAlignOfExpr *Node);
    CLANG_STMT_RESULT VisitMemberExpr (clang::MemberExpr *Node);
    CLANG_STMT_RESULT VisitExtVectorElementExpr (clang::ExtVectorElementExpr *Node);
    CLANG_STMT_RESULT VisitBinaryOperator (clang::BinaryOperator *Node);
//  CLANG_STMT_RESULT VisitCompoundAssignOperator (clang::CompoundAssignOperator *Node);
    CLANG_STMT_RESULT VisitAddrLabelExpr (clang::AddrLabelExpr *Node);
    CLANG_STMT_RESULT VisitTypesCompatibleExpr (clang::TypesCompatibleExpr *Node);
    CLANG_STMT_RESULT VisitParenExpr(clang::ParenExpr *Node);
    CLANG_STMT_RESULT VisitInitListExpr (clang::InitListExpr *Node);
    CLANG_STMT_RESULT VisitCastExpr (clang::CastExpr *Node);
//  CLANG_STMT_RESULT VisitImplicitCastExpr (clang::ImplicitCastExpr *Node);
    CLANG_STMT_RESULT VisitArraySubscriptExpr (clang::ArraySubscriptExpr *Node);
    // C++
    CLANG_STMT_RESULT VisitCXXNamedCastExpr (clang::CXXNamedCastExpr *Node);
    CLANG_STMT_RESULT VisitCXXBoolLiteralExpr (clang::CXXBoolLiteralExpr *Node);
    CLANG_STMT_RESULT VisitCXXThisExpr (clang::CXXThisExpr *Node);
    CLANG_STMT_RESULT VisitCXXFunctionalCastExpr (clang::CXXFunctionalCastExpr *Node);

    // ObjC
    CLANG_STMT_RESULT VisitObjCEncodeExpr (clang::ObjCEncodeExpr *Node);
    CLANG_STMT_RESULT VisitObjCMessageExpr (clang::ObjCMessageExpr* Node);
    CLANG_STMT_RESULT VisitObjCSelectorExpr (clang::ObjCSelectorExpr *Node);
    CLANG_STMT_RESULT VisitObjCProtocolExpr (clang::ObjCProtocolExpr *Node);
    CLANG_STMT_RESULT VisitObjCPropertyRefExpr (clang::ObjCPropertyRefExpr *Node);
    CLANG_STMT_RESULT VisitObjCImplicitSetterGetterRefExpr (clang::ObjCImplicitSetterGetterRefExpr *Node);
    CLANG_STMT_RESULT VisitObjCIvarRefExpr (clang::ObjCIvarRefExpr *Node);
    CLANG_STMT_RESULT VisitObjCSuperExpr (clang::ObjCSuperExpr *Node);

protected:
    //------------------------------------------------------------------
    // Classes that inherit from ClangStmtVisitor can see and modify these
    //------------------------------------------------------------------
    clang::ASTContext &m_ast_context;
    ClangExpressionDeclMap *m_decl_map;
    ClangExpressionVariableList &m_variable_list;
    StreamString &m_stream;
private:
    //------------------------------------------------------------------
    // For ClangStmtVisitor only
    //------------------------------------------------------------------
    ClangStmtVisitor (const ClangStmtVisitor&);
    const ClangStmtVisitor& operator= (const ClangStmtVisitor&);

    bool
    EncodeUInt64 (uint64_t uval, uint32_t bit_size);

    bool
    EncodeSInt64 (int64_t sval, uint32_t bit_size);
};

} // namespace lldb_private

#endif  // liblldb_ClangStmtVisitor_h_
