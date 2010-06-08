//===-- ClangForward.h ------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ClangForward_h_
#define liblldb_ClangForward_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes

#if defined(__cplusplus)

namespace clang
{
    namespace Builtin
    {
        class Context;
    }

    class ASTContext;
    class AddrLabelExpr;
    class BinaryOperator;
    class CodeGenerator;
    class CompilerInstance;
    class CXXBaseSpecifier;
    class CXXBoolLiteralExpr;
    class CXXFunctionalCastExpr;
    class CXXNamedCastExpr;
    class CXXRecordDecl;
    class CXXThisExpr;
    class CharacterLiteral;
    class CompoundAssignOperator;
    class Decl;
    class DeclaratorDecl;
    class DeclContext;
    class DeclRefExpr;
    class DeclStmt;
    class Diagnostic;
    class EnumDecl;
    class Expr;
    class ExtVectorElementExpr;
    class FieldDecl;
    class FloatingLiteral;
    class FunctionDecl;
    class GotoStmt;
    class IdentifierTable;
    class IntegerLiteral;
    class LabelStmt;
    class LangOptions;
    class MemberExpr;
    class NamedDecl;
    class NamespaceDecl;
    class NonTypeTemplateParmDecl;
    class ObjCEncodeExpr;
    class ObjCImplicitSetterGetterRefExpr;
    class ObjCInterfaceDecl;
    class ObjCIvarRefExpr;
    class ObjCMessageExpr;
    class ObjCMethodDecl;
    class ObjCPropertyRefExpr;
    class ObjCProtocolDecl;
    class ObjCProtocolExpr;
    class ObjCSelectorExpr;
    class ObjCSuperExpr;
    class ParenExpr;
    class ParmVarDecl;
    class PredefinedExpr;
    class QualType;
    class QualifiedNameType;
    class RecordDecl;
    class SelectorTable;
    class SizeOfAlignOfExpr;
    class SourceLocation;
    class SourceManager;
    class Stmt;
    class StmtIteratorBase;
    class StringLiteral;
    class TagDecl;
    class TargetInfo;
    class TargetOptions;
    class TemplateArgument;
    class TemplateDecl;
    class TemplateTemplateParmDecl;
    class TemplateTypeParmDecl;
    class TextDiagnosticBuffer;
    class Type;
    class TypedefDecl;
    class TypesCompatibleExpr;
    class UnaryOperator;
    class ValueDecl;
    class VarDecl;
    struct PrintingPolicy;
}

#endif  // #if defined(__cplusplus)
#endif  // liblldb_ClangForward_h_
