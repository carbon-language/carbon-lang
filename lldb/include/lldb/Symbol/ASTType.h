//===-- ASTType.h -----------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

namespace clang
{
    class ASTContext;
}

namespace lldb_private
{
    
class ASTTypeBase
{
protected:
    ASTTypeBase (void *type, clang::ASTContext *ast_context) :
        m_type(type),
        m_ast_context(ast_context) 
    {
    }
    
    ASTTypeBase (const ASTTypeBase &tw) :
        m_type(tw.m_type),
        m_ast_context(tw.m_ast_context)
    {
    }
    
    ASTTypeBase () :
        m_type(0),
        m_ast_context(0)
    {
    }
    
    ~ASTTypeBase();
    
    ASTTypeBase &operator= (const ASTTypeBase &atb)
    {
        m_type = atb.m_type;
        m_ast_context = atb.m_ast_context;
        return *this;
    }
    
public:
    void *GetType() const
    { 
        return m_type; 
    }
    
    clang::ASTContext *GetASTContext() const
    { 
        return m_ast_context; 
    }
    
private:
    void               *m_type;
    clang::ASTContext  *m_ast_context;
};
    
class ASTType : public ASTTypeBase
{
public:
    ASTType (void *type, clang::ASTContext *ast_context) :
        ASTTypeBase(type, ast_context) { }
    
    ASTType (const ASTType &at) :
        ASTTypeBase(at) { }
    
    ASTType () :
        ASTTypeBase() { }
    
    ~ASTType();
    
    ASTType &operator= (const ASTType &at)
    {
        ASTTypeBase::operator= (at);
        return *this;
    }
};

// For cases in which there are multiple classes of types that are not
// interchangeable, to allow static type checking.
template <unsigned int C> class TaggedASTType : public ASTTypeBase
{
public:
    TaggedASTType (void *type, clang::ASTContext *ast_context) :
        ASTTypeBase(type, ast_context) { }
    
    TaggedASTType (const TaggedASTType<C> &tw) :
        ASTTypeBase(tw) { }
    
    TaggedASTType () :
        ASTTypeBase() { }
    
    ~TaggedASTType() { }
    
    TaggedASTType<C> &operator= (const TaggedASTType<C> &tw)
    {
        ASTTypeBase::operator= (tw);
        return *this;
    }
};
    
}
