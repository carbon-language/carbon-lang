//===-- FormatClasses.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef lldb_FormatClasses_h_
#define lldb_FormatClasses_h_

// C Includes

#ifdef LLDB_DISABLE_PYTHON

struct PyObject;

#else   // #ifdef LLDB_DISABLE_PYTHON

#if defined (__APPLE__)
#include <Python/Python.h>
#else
#include <Python.h>
#endif

#endif  // #ifdef LLDB_DISABLE_PYTHON


#include <stdint.h>
#include <unistd.h>

// C++ Includes
#include <string>
#include <vector>

// Other libraries and framework includes

// Project includes
#include "lldb/lldb-public.h"
#include "lldb/lldb-enumerations.h"

#include "lldb/Core/ValueObject.h"
#include "lldb/Interpreter/ScriptInterpreterPython.h"

namespace lldb_private {

struct ValueFormat
{
    uint32_t m_my_revision;
    bool m_cascades;
    bool m_skip_pointers;
    bool m_skip_references;
    lldb::Format m_format;
    ValueFormat (lldb::Format f = lldb::eFormatInvalid,
                 bool casc = false,
                 bool skipptr = false,
                 bool skipref = false);
    
    typedef SHARED_PTR(ValueFormat) SharedPointer;
    typedef bool(*ValueCallback)(void*, ConstString, const lldb::ValueFormatSP&);
    
    ~ValueFormat()
    {
    }
    
    bool
    Cascades() const
    {
        return m_cascades;
    }
    bool
    SkipsPointers() const
    {
        return m_skip_pointers;
    }
    bool
    SkipsReferences() const
    {
        return m_skip_references;
    }
    
    lldb::Format
    GetFormat() const
    {
        return m_format;
    }
        
};
    
class SyntheticChildrenFrontEnd
{
protected:
    lldb::ValueObjectSP m_backend;
public:
    
    SyntheticChildrenFrontEnd(lldb::ValueObjectSP be) :
    m_backend(be)
    {}
    
    virtual
    ~SyntheticChildrenFrontEnd()
    {
    }
    
    virtual uint32_t
    CalculateNumChildren() = 0;
    
    virtual lldb::ValueObjectSP
    GetChildAtIndex (uint32_t idx, bool can_create) = 0;
    
    virtual uint32_t
    GetIndexOfChildWithName (const ConstString &name) = 0;
    
    virtual void
    Update() = 0;
    
    typedef SHARED_PTR(SyntheticChildrenFrontEnd) SharedPointer;

};
    
class SyntheticChildren
{
public:
    uint32_t m_my_revision;
    bool m_cascades;
    bool m_skip_pointers;
    bool m_skip_references;
public:
    SyntheticChildren(bool casc = false,
                      bool skipptr = false,
                      bool skipref = false) :
    m_cascades(casc),
    m_skip_pointers(skipptr),
    m_skip_references(skipref)
    {
    }
    
    virtual
    ~SyntheticChildren()
    {
    }
    
    bool
    Cascades() const
    {
        return m_cascades;
    }
    bool
    SkipsPointers() const
    {
        return m_skip_pointers;
    }
    bool
    SkipsReferences() const
    {
        return m_skip_references;
    }
    
    virtual bool
    IsScripted() = 0;
    
    virtual std::string
    GetDescription() = 0;
    
    virtual SyntheticChildrenFrontEnd::SharedPointer
    GetFrontEnd(lldb::ValueObjectSP backend) = 0;
    
    typedef SHARED_PTR(SyntheticChildren) SharedPointer;
    typedef bool(*SyntheticChildrenCallback)(void*, ConstString, const SyntheticChildren::SharedPointer&);
    
};

class SyntheticFilter : public SyntheticChildren
{
    std::vector<std::string> m_expression_paths;
public:
    SyntheticFilter(bool casc = false,
                    bool skipptr = false,
                    bool skipref = false) :
    SyntheticChildren(casc, skipptr, skipref),
    m_expression_paths()
    {
    }
    
    void
    AddExpressionPath(std::string path)
    {
        bool need_add_dot = true;
        if (path[0] == '.' ||
            (path[0] == '-' && path[1] == '>') ||
            path[0] == '[')
            need_add_dot = false;
        // add a '.' symbol to help forgetful users
        if(!need_add_dot)
            m_expression_paths.push_back(path);
        else
            m_expression_paths.push_back(std::string(".") + path);
    }
        
    int
    GetCount() const
    {
        return m_expression_paths.size();
    }
    
    const std::string&
    GetExpressionPathAtIndex(int i) const
    {
        return m_expression_paths[i];
    }
    
    bool
    IsScripted()
    {
        return false;
    }
    
    std::string
    GetDescription();
    
    class FrontEnd : public SyntheticChildrenFrontEnd
    {
    private:
        SyntheticFilter* filter;
    public:
        
        FrontEnd(SyntheticFilter* flt,
                 lldb::ValueObjectSP be) :
        SyntheticChildrenFrontEnd(be),
        filter(flt)
        {}
        
        virtual
        ~FrontEnd()
        {
        }
        
        virtual uint32_t
        CalculateNumChildren()
        {
            return filter->GetCount();
        }
        
        virtual lldb::ValueObjectSP
        GetChildAtIndex (uint32_t idx, bool can_create)
        {
            if (idx >= filter->GetCount())
                return lldb::ValueObjectSP();
            return m_backend->GetSyntheticExpressionPathChild(filter->GetExpressionPathAtIndex(idx).c_str(), can_create);
        }
        
        virtual void
        Update() {}
        
        virtual uint32_t
        GetIndexOfChildWithName (const ConstString &name)
        {
            const char* name_cstr = name.GetCString();
            for (int i = 0; i < filter->GetCount(); i++)
            {
                const char* expr_cstr = filter->GetExpressionPathAtIndex(i).c_str();
                if (::strcmp(name_cstr, expr_cstr))
                    return i;
            }
            return UINT32_MAX;
        }
        
        typedef SHARED_PTR(SyntheticChildrenFrontEnd) SharedPointer;
        
    };
    
    virtual SyntheticChildrenFrontEnd::SharedPointer
    GetFrontEnd(lldb::ValueObjectSP backend)
    {
        return SyntheticChildrenFrontEnd::SharedPointer(new FrontEnd(this, backend));
    }
    
};

#ifndef LLDB_DISABLE_PYTHON

class SyntheticScriptProvider : public SyntheticChildren
{
    std::string m_python_class;
public:
    SyntheticScriptProvider(bool casc = false,
                            bool skipptr = false,
                            bool skipref = false,
                            std::string pclass = "") :
    SyntheticChildren(casc, skipptr, skipref),
    m_python_class(pclass)
    {
    }
    
    
    std::string
    GetPythonClassName()
    {
        return m_python_class;
    }
    
    std::string
    GetDescription();
    
    bool
    IsScripted()
    {
        return true;
    }

    class FrontEnd : public SyntheticChildrenFrontEnd
    {
    private:
        std::string m_python_class;
        void *m_wrapper; // Wraps PyObject.
        ScriptInterpreter *m_interpreter;
    public:
        
        FrontEnd(std::string pclass,
                 lldb::ValueObjectSP be);
        
        virtual
        ~FrontEnd();
        
        virtual uint32_t
        CalculateNumChildren()
        {
            if (m_wrapper == NULL || m_interpreter == NULL)
                return 0;
            return m_interpreter->CalculateNumChildren(m_wrapper);
        }
        
        virtual lldb::ValueObjectSP
        GetChildAtIndex (uint32_t idx, bool can_create);
        
        virtual void
        Update()
        {
            if (m_wrapper == NULL || m_interpreter == NULL)
                return;
            
            m_interpreter->UpdateSynthProviderInstance(m_wrapper);
        }
                
        virtual uint32_t
        GetIndexOfChildWithName (const ConstString &name)
        {
            if (m_wrapper == NULL || m_interpreter == NULL)
                return UINT32_MAX;
            return m_interpreter->GetIndexOfChildWithName(m_wrapper, name.GetCString());
        }
        
        typedef SHARED_PTR(SyntheticChildrenFrontEnd) SharedPointer;
        
    };
    
    virtual SyntheticChildrenFrontEnd::SharedPointer
    GetFrontEnd(lldb::ValueObjectSP backend)
    {
        return SyntheticChildrenFrontEnd::SharedPointer(new FrontEnd(m_python_class, backend));
    }    
};

#endif // #ifndef LLDB_DISABLE_PYTHON
class SyntheticArrayView : public SyntheticChildren
{
public:
    
    struct SyntheticArrayRange
    {
    private:
        int m_low;
        int m_high;
        SyntheticArrayRange* m_next;
        
    public:
        
        SyntheticArrayRange () : 
        m_low(-1),
        m_high(-2),
        m_next(NULL)
        {}
        
        SyntheticArrayRange (int L) : 
        m_low(L),
        m_high(L),
        m_next(NULL)
        {}
        
        SyntheticArrayRange (int L, int H) : 
        m_low(L),
        m_high(H),
        m_next(NULL)
        {}
        
        SyntheticArrayRange (int L, int H, SyntheticArrayRange* N) : 
        m_low(L),
        m_high(H),
        m_next(N)
        {}
        
        inline int
        GetLow ()
        {
            return m_low;
        }
        
        inline int
        GetHigh ()
        {
            return m_high;
        }
        
        inline void
        SetLow (int L)
        {
            m_low = L;
        }
        
        inline void
        SetHigh (int H)
        {
            m_high = H;
        }
        
        inline  int
        GetSelfCount()
        {
            return GetHigh() - GetLow() + 1;
        }
        
        int
        GetCount()
        {
            int count = GetSelfCount();
            if (m_next)
                count += m_next->GetCount();
            return count;
        }
        
        inline SyntheticArrayRange*
        GetNext()
        {
            return m_next;
        }
        
        void
        SetNext(SyntheticArrayRange* N)
        {
            if (m_next)
                delete m_next;
            m_next = N;
        }
        
        void
        SetNext(int L, int H)
        {
            if (m_next)
                delete m_next;
            m_next = new SyntheticArrayRange(L, H);
        }
        
        void
        SetNext(int L)
        {
            if (m_next)
                delete m_next;
            m_next = new SyntheticArrayRange(L);
        }
        
        ~SyntheticArrayRange()
        {
            delete m_next;
            m_next = NULL;
        }
        
    };
    
    SyntheticArrayView(bool casc = false,
                       bool skipptr = false,
                       bool skipref = false) :
    SyntheticChildren(casc, skipptr, skipref),
    m_head(),
    m_tail(&m_head)
    {
    }
    
    void
    AddRange(int L, int H)
    {
        m_tail->SetLow(L);
        m_tail->SetHigh(H);
        m_tail->SetNext(new SyntheticArrayRange());
        m_tail = m_tail->GetNext();
    }
    
    int
    GetCount()
    {
        return m_head.GetCount();
    }
    
    int
    GetRealIndexForIndex(int i);
    
    bool
    IsScripted()
    {
        return false;
    }
    
    std::string
    GetDescription();
    
    class FrontEnd : public SyntheticChildrenFrontEnd
    {
    private:
        SyntheticArrayView* filter;
    public:
        
        FrontEnd(SyntheticArrayView* flt,
                 lldb::ValueObjectSP be) :
        SyntheticChildrenFrontEnd(be),
        filter(flt)
        {}
        
        virtual
        ~FrontEnd()
        {
        }
        
        virtual uint32_t
        CalculateNumChildren()
        {
            return filter->GetCount();
        }
        
        virtual lldb::ValueObjectSP
        GetChildAtIndex (uint32_t idx, bool can_create)
        {
            if (idx >= filter->GetCount())
                return lldb::ValueObjectSP();
            return m_backend->GetSyntheticArrayMember(filter->GetRealIndexForIndex(idx), can_create);
        }
        
        virtual void
        Update() {}
        
        virtual uint32_t
        GetIndexOfChildWithName (const ConstString &name_cs);
        
        typedef SHARED_PTR(SyntheticChildrenFrontEnd) SharedPointer;
        
    };
    
    virtual SyntheticChildrenFrontEnd::SharedPointer
    GetFrontEnd(lldb::ValueObjectSP backend)
    {
        return SyntheticChildrenFrontEnd::SharedPointer(new FrontEnd(this, backend));
    }
private:
    SyntheticArrayRange m_head;
    SyntheticArrayRange *m_tail;
    
};


class SummaryFormat
{
public:
    class Flags
    {
    public:
        
        Flags () :
            m_flags (FVCascades)
        {}
        
        Flags (const Flags& other) :
            m_flags (other.m_flags)
        {}
        
        Flags&
        operator = (const Flags& rhs)
        {
            if (&rhs != this)
                m_flags = rhs.m_flags;
            
            return *this;
        }
        
        Flags&
        Clear()
        {
            m_flags = 0;
            return *this;
        }
        
        bool
        GetCascades () const
        {
            return (m_flags & FVCascades) == FVCascades;
        }
        
        Flags&
        SetCascades (bool value = true)
        {
            if (value)
                m_flags |= FVCascades;
            else
                m_flags &= ~FVCascades;
            return *this;
        }

        bool
        GetSkipPointers () const
        {
            return (m_flags & FVSkipPointers) == FVSkipPointers;
        }

        Flags&
        SetSkipPointers (bool value = true)
        {
            if (value)
                m_flags |= FVSkipPointers;
            else
                m_flags &= ~FVSkipPointers;
            return *this;
        }
        
        bool
        GetSkipReferences () const
        {
            return (m_flags & FVSkipReferences) == FVSkipReferences;
        }
        
        Flags&
        SetSkipReferences (bool value = true)
        {
            if (value)
                m_flags |= FVSkipReferences;
            else
                m_flags &= ~FVSkipReferences;
            return *this;
        }
        
        bool
        GetDontShowChildren () const
        {
            return (m_flags & FVDontShowChildren) == FVDontShowChildren;
        }
        
        Flags&
        SetDontShowChildren (bool value = true)
        {
            if (value)
                m_flags |= FVDontShowChildren;
            else
                m_flags &= ~FVDontShowChildren;
            return *this;
        }
        
        bool
        GetDontShowValue () const
        {
            return (m_flags & FVDontShowValue) == FVDontShowValue;
        }
        
        Flags&
        SetDontShowValue (bool value = true)
        {
            if (value)
                m_flags |= FVDontShowValue;
            else
                m_flags &= ~FVDontShowValue;
            return *this;
        }
        
        bool
        GetShowMembersOneLiner () const
        {
            return (m_flags & FVShowMembersOneLiner) == FVShowMembersOneLiner;
        }
        
        Flags&
        SetShowMembersOneLiner (bool value = true)
        {
            if (value)
                m_flags |= FVShowMembersOneLiner;
            else
                m_flags &= ~FVShowMembersOneLiner;
            return *this;
        }
        
        bool
        GetHideItemNames () const
        {
            return (m_flags & FVHideItemNames) == FVHideItemNames;
        }
        
        Flags&
        SetHideItemNames (bool value = true)
        {
            if (value)
                m_flags |= FVHideItemNames;
            else
                m_flags &= ~FVHideItemNames;
            return *this;
        }
                
    private:
        uint32_t m_flags;
        enum FlagValues
        {
            FVCascades            = 0x0001u,
            FVSkipPointers        = 0x0002u,
            FVSkipReferences      = 0x0004u,
            FVDontShowChildren    = 0x0008u,
            FVDontShowValue       = 0x0010u,
            FVShowMembersOneLiner = 0x0020u,
            FVHideItemNames       = 0x0040u
        };
    };
    
    uint32_t m_my_revision;
    Flags m_flags;
    
    SummaryFormat(const SummaryFormat::Flags& flags);
    
    bool
    Cascades() const
    {
        return m_flags.GetCascades();
    }
    bool
    SkipsPointers() const
    {
        return m_flags.GetSkipPointers();
    }
    bool
    SkipsReferences() const
    {
        return m_flags.GetSkipReferences();
    }
    
    bool
    DoesPrintChildren() const
    {
        return !m_flags.GetDontShowChildren();
    }
    
    bool
    DoesPrintValue() const
    {
        return !m_flags.GetDontShowValue();
    }
    
    bool
    IsOneliner() const
    {
        return m_flags.GetShowMembersOneLiner();
    }
    
    bool
    HideNames() const
    {
        return m_flags.GetHideItemNames();
    }
            
    virtual
    ~SummaryFormat()
    {
    }
    
    virtual std::string
    FormatObject(lldb::ValueObjectSP object) = 0;
    
    virtual std::string
    GetDescription() = 0;
    
    typedef SHARED_PTR(SummaryFormat) SharedPointer;
    typedef bool(*SummaryCallback)(void*, ConstString, const lldb::SummaryFormatSP&);
    typedef bool(*RegexSummaryCallback)(void*, lldb::RegularExpressionSP, const lldb::SummaryFormatSP&);
    
};

// simple string-based summaries, using ${var to show data
struct StringSummaryFormat : public SummaryFormat
{
    std::string m_format;
    
    StringSummaryFormat(const SummaryFormat::Flags& flags,
                        std::string f);
    
    std::string
    GetFormat() const
    {
        return m_format;
    }
    
    virtual
    ~StringSummaryFormat()
    {
    }
    
    virtual std::string
    FormatObject(lldb::ValueObjectSP object);
    
    virtual std::string
    GetDescription();
        
};
    
#ifndef LLDB_DISABLE_PYTHON

// Python-based summaries, running script code to show data
struct ScriptSummaryFormat : public SummaryFormat
{
    std::string m_function_name;
    std::string m_python_script;
    
    ScriptSummaryFormat(const SummaryFormat::Flags& flags,
                        std::string fname,
                        std::string pscri);
    
    std::string
    GetFunctionName() const
    {
        return m_function_name;
    }
    
    std::string
    GetPythonScript() const
    {
        return m_python_script;
    }
    
    virtual
    ~ScriptSummaryFormat()
    {
    }
    
    virtual std::string
    FormatObject(lldb::ValueObjectSP object);
    
    virtual std::string
    GetDescription();
    
    typedef SHARED_PTR(ScriptSummaryFormat) SharedPointer;

};

#endif // #ifndef LLDB_DISABLE_PYTHON

} // namespace lldb_private

#endif	// lldb_FormatClasses_h_
