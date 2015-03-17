//===-- CXXFormatterFunctions.h------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_CXXFormatterFunctions_h_
#define liblldb_CXXFormatterFunctions_h_

#include <stdint.h>
#include <time.h>

#include "lldb/lldb-forward.h"

#include "lldb/Core/ConstString.h"
#include "lldb/DataFormatters/FormatClasses.h"
#include "lldb/DataFormatters/TypeSynthetic.h"
#include "lldb/Target/ExecutionContext.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Target/Target.h"

#include "clang/AST/ASTContext.h"

namespace lldb_private {
    namespace formatters
    {
        StackFrame*
        GetViableFrame (ExecutionContext exe_ctx);
        
        bool
        ExtractValueFromObjCExpression (ValueObject &valobj,
                                        const char* target_type,
                                        const char* selector,
                                        uint64_t &value);
        
        bool
        ExtractSummaryFromObjCExpression (ValueObject &valobj,
                                          const char* target_type,
                                          const char* selector,
                                          Stream &stream);

        lldb::ValueObjectSP
        CallSelectorOnObject (ValueObject &valobj,
                              const char* return_type,
                              const char* selector,
                              uint64_t index);
        
        lldb::ValueObjectSP
        CallSelectorOnObject (ValueObject &valobj,
                              const char* return_type,
                              const char* selector,
                              const char* key);
        
        size_t
        ExtractIndexFromString (const char* item_name);
        
        time_t
        GetOSXEpoch ();
        
        bool
        FunctionPointerSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options); // C++ function pointer
        
        bool
        Char16StringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options); // char16_t* and unichar*
        
        bool
        Char32StringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options); // char32_t*
        
        bool
        WCharStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options); // wchar_t*
        
        bool
        Char16SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options); // char16_t and unichar
        
        bool
        Char32SummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options); // char32_t
        
        bool
        WCharSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options); // wchar_t
        
        bool
        LibcxxStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options); // libc++ std::string

        bool
        LibcxxWStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options); // libc++ std::wstring

        bool
        LibcxxSmartPointerSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options); // libc++ std::shared_ptr<> and std::weak_ptr<>
        
        bool
        ObjCClassSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        SyntheticChildrenFrontEnd* ObjCClassSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        template<bool name_entries>
        bool
        NSDictionarySummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSIndexSetSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSArraySummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        template<bool cf_style>
        bool
        NSSetSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        template<bool needs_at>
        bool
        NSDataSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSNumberSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);

        bool
        NSNotificationSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSTimeZoneSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSMachPortSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        CFBagSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        CFBinaryHeapSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        CFBitVectorSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSDateSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        CFAbsoluteTimeSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSBundleSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSTaggedString_SummaryProvider (ObjCLanguageRuntime::ClassDescriptorSP descriptor, Stream& stream);
        
        bool
        NSAttributedStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSMutableAttributedStringSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        NSURLSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        ObjCBOOLSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        template <bool is_sel_ptr>
        bool
        ObjCSELSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        bool
        RuntimeSpecificDescriptionSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        extern template bool
        NSDictionarySummaryProvider<true> (ValueObject&, Stream&, const TypeSummaryOptions&) ;
        
        extern template bool
        NSDictionarySummaryProvider<false> (ValueObject&, Stream&, const TypeSummaryOptions&) ;
        
        extern template bool
        NSDataSummaryProvider<true> (ValueObject&, Stream&, const TypeSummaryOptions&) ;
        
        extern template bool
        NSDataSummaryProvider<false> (ValueObject&, Stream&, const TypeSummaryOptions&) ;
        
        extern template bool
        ObjCSELSummaryProvider<true> (ValueObject&, Stream&, const TypeSummaryOptions&);

        extern template bool
        ObjCSELSummaryProvider<false> (ValueObject&, Stream&, const TypeSummaryOptions&);
        
        bool
        CMTimeSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        SyntheticChildrenFrontEnd* NSArraySyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* NSDictionarySyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* NSSetSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* NSIndexPathSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class LibcxxVectorBoolSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibcxxVectorBoolSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibcxxVectorBoolSyntheticFrontEnd ();
        private:
            ClangASTType m_bool_type;
            ExecutionContextRef m_exe_ctx_ref;
            uint64_t m_count;
            lldb::addr_t m_base_data_address;
            std::map<size_t,lldb::ValueObjectSP> m_children;
        };
        
        SyntheticChildrenFrontEnd* LibcxxVectorBoolSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        bool
        LibcxxContainerSummaryProvider (ValueObject& valobj, Stream& stream, const TypeSummaryOptions& options);
        
        class LibstdcppMapIteratorSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibstdcppMapIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibstdcppMapIteratorSyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            lldb::addr_t m_pair_address;
            ClangASTType m_pair_type;
            EvaluateExpressionOptions m_options;
            lldb::ValueObjectSP m_pair_sp;
        };
        
        SyntheticChildrenFrontEnd* LibstdcppMapIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class LibCxxMapIteratorSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibCxxMapIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibCxxMapIteratorSyntheticFrontEnd ();
        private:
            ValueObject *m_pair_ptr;
        };
        
        SyntheticChildrenFrontEnd* LibCxxMapIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);

        class VectorIteratorSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            VectorIteratorSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp,
                                             ConstString item_name);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~VectorIteratorSyntheticFrontEnd ();
        private:
            ExecutionContextRef m_exe_ctx_ref;
            ConstString m_item_name;
            lldb::ValueObjectSP m_item_sp;
        };
        
        SyntheticChildrenFrontEnd* LibCxxVectorIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* LibStdcppVectorIteratorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        class LibcxxSharedPtrSyntheticFrontEnd : public SyntheticChildrenFrontEnd
        {
        public:
            LibcxxSharedPtrSyntheticFrontEnd (lldb::ValueObjectSP valobj_sp);
            
            virtual size_t
            CalculateNumChildren ();
            
            virtual lldb::ValueObjectSP
            GetChildAtIndex (size_t idx);
            
            virtual bool
            Update();
            
            virtual bool
            MightHaveChildren ();
            
            virtual size_t
            GetIndexOfChildWithName (const ConstString &name);
            
            virtual
            ~LibcxxSharedPtrSyntheticFrontEnd ();
        private:
            ValueObject* m_cntrl;
            lldb::ValueObjectSP m_count_sp;
            lldb::ValueObjectSP m_weak_count_sp;
            uint8_t m_ptr_size;
            lldb::ByteOrder m_byte_order;
        };
        
        SyntheticChildrenFrontEnd* LibcxxSharedPtrSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* LibcxxStdVectorSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* LibcxxStdListSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* LibcxxStdMapSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* LibcxxStdUnorderedMapSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* LibcxxInitializerListSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
        
        SyntheticChildrenFrontEnd* VectorTypeSyntheticFrontEndCreator (CXXSyntheticChildren*, lldb::ValueObjectSP);
    } // namespace formatters
} // namespace lldb_private

#endif // liblldb_CXXFormatterFunctions_h_
