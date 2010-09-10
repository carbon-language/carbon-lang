//===-- ObjCObjectPrinter.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_ObjCObjectPrinter_h_
#define liblldb_ObjCObjectPrinter_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
#include "lldb/Core/ConstString.h"
#include "lldb/Core/Error.h"
// Project includes

namespace lldb_private {
    
    class ObjCObjectPrinter
    {
    public:
        //------------------------------------------------------------------
        // Constructors and Destructors
        //------------------------------------------------------------------
        ObjCObjectPrinter (Process &process);
        
        virtual
        ~ObjCObjectPrinter ();
        
        bool
        PrintObject (Stream &str, Value &object_ptr, ExecutionContext &exe_ctx);
    protected:
        Process                 &m_process;
        std::auto_ptr<Address>  m_PrintForDebugger_addr;
        
        Address *GetPrintForDebuggerAddr();
    private:
        //------------------------------------------------------------------
        // For ObjCObjectPrinter only
        //------------------------------------------------------------------
        DISALLOW_COPY_AND_ASSIGN (ObjCObjectPrinter);
    };
    
} // namespace lldb_private

#endif  // liblldb_ObjCObjectPrinter_h_
