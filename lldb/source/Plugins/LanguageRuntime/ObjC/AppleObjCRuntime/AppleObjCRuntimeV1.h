//===-- AppleObjCRuntimeV1.h ----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef liblldb_AppleObjCRuntimeV1_h_
#define liblldb_AppleObjCRuntimeV1_h_

// C Includes
// C++ Includes
// Other libraries and framework includes
// Project includes
#include "lldb/lldb-private.h"
#include "lldb/Target/LanguageRuntime.h"
#include "lldb/Target/ObjCLanguageRuntime.h"
#include "lldb/Core/ValueObject.h"
#include "AppleObjCRuntime.h"
#include "AppleObjCTrampolineHandler.h"
#include "AppleThreadPlanStepThroughObjCTrampoline.h"

namespace lldb_private {
    
class AppleObjCRuntimeV1 :
        public AppleObjCRuntime
{
public:
    ~AppleObjCRuntimeV1() { }
    
    // These are generic runtime functions:
    virtual bool
    GetDynamicValue (ValueObject &in_value, lldb::TypeSP &type_sp, Address &address);

    virtual ClangUtilityFunction *
    CreateObjectChecker (const char *);

    //------------------------------------------------------------------
    // Static Functions
    //------------------------------------------------------------------
    static void
    Initialize();
    
    static void
    Terminate();
    
    static lldb_private::LanguageRuntime *
    CreateInstance (Process *process, lldb::LanguageType language);
    
    //------------------------------------------------------------------
    // PluginInterface protocol
    //------------------------------------------------------------------
    virtual const char *
    GetPluginName();
    
    virtual const char *
    GetShortPluginName();
    
    virtual uint32_t
    GetPluginVersion();
    
    virtual void
    SetExceptionBreakpoints ();
    
    virtual ObjCRuntimeVersions
    GetRuntimeVersion ()
    {
        return eAppleObjC_V1;
    }
protected:
    
private:
    AppleObjCRuntimeV1(Process *process) : 
        lldb_private::AppleObjCRuntime (process)
     { } // Call CreateInstance instead.
};
    
} // namespace lldb_private

#endif  // liblldb_AppleObjCRuntimeV1_h_
