/*
   lldb.swig

   Created by Caroline Tice 1/18/2010

   This is the input file for SWIG, to create the appropriate C++ wrappers and
   functions for various scripting languages, to enable them to call the
   liblldb Script Bridge functions.

*/

/*  The name of the module to be created.  */

%module lldb

%typemap(in) lldb::ReturnStatus {
  $1 = (int) $input;
}

%typemap(freearg) lldb::ReturnStatus {
}

%typemap(out) lldb::ReturnStatus {
  $result = SWIG_From_unsigned_SS_int(static_cast< unsigned int >($1));
}

/* Typemap definitions, to allow SWIG to properly handle 'char**' data types. */

%typemap(in) char ** {
  /* Check if is a list  */
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $1 = (char **) malloc((size+1) * sizeof(char));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyString_Check(o))
        $1[i] = PyString_AsString(PyList_GetItem($input,i));
      else {
        PyErr_SetString(PyExc_TypeError,"list must contain strings");
        free($1);
        return NULL;
      }
    }
    $1[i] = 0;
  } else if ($input == Py_None) {
    $1 =  NULL;
  } else {
    PyErr_SetString(PyExc_TypeError,"not a list");
    return NULL;
  }
}

%typemap(freearg) char** {
  free((char *) $1);
}

%typemap(out) char** {
  int len;
  int i;
  len = 0;
  while ($1[len]) len++;
  $result = PyList_New(len);
  for (i = 0; i < len; i++) {
    PyList_SetItem($result, i, PyString_FromString($1[i]));
  }
}


/* The liblldb header files to be included. */

%{
#include "lldb/lldb-types.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBBlock.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandContext.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBCompileUnit.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBFunction.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBSymbol.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBType.h"
#include "lldb/API/SBValue.h"
using namespace lldb_private;
%}

/* Various liblldb typedefs that SWIG needs to know about.  */

%{
typedef unsigned int uint32_t;
typedef int int32_t;
typedef uint32_t tid_t;
typedef uint64_t addr_t;
typedef int32_t break_id_t;
typedef lldb::SBStringList SBStringList;
typedef lldb::RegisterKind RegisterKind;
const RegisterKind kNumRegisterKinds = lldb::kNumRegisterKinds ;
%}

typedef unsigned int uint32_t;
typedef int int32_t;
typedef uint32_t tid_t;
typedef uint64_t addr_t;
typedef int32_t break_id_t;
typedef lldb::SBStringList SBStringList;
typedef lldb::RegisterKind RegisterKind;
const RegisterKind kNumRegisterKinds = lldb::kNumRegisterKinds ;
typedef int StopReason;


%include "lldb/API/SBAddress.h"
%include "lldb/API/SBBlock.h"
%include "lldb/API/SBBreakpoint.h"
%include "lldb/API/SBBreakpointLocation.h"
%include "lldb/API/SBBroadcaster.h"
%include "lldb/API/SBCommandContext.h"
%include "lldb/API/SBCommandInterpreter.h"
%include "lldb/API/SBCommandReturnObject.h"
%include "lldb/API/SBCompileUnit.h"
%include "lldb/API/SBDebugger.h"
%include "lldb/API/SBError.h"
%include "lldb/API/SBEvent.h"
%include "lldb/API/SBFileSpec.h"
%include "lldb/API/SBFrame.h"
%include "lldb/API/SBFunction.h"
%include "lldb/API/SBLineEntry.h"
%include "lldb/API/SBListener.h"
%include "lldb/API/SBModule.h"
%include "lldb/API/SBProcess.h"
%include "lldb/API/SBSourceManager.h"
%include "lldb/API/SBStringList.h"
%include "lldb/API/SBSymbol.h"
%include "lldb/API/SBSymbolContext.h"
%include "lldb/API/SBTarget.h"
%include "lldb/API/SBThread.h"
%include "lldb/API/SBType.h"
%include "lldb/API/SBValue.h"
%include "lldb/lldb-types.h"


