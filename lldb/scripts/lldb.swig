/*
   lldb.swig

   This is the input file for SWIG, to create the appropriate C++ wrappers and
   functions for various scripting languages, to enable them to call the
   liblldb Script Bridge functions.

*/

/*  The name of the module to be created.  */

%module lldb

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
#include "lldb/lldb-include.h"
#include "lldb/API/SBAddress.h"
#include "lldb/API/SBBlock.h"
#include "lldb/API/SBBreakpoint.h"
#include "lldb/API/SBBreakpointLocation.h"
#include "lldb/API/SBBroadcaster.h"
#include "lldb/API/SBCommandInterpreter.h"
#include "lldb/API/SBCommandReturnObject.h"
#include "lldb/API/SBCommunication.h"
#include "lldb/API/SBCompileUnit.h"
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBError.h"
#include "lldb/API/SBEvent.h"
#include "lldb/API/SBFileSpec.h"
#include "lldb/API/SBFrame.h"
#include "lldb/API/SBFunction.h"
#include "lldb/API/SBHostOS.h"
#include "lldb/API/SBInputReader.h"
#include "lldb/API/SBInstruction.h"
#include "lldb/API/SBInstructionList.h"
#include "lldb/API/SBLineEntry.h"
#include "lldb/API/SBListener.h"
#include "lldb/API/SBModule.h"
#include "lldb/API/SBProcess.h"
#include "lldb/API/SBSourceManager.h"
#include "lldb/API/SBStream.h"
#include "lldb/API/SBStringList.h"
#include "lldb/API/SBSymbol.h"
#include "lldb/API/SBSymbolContext.h"
#include "lldb/API/SBSymbolContextList.h"
#include "lldb/API/SBTarget.h"
#include "lldb/API/SBThread.h"
#include "lldb/API/SBType.h"
#include "lldb/API/SBValue.h"
#include "lldb/API/SBValueList.h"
%}

/* Various liblldb typedefs that SWIG needs to know about.  */
#define __extension__ /* Undefine GCC keyword to make Swig happy when processing glibc's stdint.h. */
%include <stdint.h>
%include "lldb/lldb-defines.h"
%include "lldb/lldb-enumerations.h"
%include "lldb/lldb-forward.h"
%include "lldb/lldb-forward-rtti.h"
%include "lldb/lldb-types.h"
%include "lldb/API/SBAddress.h"
%include "lldb/API/SBBlock.h"
%include "lldb/API/SBBreakpoint.h"
%include "lldb/API/SBBreakpointLocation.h"
%include "lldb/API/SBBroadcaster.h"
%include "lldb/API/SBCommandInterpreter.h"
%include "lldb/API/SBCommandReturnObject.h"
%include "lldb/API/SBCommunication.h"
%include "lldb/API/SBCompileUnit.h"
%include "lldb/API/SBDebugger.h"
%include "lldb/API/SBError.h"
%include "lldb/API/SBEvent.h"
%include "lldb/API/SBFileSpec.h"
%include "lldb/API/SBFrame.h"
%include "lldb/API/SBFunction.h"
%include "lldb/API/SBHostOS.h"
%include "lldb/API/SBInputReader.h"
%include "lldb/API/SBInstruction.h"
%include "lldb/API/SBInstructionList.h"
%include "lldb/API/SBLineEntry.h"
%include "lldb/API/SBListener.h"
%include "lldb/API/SBModule.h"
%include "lldb/API/SBProcess.h"
%include "lldb/API/SBSourceManager.h"
%include "lldb/API/SBStream.h"
%include "lldb/API/SBStringList.h"
%include "lldb/API/SBSymbol.h"
%include "lldb/API/SBSymbolContext.h"
%include "lldb/API/SBSymbolContextList.h"
%include "lldb/API/SBTarget.h"
%include "lldb/API/SBThread.h"
%include "lldb/API/SBType.h"
%include "lldb/API/SBValue.h"
%include "lldb/API/SBValueList.h"

%include "./Python/python-extensions.swig"


%wrapper %{

// This function is called by lldb_private::ScriptInterpreterPython::BreakpointCallbackFunction(...)
// and is used when a script command is attached to a breakpoint for execution.

SWIGEXPORT bool
LLDBSWIGPythonBreakpointCallbackFunction 
(
    const char *python_function_name,
    const char *session_dictionary_name,
    lldb::SBFrame& sb_frame, 
    lldb::SBBreakpointLocation& sb_bp_loc
)
{
    bool stop_at_breakpoint = true;
    PyObject *Frame_PyObj = SWIG_NewPointerObj((void *) &sb_frame, SWIGTYPE_p_lldb__SBFrame, 0);
    PyObject *Bp_Loc_PyObj = SWIG_NewPointerObj ((void *) &sb_bp_loc, SWIGTYPE_p_lldb__SBBreakpointLocation, 0);
    
    if (Frame_PyObj == NULL || Bp_Loc_PyObj == NULL)
        return stop_at_breakpoint;
        
    if (!python_function_name || !session_dictionary_name)
        return stop_at_breakpoint;

    PyObject *pmodule, *main_dict, *session_dict, *pfunc;
    PyObject *pargs, *pvalue;
    
    pmodule = PyImport_AddModule ("__main__");
    if (pmodule != NULL)
    {
        main_dict = PyModule_GetDict (pmodule);
        if (main_dict != NULL)
        {
            PyObject *key, *value;
            Py_ssize_t pos = 0;
             
            // Find the current session's dictionary in the main module's dictionary.

            if (PyDict_Check (main_dict))
             
            {
                session_dict = NULL;   
                while (PyDict_Next (main_dict, &pos, &key, &value))
                {
                    // We have stolen references to the key and value objects in the dictionary; we need to increment 
                    // them now so that Python's garbage collector doesn't collect them out from under us.
                    Py_INCREF (key);
                    Py_INCREF (value);
                    if (strcmp (PyString_AsString (key), session_dictionary_name) == 0)
                    {
                        session_dict = value;
                        break;
                    }
                }
            }
            
            if (!session_dict || !PyDict_Check (session_dict))
                return stop_at_breakpoint;
                
            // Find the function we need to call in the current session's dictionary.

            pos = 0;
            pfunc = NULL;
            while (PyDict_Next (session_dict, &pos, &key, &value))
            {
                if (PyString_Check (key))
                {
                    // We have stolen references to the key and value objects in the dictionary; we need to increment 
                    // them now so that Python's garbage collector doesn't collect them out from under us.
                    Py_INCREF (key);
                    Py_INCREF (value);
                    if (strcmp (PyString_AsString (key), python_function_name) == 0)
                    {
                        pfunc = value;
                        break;
                    }
                }
            }

            // Set up the arguments and call the function.
                
            if (pfunc && PyCallable_Check (pfunc))
            {
                pargs = PyTuple_New (3);
                if (pargs == NULL)
                {
                    if (PyErr_Occurred())
                        PyErr_Clear();
                    return stop_at_breakpoint;
                }
                
                PyTuple_SetItem (pargs, 0, Frame_PyObj);  // This "steals" a reference to Frame_PyObj
                PyTuple_SetItem (pargs, 1, Bp_Loc_PyObj); // This "steals" a reference to Bp_Loc_PyObj
                PyTuple_SetItem (pargs, 2, session_dict); // This "steals" a reference to session_dict
                pvalue = PyObject_CallObject (pfunc, pargs);
                Py_DECREF (pargs);
                
                if (pvalue != NULL)
                {
                    Py_DECREF (pvalue);
                }
                else if (PyErr_Occurred ())
                {
                    PyErr_Clear();
                }
                Py_INCREF (session_dict);
            }
            else if (PyErr_Occurred())
            {
                PyErr_Clear();
            }
        }
        else if (PyErr_Occurred())
        {
            PyErr_Clear();
        }
    }
    else if (PyErr_Occurred ())
    {
        PyErr_Clear ();
    }
    return stop_at_breakpoint;
}

%}
