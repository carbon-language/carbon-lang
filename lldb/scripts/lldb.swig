/*
   lldb.swig

   This is the input file for SWIG, to create the appropriate C++ wrappers and
   functions for various scripting languages, to enable them to call the
   liblldb Script Bridge functions.
*/

/* Define our module docstring. */
%define DOCSTRING
"The lldb module contains the public APIs for Python binding.

Some of the important classes are describe here:

o SBTarget: Represents the target program running under the debugger.
o SBProcess: Represents the process associated with the target program.
o SBThread: Represents a thread of execution. SBProcess contains SBThread(s).
o SBFrame: Represents one of the stack frames associated with a thread. SBThread
      contains SBFrame(s).
o SBSymbolContext: A container that stores various debugger related info.
o SBValue: Represents the value of a variable, a register, or an expression.
o SBModule: Represents an executable image and its associated object and symbol
      files.  SBTarget conatins SBModule(s).
o SBBreakpoint: Represents a logical breakpoint and its associated settings.
      SBTarget conatins SBBreakpoint(s).
o SBSymbol: Represents the symbol possibly associated with a stack frame.
o SBCompileUnit: Represents a compilation unit, or compiled source file.
o SBFunction: Represents a generic function, which can be inlined or not.
o SBBlock: Represents a lexical block. SBFunction contains SBBlock(s).
o SBLineEntry: Specifies an association with a contiguous range of instructions
      and a source file location. SBCompileUnit contains SBLineEntry(s)."
%enddef

/*  The name of the module to be created.  */
%module(docstring=DOCSTRING) lldb


/* Typemap definitions, to allow SWIG to properly handle 'char**' data types. */

%typemap(in) char ** {
  /* Check if is a list  */
  if (PyList_Check($input)) {
    int size = PyList_Size($input);
    int i = 0;
    $1 = (char **) malloc((size+1) * sizeof(char*));
    for (i = 0; i < size; i++) {
      PyObject *o = PyList_GetItem($input,i);
      if (PyString_Check(o))
        $1[i] = PyString_AsString(o);
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

/* Typemap definitions to allow SWIG to properly handle char buffer. */

// typemap for a char buffer
// See also SBThread::GetStopDescription.
%typemap(in) (char *dst, size_t dst_len) {
   if (!PyInt_Check($input)) {
       PyErr_SetString(PyExc_ValueError, "Expecting an integer");
       return NULL;
   }
   $2 = PyInt_AsLong($input);
   if ($2 <= 0) {
       PyErr_SetString(PyExc_ValueError, "Positive integer expected");
       return NULL;
   }
   $1 = (char *) malloc($2);
}

// Return the char buffer.  Discarding any previous return result
// See also SBThread::GetStopDescription.
%typemap(argout) (char *dst, size_t dst_len) {
   Py_XDECREF($result);   /* Blow away any previous result */
   $result = PyString_FromStringAndSize(($1),result);
   free($1);
}


// typemap for an outgoing buffer
// See also SBEvent::SBEvent(uint32_t event, const char *cstr, uint32_t cstr_len).
%typemap(in) (const char *cstr, uint32_t cstr_len) {
   if (!PyString_Check($input)) {
       PyErr_SetString(PyExc_ValueError, "Expecting a string");
       return NULL;
   }
   $1 = (char *) PyString_AsString($input);
   $2 = PyString_Size($input);
}
// And SBProcess::WriteMemory.
%typemap(in) (const void *buf, size_t size) {
   if (!PyString_Check($input)) {
       PyErr_SetString(PyExc_ValueError, "Expecting a string");
       return NULL;
   }
   $1 = (void *) PyString_AsString($input);
   $2 = PyString_Size($input);
}

// typemap for an incoming buffer
// See also SBProcess::ReadMemory.
%typemap(in) (void *buf, size_t size) {
   if (!PyInt_Check($input)) {
       PyErr_SetString(PyExc_ValueError, "Expecting an integer");
       return NULL;
   }
   $2 = PyInt_AsLong($input);
   if ($2 <= 0) {
       PyErr_SetString(PyExc_ValueError, "Positive integer expected");
       return NULL;
   }
   $1 = (void *) malloc($2);
}

// Return the buffer.  Discarding any previous return result
// See also SBProcess::ReadMemory.
%typemap(argout) (void *buf, size_t size) {
   Py_XDECREF($result);   /* Blow away any previous result */
   $result = PyString_FromStringAndSize(static_cast<const char*>($1),result);
   free($1);
}

/* The liblldb header files to be included. */

%{
#include "lldb/lldb-public.h"
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
LLDBSwigPythonBreakpointCallbackFunction 
(
    const char *python_function_name,
    const char *session_dictionary_name,
    const lldb::StackFrameSP& frame_sp, 
    const lldb::BreakpointLocationSP& bp_loc_sp
)
{
    lldb::SBFrame sb_frame (frame_sp);
    lldb::SBBreakpointLocation sb_bp_loc(bp_loc_sp);

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

SWIGEXPORT std::string
LLDBSwigPythonCallTypeScript 
(
    const char *python_function_name,
    const char *session_dictionary_name,
    const lldb::ValueObjectSP& valobj_sp
)
{
    lldb::SBValue sb_value (valobj_sp);

    std::string retval = "";

    PyObject *ValObj_PyObj = SWIG_NewPointerObj((void *) &valobj_sp, SWIGTYPE_p_lldb__SBValue, 0);
    
    if (ValObj_PyObj == NULL)
        return retval;
        
    if (!python_function_name || !session_dictionary_name)
        return retval;

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
                return retval;
                
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
                pargs = PyTuple_New (2);
                if (pargs == NULL)
                {
                    if (PyErr_Occurred())
                        PyErr_Clear();
                    return retval;
                }
                
                PyTuple_SetItem (pargs, 0, ValObj_PyObj);  // This "steals" a reference to ValObj_PyObj
                PyTuple_SetItem (pargs, 1, session_dict); // This "steals" a reference to session_dict
                pvalue = PyObject_CallObject (pfunc, pargs);
                Py_DECREF (pargs);
                
                if (pvalue != NULL)
                {
                    if (pvalue != Py_None)
                        retval = std::string(PyString_AsString(pvalue));
                    else
                        retval = "None";
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
    return retval;
}


%}
