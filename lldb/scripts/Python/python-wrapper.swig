%wrapper %{

class PyErr_Cleaner
{
public:
    PyErr_Cleaner(bool print=false) :
        m_print(print)
    {
    }

    ~PyErr_Cleaner()
    {
        if (PyErr_Occurred())
        {
            if(m_print)
                PyErr_Print();
            PyErr_Clear();
        }
    }

private:
    bool m_print;
};

// resolve a dotted Python name in the form
// foo.bar.baz.Foobar to an actual Python object
// if pmodule is NULL, the __main__ module will be used
// as the starting point for the search

static PyObject*
ResolvePythonName(const char* name,
                  PyObject* pmodule = NULL)
{

    //printf("Resolving %s\n", name);

    if (!name || !name[0])
        return pmodule;

    PyErr_Cleaner pyerr_cleanup(true);  // show Python errors

    PyObject* main_dict;

    if (!pmodule)
    {
        pmodule = PyImport_AddModule ("__main__");
        if (!pmodule)
            return NULL;
    }

    if (!PyDict_Check(pmodule))
    {
        main_dict = PyModule_GetDict (pmodule);
        if (!main_dict)
            return NULL;
    }
    else
        main_dict = pmodule;

    const char* dot_pos = ::strchr(name, '.');

    PyObject *dest_object;
    PyObject *key, *value;
    Py_ssize_t pos = 0;

    if (!dot_pos)
    {
        if (PyDict_Check (main_dict))
        {
            dest_object = NULL;   
            while (PyDict_Next (main_dict, &pos, &key, &value))
            {
                // We have stolen references to the key and value objects in the dictionary; we need to increment 
                // them now so that Python's garbage collector doesn't collect them out from under us.
                Py_INCREF (key);
                Py_INCREF (value);
                //printf("Comparing %s and %s\n", name, PyString_AsString (key));
                if (strcmp (PyString_AsString (key), name) == 0)
                {
                    dest_object = value;
                    break;
                }
            }
        }
        
        if (!dest_object || dest_object == Py_None)
            return NULL;
        return dest_object;
    }
    // foo.bar.ba
    // 0123456789
    // len = 3 - 0 
    size_t len = dot_pos - name;
    std::string piece(name,len);
    dest_object = ResolvePythonName(piece.c_str(), main_dict);
    //printf("Resolved %s to %p\n", piece.c_str(), dest_object);
    if (!dest_object)
        return NULL;
    //printf("Now moving to resolve %s\n", dot_pos+1);
    return ResolvePythonName(dot_pos+1,dest_object); // tail recursion.. should be optimized by the compiler

}

static PyObject*
FindSessionDictionary(const char *session_dictionary_name)
{
    return ResolvePythonName(session_dictionary_name, NULL);
}

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

    PyObject *session_dict, *pfunc;
    PyObject *pargs, *pvalue;
    
    session_dict = FindSessionDictionary (session_dictionary_name);
    if (session_dict != NULL)
    {
        pfunc = ResolvePythonName (python_function_name, session_dict);
        if (pfunc != NULL)
        {
            // Set up the arguments and call the function.
                
            if (PyCallable_Check (pfunc))
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

    PyObject *ValObj_PyObj = SWIG_NewPointerObj((void *) &sb_value, SWIGTYPE_p_lldb__SBValue, 0);
    
    if (ValObj_PyObj == NULL)
        return retval;
        
    if (!python_function_name || !session_dictionary_name)
        return retval;

    PyObject *session_dict, *pfunc;
    PyObject *pargs, *pvalue;
    
    session_dict = FindSessionDictionary (session_dictionary_name);
    if (session_dict != NULL)
    {
        pfunc = ResolvePythonName (python_function_name, session_dict);
        if (pfunc != NULL)
        {
            // Set up the arguments and call the function.
                
            if (PyCallable_Check (pfunc))
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
                    PyErr_Print();
                    PyErr_Clear();
                }
                Py_INCREF (session_dict);
            }
            else if (PyErr_Occurred())
            {
                PyErr_Print();
                PyErr_Clear();
            }
        }
        else if (PyErr_Occurred())
        {
            PyErr_Print();
            PyErr_Clear();
        }
    }
    else if (PyErr_Occurred ())
    {
        PyErr_Print();
        PyErr_Clear ();
    }
    return retval;
}

SWIGEXPORT void*
LLDBSwigPythonCreateSyntheticProvider 
(
    const std::string python_class_name,
    const char *session_dictionary_name,
    const lldb::ValueObjectSP& valobj_sp
)
{
    PyObject* retval = NULL;

    if (python_class_name.empty() || !session_dictionary_name)
        Py_RETURN_NONE;

    // I do not want the SBValue to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBValue *valobj_sb = new lldb::SBValue(valobj_sp);

    PyObject *ValObj_PyObj = SWIG_NewPointerObj((void *)valobj_sb, SWIGTYPE_p_lldb__SBValue, SWIG_POINTER_OWN);

    if (ValObj_PyObj == NULL)
        Py_RETURN_NONE;

    const char* python_function_name = python_class_name.c_str();

    PyObject *session_dict, *pfunc;
    PyObject *pvalue;
    
    session_dict = FindSessionDictionary (session_dictionary_name);
    if (session_dict != NULL)
    {
        pfunc = ResolvePythonName (python_function_name, session_dict);
        if (pfunc != NULL)
        {
            // Set up the arguments and call the function.
                
            if (PyCallable_Check (pfunc))
            {
                PyObject *argList = Py_BuildValue("SS", ValObj_PyObj, session_dict);

                if (PyErr_Occurred ())
                {
                    PyErr_Print();
                    PyErr_Clear();
                    return retval;
                }

                if (argList == NULL)
                {
                    return retval;
                }

                Py_INCREF(ValObj_PyObj);

                pvalue = PyObject_CallObject(pfunc, argList);

                Py_DECREF(argList);

                if (pvalue != NULL)
                {
                    if (pvalue != Py_None)
                        retval = pvalue;
                    else
                    {
                        retval = Py_None;
                        Py_INCREF(retval);
                    }
                }
                else if (PyErr_Occurred ())
                {
                    PyErr_Print();
                    PyErr_Clear();
                }
                Py_INCREF (session_dict);
            }
            else if (PyErr_Occurred())
            {
                PyErr_Print();
                PyErr_Clear();
            }
        }
        else if (PyErr_Occurred())
        {
            PyErr_Print();
            PyErr_Clear();
        }
    }
    else if (PyErr_Occurred ())
    {
        PyErr_Print();
        PyErr_Clear ();
    }
    if (retval)
        return retval;
    else
        Py_RETURN_NONE;
}

/*
these four calls below are meant to support
Python-based synthetic children providers
they essentially mimic the four pure virtual
method calls provided by the frontend class
*/

SWIGEXPORT uint32_t
LLDBSwigPython_CalculateNumChildren
(
    PyObject *implementor
)
{

    static char callee_name[] = "num_children";

    if (implementor == NULL || implementor == Py_None)
        return 0;
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, NULL);
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }

    if (py_return == NULL || py_return == Py_None)
    {
        Py_XDECREF(py_return);
        return UINT32_MAX;
    }
    long retval = PyInt_AsLong(py_return);
    Py_DECREF(py_return);
    if (retval >= 0)
        return (uint32_t)retval;
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    return 0;
}

SWIGEXPORT PyObject*
LLDBSwigPython_GetChildAtIndex
(
    PyObject *implementor,
    uint32_t idx
)
{

    static char callee_name[] = "get_child_at_index";
    static char param_format[] = "i";

    if (implementor == NULL || implementor == Py_None)
        return NULL;
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, param_format, idx);
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    if (py_return == NULL || py_return == Py_None)
    {
        Py_XDECREF(py_return);
        return NULL;
    }
    
    lldb::SBValue* sbvalue_ptr = NULL;

    if (SWIG_ConvertPtr(py_return, (void**)&sbvalue_ptr, SWIGTYPE_p_lldb__SBValue, 0) == -1)
    {
        Py_DECREF(py_return);
        return NULL;
    }
    
    if (sbvalue_ptr == NULL)
        return NULL;

    return py_return;
}

SWIGEXPORT int
LLDBSwigPython_GetIndexOfChildWithName
(
    PyObject *implementor,
    const char* child_name
)
{
    static char callee_name[] = "get_child_index";
    static char param_format[] = "s";

    if (implementor == NULL || implementor == Py_None)
        return 0;
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, param_format, child_name);
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    if (py_return == NULL || py_return == Py_None)
    {
        Py_XDECREF(py_return);
        return UINT32_MAX;
    }
    long retval = PyInt_AsLong(py_return);
    Py_DECREF(py_return);
    if (retval >= 0)
        return (uint32_t)retval;
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    return 0;
}

SWIGEXPORT void
LLDBSwigPython_UpdateSynthProviderInstance
(
    PyObject *implementor
)
{
    static char callee_name[] = "update";

    if (implementor == NULL || implementor == Py_None)
        return;

    // all this code is here because update is optional, so we don't want to bother trying to call it unless it's been def:ined for us
    // other synth provider calls are mandatory, so we want to fail in a very obvious way if they are missing!
    PyObject* pmeth  = PyObject_GetAttrString(implementor, callee_name);

    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }

    if (pmeth == NULL || pmeth == Py_None)
    {
        Py_XDECREF(pmeth);
        return;
    }

    if (PyCallable_Check(pmeth) == 0)
    {
        if (PyErr_Occurred())
        {
            PyErr_Clear();
        }

        Py_XDECREF(pmeth);
        return;
    }

    if (PyErr_Occurred())
    {
        PyErr_Clear();
    }

    Py_XDECREF(pmeth);

    // right now we know this function exists and is callable..
    PyObject* py_return = PyObject_CallMethod(implementor, callee_name, NULL);
    
    // if it fails, print the error but otherwise go on
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }

    Py_XDECREF(py_return);

}

SWIGEXPORT lldb::SBValue*
LLDBSWIGPython_CastPyObjectToSBValue
(
    PyObject* data
)
{
    lldb::SBValue* sb_ptr = NULL;
    
    int valid_cast = SWIG_ConvertPtr(data, (void**)&sb_ptr, SWIGTYPE_p_lldb__SBValue, 0);

    if (valid_cast == -1)
        return NULL;

    return sb_ptr;
}

// Currently, SBCommandReturnObjectReleaser wraps an std::auto_ptr to an
// lldb_private::CommandReturnObject. This means that the destructor for the
// SB object will deallocate its contained CommandReturnObject. Because that
// object is used as the real return object for Python-based commands, we want
// it to stay around. Thus, we release the auto_ptr before returning from
// LLDBSwigPythonCallCommand, and to guarantee that the release will occur no
// matter how we exit from the function, we have a releaser object whose
// destructor does the right thing for us
class SBCommandReturnObjectReleaser
{
public:
    SBCommandReturnObjectReleaser (lldb::SBCommandReturnObject &obj) :
        m_command_return_object_ref (obj)
    {
    }

    ~SBCommandReturnObjectReleaser ()
    {
        m_command_return_object_ref.Release();
    }
private:
    lldb::SBCommandReturnObject &m_command_return_object_ref;
};

SWIGEXPORT bool
LLDBSwigPythonCallCommand 
(
    const char *python_function_name,
    const char *session_dictionary_name,
    lldb::DebuggerSP& debugger,
    const char* args,
    std::string& err_msg,
    lldb_private::CommandReturnObject& cmd_retobj
)
{

    lldb::SBCommandReturnObject cmd_retobj_sb(&cmd_retobj);
    SBCommandReturnObjectReleaser cmd_retobj_sb_releaser(cmd_retobj_sb);
    lldb::SBDebugger debugger_sb(debugger);

    bool retval = false;

    PyObject *DebuggerObj_PyObj = SWIG_NewPointerObj((void *) &debugger_sb, SWIGTYPE_p_lldb__SBDebugger, 0);
    PyObject *CmdRetObj_PyObj = SWIG_NewPointerObj((void *) &cmd_retobj_sb, SWIGTYPE_p_lldb__SBCommandReturnObject, 0);

    if (DebuggerObj_PyObj == NULL)
        return retval;
        
    if (CmdRetObj_PyObj == NULL)
        return retval;

    if (!python_function_name || !session_dictionary_name)
        return retval;

    PyObject *session_dict, *pfunc;
    PyObject *pargs, *pvalue;
    
    session_dict = FindSessionDictionary (session_dictionary_name);
    if (session_dict != NULL)
    {
        pfunc = ResolvePythonName (python_function_name, session_dict);
        if (pfunc != NULL)
        {
            // Set up the arguments and call the function.
                
            if (PyCallable_Check (pfunc))
            {
                pargs = PyTuple_New (4);
                if (pargs == NULL)
                {
                    if (PyErr_Occurred())
                        PyErr_Clear();
                    return retval;
                }
                
                PyTuple_SetItem (pargs, 0, DebuggerObj_PyObj);  // This "steals" a reference to DebuggerObj_PyObj
                PyTuple_SetItem (pargs, 1, PyString_FromString(args));
                PyTuple_SetItem (pargs, 2, CmdRetObj_PyObj);  // This "steals" a reference to CmdRetObj_PyObj
                PyTuple_SetItem (pargs, 3, session_dict); // This "steals" a reference to session_dict
                pvalue = PyObject_CallObject (pfunc, pargs);
                Py_DECREF (pargs);
                
                if (pvalue != NULL)
                {
                    if (pvalue == Py_None) // no error
                    {
                        err_msg.clear();
                        retval = true;
                    }
                    else // return value is an error string
                    {
                        err_msg.assign(PyString_AsString(pvalue));
                        retval = false;
                    }
                    Py_DECREF (pvalue);
                }
                else if (PyErr_Occurred ())
                {
                    PyErr_Print();
                    PyErr_Clear();
                }
                Py_INCREF (session_dict);
            }
            else if (PyErr_Occurred())
            {
                PyErr_Print();
                PyErr_Clear();
            }
        }
        else if (PyErr_Occurred())
        {
            PyErr_Print();
            PyErr_Clear();
        }
    }
    else if (PyErr_Occurred ())
    {
        PyErr_Print();
        PyErr_Clear ();
    }
return retval;
}

#undef RETURN_RETVAL

%}
