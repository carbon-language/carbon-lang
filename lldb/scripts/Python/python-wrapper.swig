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
    if (!name)
        return pmodule;

    PyErr_Cleaner pyerr_cleanup(true);  // show Python errors

    PyObject* main_dict;

    if (!pmodule)
    {
        pmodule = PyImport_AddModule ("__main__");
        if (!pmodule)
            return NULL;
    }

    if (PyType_Check(pmodule))
    {
        main_dict = ((PyTypeObject*)pmodule)->tp_dict;
        if (!main_dict)
            return NULL;
    }
    else if (!PyDict_Check(pmodule))
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
        dest_object = NULL;   
        while (PyDict_Next (main_dict, &pos, &key, &value))
        {
            // We have stolen references to the key and value objects in the dictionary; we need to increment 
            // them now so that Python's garbage collector doesn't collect them out from under us.
            Py_INCREF (key);
            Py_INCREF (value);
            if (strcmp (PyString_AsString (key), name) == 0)
            {
                dest_object = value;
                break;
            }
        }        
        if (!dest_object || dest_object == Py_None)
            return NULL;
        return dest_object;
    }
    else
    {
        size_t len = dot_pos - name;
        std::string piece(name,len);
        pmodule = ResolvePythonName(piece.c_str(), main_dict);
        if (!pmodule)
            return NULL;
        name = dot_pos+1;
        return ResolvePythonName(dot_pos+1,pmodule); // tail recursion.. should be optimized by the compiler
    }
}

static PyObject*
FindSessionDictionary(const char *session_dictionary_name)
{
    return ResolvePythonName(session_dictionary_name, NULL);
}

class PyCallable
{
public:

    operator
    bool ()
    {
        return m_callable != NULL;
    }

    template<typename ...Args>
    PyObject*
    operator () (Args... args)
    {
        return (*this)({SBTypeToSWIGWrapper(args)...});
    }
    
    PyObject*
    operator () (std::initializer_list<PyObject*> args)
    {
        PyObject* retval = NULL;
        PyObject* pargs = PyTuple_New (args.size());
        if (pargs == NULL)
        {
            if (PyErr_Occurred())
                PyErr_Clear();
            return retval;
        }
        size_t idx = 0;
        for (auto arg : args)
        {
            if (!arg)
                return retval;
            PyTuple_SetItem(pargs,idx,arg);
            idx++;
        }
        retval = PyObject_CallObject (m_callable, pargs);
        Py_XDECREF (pargs);
        return retval;
    }
    
    static PyCallable
    FindWithPythonObject (PyObject* pfunc)
    {
        return PyCallable(pfunc);
    }
    
    static PyCallable
    FindWithFunctionName (const char *python_function_name,
                          const char *session_dictionary_name)
    {
        if (!python_function_name || !session_dictionary_name)
            return PyCallable();
        if ( (python_function_name[0] == 0) || (session_dictionary_name[0] == 0) )
            return PyCallable();
        return FindWithFunctionName(python_function_name,FindSessionDictionary (session_dictionary_name));
    }
    
    static PyCallable
    FindWithFunctionName (const char *python_function_name,
                          PyObject *session_dict)
    {
        if (!python_function_name || !session_dict)
            return PyCallable();
        if ( (python_function_name[0] == 0))
            return PyCallable();
        return PyCallable(ResolvePythonName (python_function_name, session_dict));
    }

    static PyCallable
    FindWithMemberFunction (PyObject *self,
                            const char *python_function_name)
    {
        if (self == NULL || self == Py_None)
            return PyCallable();
        if (!python_function_name || (python_function_name[0] == 0))
            return PyCallable();
        return PyCallable(PyObject_GetAttrString(self, python_function_name));
    }

private:
    PyObject* m_callable;
    
    PyCallable (PyObject *callable = NULL) :
    m_callable(callable)
    {
        if (m_callable && PyCallable_Check(m_callable) == false)
            m_callable = NULL;
    }
};

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

    {
        PyErr_Cleaner py_err_cleaner(true);
        
        PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,session_dictionary_name);
        
        if (!pfunc)
            return stop_at_breakpoint;
        
        PyObject* session_dict = NULL;
        PyObject* pvalue = NULL;
        pvalue = pfunc(sb_frame, sb_bp_loc, session_dict = FindSessionDictionary(session_dictionary_name));
        
        Py_XINCREF (session_dict);
        
        if (pvalue == Py_False)
            stop_at_breakpoint = false;

        Py_XDECREF (pvalue);
    }
    
    return stop_at_breakpoint;
}

// This function is called by lldb_private::ScriptInterpreterPython::WatchpointCallbackFunction(...)
// and is used when a script command is attached to a watchpoint for execution.

SWIGEXPORT bool
LLDBSwigPythonWatchpointCallbackFunction 
(
    const char *python_function_name,
    const char *session_dictionary_name,
    const lldb::StackFrameSP& frame_sp, 
    const lldb::WatchpointSP& wp_sp
)
{
    lldb::SBFrame sb_frame (frame_sp);
    lldb::SBWatchpoint sb_wp(wp_sp);

    bool stop_at_watchpoint = true;
    
    {
        PyErr_Cleaner py_err_cleaner(true);
        
        PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,session_dictionary_name);
        
        if (!pfunc)
            return stop_at_watchpoint;
        
        PyObject* session_dict = NULL;
        PyObject* pvalue = NULL;
        pvalue = pfunc(sb_frame, sb_wp, session_dict = FindSessionDictionary(session_dictionary_name));
        
        Py_XINCREF (session_dict);
        
        if (pvalue == Py_False)
            stop_at_watchpoint = false;
        
        Py_XDECREF (pvalue);
    }
    
    return stop_at_watchpoint;
}

bool
PyObjectToString (PyObject* object,
                  std::string& retval)
{
    retval.clear();
    bool was_ok = false;
    if (object != NULL && object != Py_None)
    {
        if (PyString_Check(object))
        {
            retval.assign(PyString_AsString(object));
            was_ok = true;
        }
        else
        {
            PyObject* value_as_string = PyObject_Str(object);
            if (value_as_string && value_as_string != Py_None && PyString_Check(value_as_string))
            {
                retval.assign(PyString_AsString(value_as_string));
                was_ok = true;
            }
            Py_XDECREF(value_as_string);
        }
    }
    return was_ok;
}

SWIGEXPORT bool
LLDBSwigPythonCallTypeScript 
(
    const char *python_function_name,
    const void *session_dictionary,
    const lldb::ValueObjectSP& valobj_sp,
    void** pyfunct_wrapper,
    std::string& retval
)
{
    lldb::SBValue sb_value (valobj_sp);

    retval.clear();

    if (!python_function_name || !session_dictionary)
        return false;

    PyObject *session_dict = (PyObject*)session_dictionary, *pfunc_impl = NULL, *pargs = NULL, *pvalue = NULL;
    
    if (pyfunct_wrapper && *pyfunct_wrapper && PyFunction_Check (*pyfunct_wrapper))
    {
        pfunc_impl = (PyObject*)(*pyfunct_wrapper);
        if (pfunc_impl->ob_refcnt == 1)
        {
            Py_XDECREF(pfunc_impl);
            pfunc_impl = NULL;
        }
    }

    if (PyDict_Check(session_dict))
    {
        PyErr_Cleaner pyerr_cleanup(true);  // show Python errors
        
        if (!pfunc_impl)
        {
            pfunc_impl = ResolvePythonName (python_function_name, session_dict);
            if (!pfunc_impl || !PyCallable_Check (pfunc_impl))
                return false;
            else
            {
                if (pyfunct_wrapper)
                    *pyfunct_wrapper = pfunc_impl;
            }
        }
        /*else
            printf("caching works!!!!\n");*/
        
        PyCallable pfunc = PyCallable::FindWithPythonObject(pfunc_impl);
        
        if (!pfunc)
            return false;
        
        pvalue = pfunc(sb_value,session_dict);

        Py_INCREF (session_dict);
        
        PyObjectToString(pvalue,retval);
        
        Py_XDECREF (pvalue);
    }
    return true;
}

SWIGEXPORT void*
LLDBSwigPythonCreateSyntheticProvider 
(
    const char *python_class_name,
    const char *session_dictionary_name,
    const lldb::ValueObjectSP& valobj_sp
)
{
    PyObject* retval = NULL;

    if (python_class_name == NULL || python_class_name[0] == '\0' || !session_dictionary_name)
        Py_RETURN_NONE;

    // I do not want the SBValue to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBValue *sb_value = new lldb::SBValue(valobj_sp);
    sb_value->SetPreferSyntheticValue(false);
    PyObject *ValObj_PyObj = SBTypeToSWIGWrapper(sb_value);

    if (ValObj_PyObj == NULL)
        Py_RETURN_NONE;
    
    {
        PyErr_Cleaner py_err_cleaner(true);
        
        PyCallable pfunc = PyCallable::FindWithFunctionName(python_class_name,session_dictionary_name);
        
        if (!pfunc)
            return retval;
        
        Py_INCREF(ValObj_PyObj);
        
        PyObject* session_dict = NULL;
        retval = pfunc(ValObj_PyObj, session_dict = FindSessionDictionary(session_dictionary_name));
        
        Py_XINCREF (session_dict);
        
        Py_XINCREF(retval);
    }
    
    if (retval)
        return retval;
    else
        Py_RETURN_NONE;
}

// wrapper that calls an optional instance member of an object taking no arguments
static PyObject*
LLDBSwigPython_CallOptionalMember
(
    PyObject* self,
    char* callee_name,
    PyObject* ret_if_not_found = Py_None,
    bool* was_found = NULL
)
{
    PyErr_Cleaner py_err_cleaner(false);
    
    PyCallable pfunc = PyCallable::FindWithMemberFunction(self,callee_name);
    
    if (!pfunc)
    {
        if (was_found)
            *was_found = false;
        Py_XINCREF(ret_if_not_found);
        return ret_if_not_found;
    }
    
    if (was_found)
        *was_found = true;
    
    PyObject* py_return = pfunc();
    return py_return;
}

SWIGEXPORT uint32_t
LLDBSwigPython_CalculateNumChildren
(
    PyObject *implementor
)
{
    uint32_t ret_val = UINT32_MAX;

    static char callee_name[] = "num_children";

    PyObject* py_return = LLDBSwigPython_CallOptionalMember(implementor,callee_name, NULL);

    if (!py_return)
        return ret_val;

    if (PyInt_Check(py_return))
        ret_val = PyInt_AsLong(py_return);

    Py_XDECREF(py_return);
    
    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }
    
    return ret_val;
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

    PyErr_Cleaner py_err_cleaner(true);
    
    PyCallable pfunc = PyCallable::FindWithMemberFunction(implementor,"get_child_at_index");
    
    if (!pfunc)
        return NULL;
    
    PyObject *py_return = NULL;
    py_return = pfunc(idx);
    
    if (py_return == NULL || py_return == Py_None)
    {
        Py_XDECREF(py_return);
        return NULL;
    }
    
    lldb::SBValue* sbvalue_ptr = NULL;
    
    if (SWIG_ConvertPtr(py_return, (void**)&sbvalue_ptr, SWIGTYPE_p_lldb__SBValue, 0) == -1)
    {
        Py_XDECREF(py_return);
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
    PyErr_Cleaner py_err_cleaner(true);
    
    PyCallable pfunc = PyCallable::FindWithMemberFunction(implementor,"get_child_index");
    
    if (!pfunc)
        return UINT32_MAX;
    
    PyObject *py_return = NULL;
    py_return = pfunc(child_name);
    
    if (py_return == NULL || py_return == Py_None)
    {
        Py_XDECREF(py_return);
        return UINT32_MAX;
    }
    
    long retval = PyInt_AsLong(py_return);
    Py_XDECREF(py_return);

    if (retval >= 0)
        return (uint32_t)retval;
    
    return UINT32_MAX;
}

SWIGEXPORT bool
LLDBSwigPython_UpdateSynthProviderInstance
(
    PyObject *implementor
)
{
    bool ret_val = false;

    static char callee_name[] = "update";

    PyObject* py_return = LLDBSwigPython_CallOptionalMember(implementor,callee_name);

    if (py_return == Py_True)
        ret_val = true;

    Py_XDECREF(py_return);
    
    return ret_val;
}

SWIGEXPORT bool
LLDBSwigPython_MightHaveChildrenSynthProviderInstance
(
    PyObject *implementor
)
{
    bool ret_val = false;

    static char callee_name[] = "has_children";

    PyObject* py_return = LLDBSwigPython_CallOptionalMember(implementor,callee_name, Py_True);

    if (py_return == Py_True)
        ret_val = true;

    Py_XDECREF(py_return);
    
    return ret_val;
}

SWIGEXPORT void*
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

// Currently, SBCommandReturnObjectReleaser wraps a unique pointer to an
// lldb_private::CommandReturnObject. This means that the destructor for the
// SB object will deallocate its contained CommandReturnObject. Because that
// object is used as the real return object for Python-based commands, we want
// it to stay around. Thus, we release the unique pointer before returning from
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

    {
        PyErr_Cleaner py_err_cleaner(true);
        PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,session_dictionary_name);
        
        if (!pfunc)
            return retval;

        PyObject* session_dict = NULL;
        // pass the pointer-to cmd_retobj_sb or watch the underlying object disappear from under you
        // see comment above for SBCommandReturnObjectReleaser for further details
        PyObject* pvalue = NULL;
        pvalue = pfunc(debugger_sb, args, &cmd_retobj_sb, session_dict = FindSessionDictionary(session_dictionary_name));
        
        Py_XINCREF (session_dict);
        
        if (pvalue != NULL)
        {
            if (pvalue == Py_None) // no error
            {
                err_msg.clear();
                retval = true;
            }
            else
            {
                // return value is an error string
                PyObjectToString(pvalue,err_msg);
                retval = false;
            }
        }
        
        Py_XDECREF (pvalue);
    }
    
    return retval;
}

SWIGEXPORT void*
LLDBSWIGPythonCreateOSPlugin
(
    const char *python_class_name,
    const char *session_dictionary_name,
    const lldb::ProcessSP& process_sp
)
{
    PyObject* retval = NULL;

    if (python_class_name == NULL || python_class_name[0] == '\0' || !session_dictionary_name)
        Py_RETURN_NONE;

    // I do not want the SBProcess to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBProcess *process_sb = new lldb::SBProcess(process_sp);

    PyObject *SBProc_PyObj = SBTypeToSWIGWrapper(process_sb);

    if (SBProc_PyObj == NULL)
        Py_RETURN_NONE;

    {
        PyErr_Cleaner py_err_cleaner(true);
        
        PyCallable pfunc = PyCallable::FindWithFunctionName(python_class_name,session_dictionary_name);
        
        if (!pfunc)
            return retval;
        
        Py_INCREF(SBProc_PyObj);
        
        PyObject* session_dict = NULL;
        retval = pfunc(SBProc_PyObj, session_dict = FindSessionDictionary(session_dictionary_name));
        
        Py_XINCREF (session_dict);
        
        Py_XINCREF(retval);
    }
    
    if (retval)
        return retval;
    else
        Py_RETURN_NONE;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordProcess
(const char* python_function_name,
const char* session_dictionary_name,
lldb::ProcessSP& process,
std::string& output)

{
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBProcess process_sb(process);

    {
        PyErr_Cleaner py_err_cleaner(true);
        
        PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,session_dictionary_name);
        
        if (!pfunc)
            return retval;
        
        PyObject* session_dict = NULL;
        PyObject* pvalue = NULL;
        pvalue = pfunc(process_sb, session_dict = FindSessionDictionary(session_dictionary_name));
        
        Py_XINCREF (session_dict);
        
        if (PyObjectToString(pvalue,output))
            retval = true;
        
        Py_XDECREF(pvalue);
    }
    
    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordThread
(const char* python_function_name,
const char* session_dictionary_name,
lldb::ThreadSP& thread,
std::string& output)

{
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBThread thread_sb(thread);
    
    {
        PyErr_Cleaner py_err_cleaner(true);
        
        PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,session_dictionary_name);
        
        if (!pfunc)
            return retval;
        
        PyObject* session_dict = NULL;
        PyObject* pvalue = NULL;
        pvalue = pfunc(thread_sb, session_dict = FindSessionDictionary(session_dictionary_name));
        
        Py_XINCREF (session_dict);
        
        if (PyObjectToString(pvalue,output))
            retval = true;
        
        Py_XDECREF(pvalue);
    }
    
    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordTarget
(const char* python_function_name,
const char* session_dictionary_name,
lldb::TargetSP& target,
std::string& output)

{
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBTarget target_sb(target);
    
    {
        PyErr_Cleaner py_err_cleaner(true);
        
        PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,session_dictionary_name);
        
        if (!pfunc)
            return retval;
        
        PyObject* session_dict = NULL;
        PyObject* pvalue = NULL;
        pvalue = pfunc(target_sb, session_dict = FindSessionDictionary(session_dictionary_name));
        
        Py_XINCREF (session_dict);
        
        if (PyObjectToString(pvalue,output))
            retval = true;
        
        Py_XDECREF(pvalue);
    }
    
    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordFrame
(const char* python_function_name,
const char* session_dictionary_name,
lldb::StackFrameSP& frame,
std::string& output)

{
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBFrame frame_sb(frame);
    
    {
        PyErr_Cleaner py_err_cleaner(true);
        
        PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,session_dictionary_name);
        
        if (!pfunc)
            return retval;
        
        PyObject* session_dict = NULL;
        PyObject* pvalue = NULL;
        pvalue = pfunc(frame_sb, session_dict = FindSessionDictionary(session_dictionary_name));
        
        Py_XINCREF (session_dict);
        
        if (PyObjectToString(pvalue,output))
            retval = true;
        
        Py_XDECREF(pvalue);
    }
    
    return retval;
}

SWIGEXPORT bool
LLDBSwigPythonCallModuleInit 
(
    const char *python_module_name,
    const char *session_dictionary_name,
    lldb::DebuggerSP& debugger
)
{
    bool retval = false;

    lldb::SBDebugger debugger_sb(debugger);

    std::string python_function_name_string = python_module_name;
    python_function_name_string += ".__lldb_init_module";
    const char* python_function_name = python_function_name_string.c_str();
    
    {
        PyErr_Cleaner py_err_cleaner(true);
        
        PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,session_dictionary_name);
        
        if (!pfunc)
            return true;
        
        PyObject* session_dict = NULL;
        PyObject* pvalue = NULL;
        pvalue = pfunc(debugger_sb, session_dict = FindSessionDictionary(session_dictionary_name));
        
        Py_XINCREF (session_dict);
        
        retval = true;
        
        Py_XDECREF(pvalue);
    }
    
    return retval;
}
%}


%runtime %{
// Forward declaration to be inserted at the start of LLDBWrapPython.h
// I used runtime as a hack to make SWIG place it where it's needed.
// This is needed to use LLDBSwigPythonCallSBInputReaderCallback in the
// typemaps and in the extensions (SBInputReader.__del__()).
#include "lldb/API/SBInputReader.h"
#include "lldb/API/SBDebugger.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t
LLDBSwigPythonCallSBInputReaderCallback(void *baton,
                                        lldb::SBInputReader *reader,
                                        lldb::InputReaderAction notification,
                                        const char*bytes,
                                        size_t bytes_len);

void LLDBSwigPythonCallPythonLogOutputCallback(const char *str, void *baton);

#ifdef __cplusplus
}
#endif
%}

%wrapper %{
// For the InputReader Callback functions
SWIGEXPORT size_t
LLDBSwigPythonCallSBInputReaderCallback(void *baton,
                                        lldb::SBInputReader *reader,
                                        lldb::InputReaderAction notification,
                                        const char*bytes,
                                        size_t bytes_len) {
    if (baton != Py_None) {
        SWIG_PYTHON_THREAD_BEGIN_BLOCK;
    
        PyObject *py_InputReader = SBTypeToSWIGWrapper(reader);
        PyObject *py_Notification = PyInt_FromLong(notification);
        PyObject *py_Bytes = PyBytes_FromStringAndSize(bytes, bytes_len);
    
        PyObject *tuple = PyTuple_Pack(3, py_InputReader, py_Notification, py_Bytes);
        PyObject *res = PyObject_Call(reinterpret_cast<PyObject*>(baton), tuple, NULL);
        Py_XDECREF(tuple);
        Py_XDECREF(py_InputReader);
        Py_XDECREF(py_Notification);
        Py_XDECREF(py_Bytes);
    
        if (res == NULL) {
          PyObject *exc = PyErr_Occurred();
          if (exc) {
            ::puts("\nErroring out at LLDBSwigPythonCallSBInputReaderCallback");
            PyErr_Print();
          }
          return 0;
        }
    
        size_t result = 0;
        // If the callback misbehaves and returns Py_None, assume it returned 0
        if (res != Py_None)
          result = static_cast<size_t>(PyInt_AsSsize_t(res));
    
        Py_XDECREF(res);
        SWIG_PYTHON_THREAD_END_BLOCK;
        return result;
    }
    return 0;
}

// For the LogOutputCallback functions
void LLDBSwigPythonCallPythonLogOutputCallback(const char *str, void *baton) {
    if (baton != Py_None) {
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      PyObject_CallFunction(reinterpret_cast<PyObject*>(baton), const_cast<char*>("s"), str);
      SWIG_PYTHON_THREAD_END_BLOCK;
    }
}
%}
