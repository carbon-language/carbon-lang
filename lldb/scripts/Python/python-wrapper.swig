%header %{

template <typename T>
PyObject *
SBTypeToSWIGWrapper (T* item);

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
            if(m_print && !PyErr_ExceptionMatches(PyExc_SystemExit))
                PyErr_Print();
            PyErr_Clear();
        }
    }

private:
    bool m_print;
};

// TODO(zturner): This entire class should be moved to PythonDataObjects.h
// and properly abstracted and unit-tested.
class PyCallable
{
public:
    struct argc {
        size_t num_args;
        bool varargs : 1;
        bool kwargs : 1;
    };

    argc
    GetNumArguments ()
    {
        PyObject *py_func_obj = NULL;
        if (m_callable)
        {
            if (PyMethod_Check(m_callable))
                py_func_obj = PyMethod_GET_FUNCTION(m_callable);
            else
                py_func_obj = m_callable;
        }

        if (py_func_obj)
        {
            PyCodeObject* code = (PyCodeObject*)PyFunction_GET_CODE(py_func_obj);
            if (code)
            {
                size_t args = code->co_argcount;
                bool va=false,kw=false;
                if ((code->co_flags & 4) == 4)
                    va = true;
                if ((code->co_flags & 8) == 8)
                    kw = true;
                return {args,va,kw};
            }
        }
        return {SIZE_MAX,false,false};
    }

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
            Py_INCREF(arg); // _SetItem steals a reference
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

        using namespace lldb_private;
        auto dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
        return FindWithFunctionName(python_function_name, dict.get());
    }

    static PyCallable
    FindWithFunctionName (const char *python_function_name,
                          PyObject *py_session_dict)
    {
        if (!python_function_name || !py_session_dict)
            return PyCallable();
        if ( (python_function_name[0] == 0))
            return PyCallable();

        using namespace lldb_private;
        PythonDictionary session_dict(PyRefType::Borrowed, py_session_dict);
        PythonCallable result = PythonObject::ResolveNameWithDictionary(python_function_name, session_dict).AsType<PythonCallable>();
        return PyCallable(result.release());
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

%}

%wrapper %{

// resolve a dotted Python name in the form
// foo.bar.baz.Foobar to an actual Python object
// if pmodule is NULL, the __main__ module will be used
// as the starting point for the search


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
    using namespace lldb_private;
    lldb::SBFrame sb_frame (frame_sp);
    lldb::SBBreakpointLocation sb_bp_loc(bp_loc_sp);

    bool stop_at_breakpoint = true;

    PyErr_Cleaner py_err_cleaner(true);
    auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(session_dictionary_name);
    auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(python_function_name, dict);

    if (!pfunc.IsAllocated())
        return stop_at_breakpoint;

    PythonObject frame_arg(PyRefType::Owned, SBTypeToSWIGWrapper(sb_frame));
    PythonObject bp_loc_arg(PyRefType::Owned, SBTypeToSWIGWrapper(sb_bp_loc));
    PythonObject result = pfunc(frame_arg, bp_loc_arg, dict);

    if (result.get() == Py_False)
        stop_at_breakpoint = false;

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
    using namespace lldb_private;
    lldb::SBFrame sb_frame (frame_sp);
    lldb::SBWatchpoint sb_wp(wp_sp);

    bool stop_at_watchpoint = true;

    PyErr_Cleaner py_err_cleaner(true);

    auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(session_dictionary_name);
    auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(python_function_name, dict);

    if (!pfunc.IsAllocated())
        return stop_at_watchpoint;

    PythonObject frame_arg(PyRefType::Owned, SBTypeToSWIGWrapper(sb_frame));
    PythonObject wp_arg(PyRefType::Owned, SBTypeToSWIGWrapper(sb_wp));
    PythonObject result = pfunc(frame_arg, wp_arg, dict);

    if (result.get() == Py_False)
        stop_at_watchpoint = false;

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
    const lldb::TypeSummaryOptionsSP& options_sp,
    std::string& retval
)
{
    using namespace lldb_private;
    lldb::SBValue sb_value (valobj_sp);
    lldb::SBTypeSummaryOptions sb_options(options_sp.get());

    retval.clear();

    if (!python_function_name || !session_dictionary)
        return false;

    PyObject *pfunc_impl = nullptr;

    if (pyfunct_wrapper && *pyfunct_wrapper && PyFunction_Check (*pyfunct_wrapper))
    {
        pfunc_impl = (PyObject*)(*pyfunct_wrapper);
        if (pfunc_impl->ob_refcnt == 1)
        {
            Py_XDECREF(pfunc_impl);
            pfunc_impl = NULL;
        }
    }

    PyObject *py_dict = (PyObject*)session_dictionary;
    if (!PythonDictionary::Check(py_dict))
        return true;

    PythonDictionary dict(PyRefType::Borrowed, py_dict);

    PyErr_Cleaner pyerr_cleanup(true);  // show Python errors

    PythonCallable pfunc(PyRefType::Borrowed, pfunc_impl);

    if (!pfunc.IsAllocated())
    {
        pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(python_function_name, dict);
        if (!pfunc.IsAllocated())
            return false;

        if (pyfunct_wrapper)
        {
            *pyfunct_wrapper = pfunc.get();
            Py_XINCREF(pfunc.get());
        }
    }

    PythonObject result;
    auto argc = pfunc.GetNumArguments();
    // if the third argument is supported, or varargs are allowed
    PythonObject value_arg(PyRefType::Owned, SBTypeToSWIGWrapper(sb_value));
    PythonObject options_arg(PyRefType::Owned, SBTypeToSWIGWrapper(sb_options));
    if (argc.count == 3 || argc.has_varargs)
        result = pfunc(value_arg,dict,options_arg);
    else
        result = pfunc(value_arg,dict);

    PyObjectToString(result.get(), retval);

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
    using namespace lldb_private;

    if (python_class_name == NULL || python_class_name[0] == '\0' || !session_dictionary_name)
        Py_RETURN_NONE;

    PyErr_Cleaner py_err_cleaner(true);

    auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(session_dictionary_name);
    auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(python_class_name,dict);

    if (!pfunc.IsAllocated())
        Py_RETURN_NONE;

    // I do not want the SBValue to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBValue *sb_value = new lldb::SBValue(valobj_sp);
    sb_value->SetPreferSyntheticValue(false);

    PythonObject val_arg(PyRefType::Owned, SBTypeToSWIGWrapper(sb_value));
    if (!val_arg.IsAllocated())
        Py_RETURN_NONE;

    PythonObject result = pfunc(val_arg, dict);

    if (result.IsAllocated())
        return result.release();

    Py_RETURN_NONE;
}

SWIGEXPORT void*
LLDBSwigPythonCreateCommandObject
(
    const char *python_class_name,
    const char *session_dictionary_name,
    const lldb::DebuggerSP debugger_sp
)
{
    using namespace lldb_private;

    if (python_class_name == NULL || python_class_name[0] == '\0' || !session_dictionary_name)
        Py_RETURN_NONE;

    PyErr_Cleaner py_err_cleaner(true);
    auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(session_dictionary_name);
    auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(python_class_name, dict);

    if (!pfunc.IsAllocated())
        return nullptr;

    lldb::SBDebugger debugger_sb(debugger_sp);
    PythonObject debugger_arg(PyRefType::Owned, SBTypeToSWIGWrapper(debugger_sb));
    PythonObject result = pfunc(debugger_arg, dict);

    if (result.IsAllocated())
        return result.release();

    Py_RETURN_NONE;
}

SWIGEXPORT void*
LLDBSwigPythonCreateScriptedThreadPlan
(
    const char *python_class_name,
    const char *session_dictionary_name,
    const lldb::ThreadPlanSP& thread_plan_sp
)
{
    using namespace lldb_private;

    if (python_class_name == NULL || python_class_name[0] == '\0' || !session_dictionary_name)
        Py_RETURN_NONE;

    // I do not want the SBThreadPlan to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBThreadPlan *tp_value = new lldb::SBThreadPlan(thread_plan_sp);

    PyErr_Cleaner py_err_cleaner(true);

    auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(session_dictionary_name);
    auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(python_class_name, dict);

    if (!pfunc.IsAllocated())
        return nullptr;

    PythonObject tp_arg(PyRefType::Owned, SBTypeToSWIGWrapper(tp_value));

    if (!tp_arg.IsAllocated())
        Py_RETURN_NONE;

    PythonObject result = pfunc(tp_arg, dict);
    // FIXME: At this point we should check that the class we found supports all the methods
    // that we need.

    if (result.IsAllocated())
        return result.release();
    Py_RETURN_NONE;
}

SWIGEXPORT bool
LLDBSWIGPythonCallThreadPlan
(
    void *implementor,
    const char *method_name,
    lldb_private::Event *event,
    bool &got_error
)
{
    using namespace lldb_private;

    got_error = false;

    PyErr_Cleaner py_err_cleaner(false);
    PythonObject self(PyRefType::Borrowed, static_cast<PyObject*>(implementor));
    auto pfunc = self.ResolveName<PythonCallable>(method_name);

    if (!pfunc.IsAllocated())
        return false;

    PythonObject result;
    if (event != nullptr)
    {
        lldb::SBEvent sb_event(event);
        PythonObject event_arg(PyRefType::Owned, SBTypeToSWIGWrapper(sb_event));
        result = pfunc(event_arg);
    }
    else
        result = pfunc();

    if (PyErr_Occurred())
    {
        got_error = true;
        printf ("Return value was neither false nor true for call to %s.\n", method_name);
        PyErr_Print();
        return false;
    }

    if (result.get() == Py_True)
        return true;
    else if (result.get() == Py_False)
        return false;

    // Somebody returned the wrong thing...
    got_error = true;
    printf ("Wrong return value type for call to %s.\n", method_name);
    return false;
}

// wrapper that calls an optional instance member of an object taking no arguments
static PyObject*
LLDBSwigPython_CallOptionalMember
(
    PyObject* implementor,
    char* callee_name,
    PyObject* ret_if_not_found = Py_None,
    bool* was_found = NULL
)
{
    using namespace lldb_private;

    PyErr_Cleaner py_err_cleaner(false);

    PythonObject self(PyRefType::Borrowed, static_cast<PyObject*>(implementor));
    auto pfunc = self.ResolveName<PythonCallable>(callee_name);

    if (!pfunc.IsAllocated())
    {
        if (was_found)
            *was_found = false;
        Py_XINCREF(ret_if_not_found);
        return ret_if_not_found;
    }

    if (was_found)
        *was_found = true;

    PythonObject result = pfunc();
    return result.release();
}

SWIGEXPORT size_t
LLDBSwigPython_CalculateNumChildren
(
    PyObject *implementor,
    uint32_t max
)
{
    using namespace lldb_private;

    PythonObject self(PyRefType::Borrowed, implementor);
    auto pfunc = self.ResolveName<PythonCallable>("num_children");

    if (!pfunc.IsAllocated())
        return 0;

    PythonObject result;
    auto argc = pfunc.GetNumArguments();
    if (argc.count == 1)
        result = pfunc();
    else if (argc.count == 2)
        result = pfunc(PythonInteger(max));

    if (!result.IsAllocated())
        return 0;

    PythonInteger int_result = result.AsType<PythonInteger>();
    if (!int_result.IsAllocated())
        return 0;

    size_t ret_val = int_result.GetInteger();

    if (PyErr_Occurred())
    {
        PyErr_Print();
        PyErr_Clear();
    }

    if (argc.count == 1)
        ret_val = std::min(ret_val, max);

    return ret_val;
}

SWIGEXPORT PyObject*
LLDBSwigPython_GetChildAtIndex
(
    PyObject *implementor,
    uint32_t idx
)
{
    using namespace lldb_private;
    PyErr_Cleaner py_err_cleaner(true);

    PythonObject self(PyRefType::Borrowed, implementor);
    auto pfunc = self.ResolveName<PythonCallable>("get_child_at_index");

    if (!pfunc.IsAllocated())
        return nullptr;

    PythonObject result = pfunc(PythonInteger(idx));

    if (!result.IsAllocated())
        return nullptr;

    lldb::SBValue* sbvalue_ptr = nullptr;
    if (SWIG_ConvertPtr(result.get(), (void**)&sbvalue_ptr, SWIGTYPE_p_lldb__SBValue, 0) == -1)
        return nullptr;

    if (sbvalue_ptr == nullptr)
        return nullptr;

    return result.release();
}

SWIGEXPORT int
LLDBSwigPython_GetIndexOfChildWithName
(
    PyObject *implementor,
    const char* child_name
)
{
    using namespace lldb_private;
    PyErr_Cleaner py_err_cleaner(true);

    PythonObject self(PyRefType::Borrowed, implementor);
    auto pfunc = self.ResolveName<PythonCallable>("get_child_index");

    if (!pfunc.IsAllocated())
        return UINT32_MAX;

    PythonObject result = pfunc(PythonString(child_name));

    if (!result.IsAllocated())
        return UINT32_MAX;

    PythonInteger int_result = result.AsType<PythonInteger>();
    if (!int_result.IsAllocated())
        return UINT32_MAX;

    int64_t retval = int_result.GetInteger();
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

SWIGEXPORT PyObject*
LLDBSwigPython_GetValueSynthProviderInstance
(
    PyObject *implementor
)
{
    PyObject* ret_val = nullptr;

    static char callee_name[] = "get_value";

    PyObject* py_return = LLDBSwigPython_CallOptionalMember(implementor,callee_name, Py_None);

    if (py_return == Py_None || py_return == nullptr)
        ret_val = nullptr;

    lldb::SBValue* sbvalue_ptr = NULL;

    if (SWIG_ConvertPtr(py_return, (void**)&sbvalue_ptr, SWIGTYPE_p_lldb__SBValue, 0) == -1)
        ret_val = nullptr;
    else if (sbvalue_ptr == NULL)
        ret_val = nullptr;
    else
        ret_val = py_return;

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
    lldb_private::CommandReturnObject& cmd_retobj,
    lldb::ExecutionContextRefSP exe_ctx_ref_sp
)
{
    using namespace lldb_private;
    lldb::SBCommandReturnObject cmd_retobj_sb(&cmd_retobj);
    SBCommandReturnObjectReleaser cmd_retobj_sb_releaser(cmd_retobj_sb);
    lldb::SBDebugger debugger_sb(debugger);
    lldb::SBExecutionContext exe_ctx_sb(exe_ctx_ref_sp);

    PyErr_Cleaner py_err_cleaner(true);
    auto dict = PythonModule::MainModule().ResolveName<PythonDictionary>(session_dictionary_name);
    auto pfunc = PythonObject::ResolveNameWithDictionary<PythonCallable>(python_function_name, dict);

    if (!pfunc.IsAllocated())
        return false;

    // pass the pointer-to cmd_retobj_sb or watch the underlying object disappear from under you
    // see comment above for SBCommandReturnObjectReleaser for further details
    auto argc = pfunc.GetNumArguments();
    PythonObject debugger_arg(PyRefType::Owned, SBTypeToSWIGWrapper(debugger_sb));
    PythonObject exe_ctx_arg(PyRefType::Owned, SBTypeToSWIGWrapper(exe_ctx_sb));
    PythonObject cmd_retobj_arg(PyRefType::Owned, SBTypeToSWIGWrapper(&cmd_retobj_sb));

    if (argc.count == 5 || argc.has_varargs)
        pfunc(debugger_arg, PythonString(args), exe_ctx_arg, cmd_retobj_arg, dict);
    else
        pfunc(debugger_arg, PythonString(args), cmd_retobj_arg, dict);

    return true;
}

SWIGEXPORT bool
LLDBSwigPythonCallCommandObject
(
    PyObject *implementor,
    lldb::DebuggerSP& debugger,
    const char* args,
    lldb_private::CommandReturnObject& cmd_retobj,
    lldb::ExecutionContextRefSP exe_ctx_ref_sp
)
{
    using namespace lldb_private;
    lldb::SBCommandReturnObject cmd_retobj_sb(&cmd_retobj);
    SBCommandReturnObjectReleaser cmd_retobj_sb_releaser(cmd_retobj_sb);
    lldb::SBDebugger debugger_sb(debugger);
    lldb::SBExecutionContext exe_ctx_sb(exe_ctx_ref_sp);

    PyErr_Cleaner py_err_cleaner(true);

    PythonObject self(PyRefType::Borrowed, implementor);
    auto pfunc = self.ResolveName<PythonCallable>("__call__");

    if (!pfunc.IsAllocated())
        return false;

    // pass the pointer-to cmd_retobj_sb or watch the underlying object disappear from under you
    // see comment above for SBCommandReturnObjectReleaser for further details
    PythonObject debugger_arg(PyRefType::Owned, SBTypeToSWIGWrapper(debugger_sb));
    PythonObject exe_ctx_arg(PyRefType::Owned, SBTypeToSWIGWrapper(exe_ctx_sb));
    PythonObject cmd_retobj_arg(PyRefType::Owned, SBTypeToSWIGWrapper(&cmd_retobj_sb));

    pfunc(debugger_arg, PythonString(args), exe_ctx_arg, cmd_retobj_arg);

    return true;
}

SWIGEXPORT void*
LLDBSWIGPythonCreateOSPlugin
(
    const char *python_class_name,
    const char *session_dictionary_name,
    const lldb::ProcessSP& process_sp
)
{
    using namespace lldb_private;
    PyObject* retval = NULL;

    if (python_class_name == NULL || python_class_name[0] == '\0' || !session_dictionary_name)
        Py_RETURN_NONE;

    // I do not want the SBProcess to be deallocated when going out of scope because python
    // has ownership of it and will manage memory for this object by itself
    lldb::SBProcess *process_sb = new lldb::SBProcess(process_sp);

    PyObject *SBProc_PyObj = SBTypeToSWIGWrapper(process_sb);

    if (SBProc_PyObj == NULL)
        Py_RETURN_NONE;

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_class_name,dict.get());

    if (!pfunc)
        return retval;

    Py_INCREF(SBProc_PyObj);

    retval = pfunc(SBProc_PyObj);

    Py_XINCREF(retval);

    if (retval)
        return retval;
    else
        Py_RETURN_NONE;
}

SWIGEXPORT void*
LLDBSWIGPython_GetDynamicSetting (void* module, const char* setting, const lldb::TargetSP& target_sp)
{

    if (!module || !setting)
        Py_RETURN_NONE;

    lldb::SBTarget target_sb(target_sp);

    PyObject *pvalue = NULL;

    {
        PyErr_Cleaner py_err_cleaner(true);
        PyCallable pfunc = PyCallable::FindWithFunctionName("get_dynamic_setting",(PyObject *)module);

        if (!pfunc)
            Py_RETURN_NONE;

        pvalue = pfunc(target_sb, setting);
    }

    return pvalue;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordProcess
(const char* python_function_name,
const char* session_dictionary_name,
lldb::ProcessSP& process,
std::string& output)

{
    using namespace lldb_private;
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBProcess process_sb(process);

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    PyObject* pvalue = NULL;
    pvalue = pfunc(process_sb, dict.get());

    if (PyObjectToString(pvalue,output))
        retval = true;

    Py_XDECREF(pvalue);

    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordThread
(const char* python_function_name,
const char* session_dictionary_name,
lldb::ThreadSP& thread,
std::string& output)

{
    using namespace lldb_private;
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBThread thread_sb(thread);

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    PyObject* pvalue = NULL;
    pvalue = pfunc(thread_sb, dict.get());

    if (PyObjectToString(pvalue,output))
        retval = true;

    Py_XDECREF(pvalue);

    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordTarget
(const char* python_function_name,
const char* session_dictionary_name,
lldb::TargetSP& target,
std::string& output)

{
    using namespace lldb_private;
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBTarget target_sb(target);

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    PyObject* pvalue = NULL;
    pvalue = pfunc(target_sb, dict.get());

    if (PyObjectToString(pvalue,output))
        retval = true;

    Py_XDECREF(pvalue);

    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordFrame
(const char* python_function_name,
const char* session_dictionary_name,
lldb::StackFrameSP& frame,
std::string& output)

{
    using namespace lldb_private;
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBFrame frame_sb(frame);

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    PyObject* pvalue = NULL;
    pvalue = pfunc(frame_sb, dict.get());

    if (PyObjectToString(pvalue,output))
        retval = true;

    Py_XDECREF(pvalue);

    return retval;
}

SWIGEXPORT bool
LLDBSWIGPythonRunScriptKeywordValue
(const char* python_function_name,
const char* session_dictionary_name,
lldb::ValueObjectSP& value,
std::string& output)

{
    using namespace lldb_private;
    bool retval = false;

    if (python_function_name == NULL || python_function_name[0] == '\0' || !session_dictionary_name)
        return retval;

    lldb::SBValue value_sb(value);

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return retval;

    PyObject* pvalue = NULL;
    pvalue = pfunc(value_sb, dict.get());

    if (PyObjectToString(pvalue,output))
        retval = true;

    Py_XDECREF(pvalue);

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
    using namespace lldb_private;
    bool retval = false;

    lldb::SBDebugger debugger_sb(debugger);

    std::string python_function_name_string = python_module_name;
    python_function_name_string += ".__lldb_init_module";
    const char* python_function_name = python_function_name_string.c_str();

    PyErr_Cleaner py_err_cleaner(true);

    PythonDictionary dict = PythonModule::MainModule().ResolveName(session_dictionary_name).AsType<PythonDictionary>();
    PyCallable pfunc = PyCallable::FindWithFunctionName(python_function_name,dict.get());

    if (!pfunc)
        return true;

    PyObject* pvalue = NULL;
    pvalue = pfunc(debugger_sb, dict.get());

    retval = true;

    Py_XDECREF(pvalue);

    return retval;
}
%}


%runtime %{
// Forward declaration to be inserted at the start of LLDBWrapPython.h
#include "lldb/API/SBDebugger.h"
#include "lldb/API/SBValue.h"

SWIGEXPORT lldb::ValueObjectSP
LLDBSWIGPython_GetValueObjectSPFromSBValue (void* data)
{
    lldb::ValueObjectSP valobj_sp;
    if (data)
    {
        lldb::SBValue* sb_ptr = (lldb::SBValue *)data;
        valobj_sp = sb_ptr->GetSP();
    }
    return valobj_sp;
}

#ifdef __cplusplus
extern "C" {
#endif

void LLDBSwigPythonCallPythonLogOutputCallback(const char *str, void *baton);

#ifdef __cplusplus
}
#endif
%}

%wrapper %{


// For the LogOutputCallback functions
void LLDBSwigPythonCallPythonLogOutputCallback(const char *str, void *baton) {
    if (baton != Py_None) {
      SWIG_PYTHON_THREAD_BEGIN_BLOCK;
      PyObject_CallFunction(reinterpret_cast<PyObject*>(baton), const_cast<char*>("s"), str);
      SWIG_PYTHON_THREAD_END_BLOCK;
    }
}
%}
