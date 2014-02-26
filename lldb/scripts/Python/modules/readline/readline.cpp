#include <stdio.h>
#include "Python.h"

// Python readline module intentionally built to not implement the
// readline module interface. This is meant to work around llvm
// pr18841 to avoid seg faults in the stock Python readline.so linked
// against GNU readline.

static struct PyMethodDef moduleMethods[] =
{
    {0, 0}
};

PyDoc_STRVAR(
    moduleDocumentation,
    "Stub module meant to effectively disable readline support.");

PyMODINIT_FUNC
initreadline(void)
{
    Py_InitModule4(
        "readline",
        moduleMethods,
        moduleDocumentation,
        static_cast<PyObject *>(NULL),
        PYTHON_API_VERSION);
}
