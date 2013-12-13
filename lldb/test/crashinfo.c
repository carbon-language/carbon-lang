/******************************************************************************
                     The LLVM Compiler Infrastructure

  This file is distributed under the University of Illinois Open Source
  License. See LICENSE.TXT for details.
 ******************************************************************************

* This C file vends a simple interface to set the Application Specific Info
* on Mac OS X through Python. To use, compile as a dylib, import crashinfo
* and call crashinfo.setCrashReporterDescription("hello world")
* The testCrashReporterDescription() API is simply there to let you test that this
* is doing what it is intended to do without having to actually cons up a crash
******************************************************************************/

#include <Python/Python.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

void *__crashreporter_info__ = NULL;

asm(".desc ___crashreporter_info__, 0x10");

static PyObject* setCrashReporterDescription(PyObject* self, PyObject* string)
{
	if (__crashreporter_info__)
	{
		free(__crashreporter_info__);
		__crashreporter_info__ = NULL;
	}
		
	if (string && PyString_Check(string))
	{
		Py_ssize_t size = PyString_Size(string);
		char* data = PyString_AsString(string);
		if (size > 0 && data)
		{
            ++size; // Include the NULL terminateor in allocation and memcpy()
			__crashreporter_info__ = malloc(size);
			memcpy(__crashreporter_info__, data, size);
			return Py_True;
		}
	}
	return Py_False;
}

static PyObject* testCrashReporterDescription(PyObject*self, PyObject* arg)
{
	int* ptr = 0;
	*ptr = 1;
	return Py_None;
}

static PyMethodDef crashinfo_methods[] = {
	{"setCrashReporterDescription", setCrashReporterDescription, METH_O},
	{"testCrashReporterDescription", testCrashReporterDescription, METH_O},
	{NULL, NULL}
};

void initcrashinfo()
{
	(void) Py_InitModule("crashinfo", crashinfo_methods);
}

